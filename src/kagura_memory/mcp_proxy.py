"""``kagura-mcp`` — refresh-aware stdio MCP proxy for Claude Code.

Claude Code's MCP client reads ``.mcp.json`` once at startup and replays the
configured ``Authorization`` header forever — it never refreshes. Baking a
short-lived OAuth ``access_token`` into ``.mcp.json`` therefore 401s silently
after ``expires_at``. This module is the fix: a stdio-mode MCP server that
Claude Code spawns as a child process, which owns ``~/.kagura/credentials.json``
and forwards every MCP request to memory-cloud's HTTP ``/mcp`` endpoint with an
always-fresh bearer token.

Design (see issue #101, gate1 review):

- **Transparent JSON-RPC bridge, not a tool-registering server.** Claude Code
  speaks newline-delimited JSON-RPC on stdin/stdout; we forward each message
  verbatim to the upstream HTTP endpoint and write the upstream response back.
  This is a thin pump rather than the ``mcp`` SDK's tool-registration server
  API, which keeps ``tools/list`` / ``tools/call`` / future methods working
  without enumerating them here.
- **Auth via the existing :class:`KaguraOAuth`** httpx.Auth — it injects a
  fresh token per request and refreshes within the skew window through the
  shared in-process lock. On an upstream ``401`` (token rotated/revoked
  out-of-band, outside the skew window) we force one refresh + retry; if the
  refresh itself fails we return an actionable MCP error pointing at
  ``kagura auth login``.
- **mcp_url comes from the profile** (or ``--server``) explicitly — never via
  ``KaguraClient`` env resolution, which ignores ``KAGURA_MCP_URL`` and would
  silently fall back to the hardcoded cloud URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ._http import SDK_VERSION, validate_https_url
from .auth.credentials import KaguraOAuth, get_shared_state

_PROXY_TIMEOUT_SEC = 60.0
_JSONRPC_INTERNAL_ERROR = -32000


class _Upstream:
    """Forwards JSON-RPC messages to an HTTP MCP endpoint with fresh bearers.

    Owns the upstream ``mcp-session-id`` lifecycle: the value returned by the
    ``initialize`` response header is captured and replayed on every
    subsequent request, mirroring :class:`KaguraClient`'s transport.
    """

    def __init__(self, http: httpx.AsyncClient, mcp_url: str, oauth: KaguraOAuth) -> None:
        self._http = http
        self._mcp_url = mcp_url
        self._oauth = oauth
        self._session_id: str | None = None

    async def forward(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Forward one JSON-RPC message; return the response dict, or ``None``.

        ``None`` means "no response body to write back" — a JSON-RPC
        notification (no ``id``) the upstream acknowledged with ``202`` /
        an empty body. The caller must not emit anything in that case.
        """
        resp = await self._post_with_retry(message)
        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        resp.raise_for_status()
        if resp.status_code == 202 or not resp.content:
            return None
        return resp.json()

    async def _post_with_retry(self, message: dict[str, Any]) -> httpx.Response:
        """POST once; on ``401`` force a token refresh and retry exactly once."""
        headers = {"mcp-session-id": self._session_id} if self._session_id else {}
        resp = await self._http.post(self._mcp_url, json=message, headers=headers)
        if resp.status_code == 401:
            # The per-request KaguraOAuth refresh only fires inside the skew
            # window; a 401 here means the token was rejected anyway, so force
            # a refresh and retry the single request.
            await self._oauth.force_refresh()
            resp = await self._http.post(self._mcp_url, json=message, headers=headers)
        return resp


def _error_response(message: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
    """Build a JSON-RPC error response for ``message``, or ``None`` if it had no id.

    A notification (no ``id``) gets no response even on failure — replying
    would violate JSON-RPC. The error text is made actionable for the common
    refresh-failure case so Claude Code surfaces the re-login hint to the user.
    """
    msg_id = message.get("id")
    if msg_id is None:
        return None
    detail = str(exc)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
        detail = (
            "Kagura authentication failed and the token could not be refreshed. "
            "Run `kagura auth login` to re-authenticate."
        )
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": _JSONRPC_INTERNAL_ERROR, "message": f"kagura-mcp: {detail}"},
    }


async def serve(
    upstream: _Upstream,
    read_line: Callable[[], Awaitable[str]],
    write_line: Callable[[str], None],
) -> None:
    """Run the stdio bridge loop until EOF.

    ``read_line`` returns the next newline-terminated line (``""`` at EOF);
    ``write_line`` emits one serialized JSON-RPC response. Both are injected so
    tests can drive the loop with in-memory streams instead of real stdio.
    """
    while True:
        line = await read_line()
        if line == "":  # EOF
            return
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            # Unparseable input has no id to correlate an error to — drop it.
            continue
        try:
            response = await upstream.forward(message)
        except Exception as exc:  # noqa: BLE001 - bridge must never crash the loop
            response = _error_response(message, exc)
        if response is not None:
            write_line(json.dumps(response))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kagura-mcp",
        description="Refresh-aware stdio MCP proxy for Claude Code (Kagura Memory).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="OAuth profile in ~/.kagura/credentials.json (default: the file's default_profile).",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Override the upstream MCP URL (default: the profile's mcp_url).",
    )
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    state = get_shared_state(profile=args.profile)
    if state is None:
        which = f" --profile {args.profile}" if args.profile else ""
        print(
            f"kagura-mcp: no OAuth profile found in ~/.kagura/credentials.json.\n"
            f"  Run: kagura auth login{which}",
            file=sys.stderr,
        )
        return 1

    mcp_url = args.server or state.credentials.mcp_url
    # Enforce HTTPS (localhost allowed for dev), matching KaguraClient.
    validate_https_url(mcp_url, label="MCP URL")

    oauth = KaguraOAuth(state)
    async with httpx.AsyncClient(
        timeout=_PROXY_TIMEOUT_SEC,
        auth=oauth,
        headers={"User-Agent": f"kagura-mcp/{SDK_VERSION}"},
    ) as http:
        upstream = _Upstream(http, mcp_url, oauth)

        async def read_line() -> str:
            # Run the blocking stdin read off the event loop so concurrent
            # refresh round-trips are not starved.
            return await asyncio.to_thread(sys.stdin.readline)

        def write_line(text: str) -> None:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

        await serve(upstream, read_line, write_line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``kagura-mcp``."""
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
