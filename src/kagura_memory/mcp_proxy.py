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
- **The proxy owns the upstream session** (issue #252). When an upstream
  drops the MCP session it answers the next request with ``404`` (MCP
  Streamable HTTP), and Claude Code, which only talks stdio to us, can neither
  see that nor re-initialize. On that ``404`` we replay the downstream's own
  cached ``initialize`` and retry the message once, mirroring the one-shot
  ``401`` refresh. (memory-cloud v0.75.0 as deployed re-adopts an unknown
  session id instead of 404ing; see ``mcp_session_expired``.)
- **Upstream JSON-RPC errors pass through.** A non-2xx whose body is a
  JSON-RPC ``error`` (``-32600``, ``-32022`` …) is forwarded with its own
  code/message/data rather than flattened into ``-32000``. Only the ``401``
  (the proxy's re-login message) and a failed ``initialize`` replay (the
  proxy's own ``-32000``, since the error is not about the message being
  answered) are reported by the proxy itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ._http import (
    SDK_VERSION,
    extract_detail,
    jsonrpc_error_body,
    mcp_session_expired,
    mcp_session_header,
    validate_https_url,
)
from .auth.credentials import KaguraOAuth, get_shared_state

_PROXY_TIMEOUT_SEC = 60.0
_JSONRPC_INTERNAL_ERROR = -32000
_INITIALIZED_NOTIFICATION: dict[str, Any] = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
}


class _Upstream:
    """Forwards JSON-RPC messages to an HTTP MCP endpoint with fresh bearers.

    Owns the upstream ``mcp-session-id`` lifecycle: the value returned by the
    ``initialize`` response header is captured and replayed on every
    subsequent request, mirroring :class:`KaguraClient`'s transport. When the
    upstream drops the session, the downstream's cached ``initialize`` is
    replayed to open a new one.
    """

    def __init__(self, http: httpx.AsyncClient, mcp_url: str, oauth: KaguraOAuth) -> None:
        self._http = http
        self._mcp_url = mcp_url
        self._oauth = oauth
        self._session_id: str | None = None
        # The last ``initialize`` the upstream accepted, kept for replay.
        self._initialize: dict[str, Any] | None = None

    async def forward(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Forward one JSON-RPC message; return the response dict, or ``None``.

        ``None`` means "no response body to write back" — a JSON-RPC
        notification (no ``id``) the upstream acknowledged with ``202`` /
        an empty body, or answered with an error. The caller must not emit
        anything in that case.

        Raises:
            httpx.HTTPStatusError: A ``401`` that survived the forced refresh,
                a non-2xx without a JSON-RPC error body to forward, or a
                replayed ``initialize`` that failed (see :meth:`_reopen_session`).
        """
        is_initialize = message.get("method") == "initialize"
        # ``initialize`` opens a new session; sending the old id with it would
        # 404 once that session had expired.
        session_id = None if is_initialize else self._session_id
        resp = await self._post_with_retry(message, session_id)
        if mcp_session_expired(resp, session_id):
            # Rejected before dispatch, so the one retry is safe for tools/call.
            await self._reopen_session()
            resp = await self._post_with_retry(message, self._session_id)
        if is_initialize and resp.is_success:
            self._initialize = message
        return _downstream_response(message, resp)

    async def _reopen_session(self) -> None:
        """Open a new upstream session in place of one the upstream dropped.

        Replays the downstream's cached ``initialize`` and the
        ``notifications/initialized`` the MCP lifecycle requires after it.
        With nothing cached the stale id is only forgotten, and the retry goes
        without one (memory-cloud then opens a session for it).

        Raises:
            httpx.HTTPStatusError: The replayed ``initialize`` failed.
        """
        self._session_id = None
        if self._initialize is None:
            return
        resp = await self._post_with_retry(self._initialize, None)
        resp.raise_for_status()
        await self._post_with_retry(_INITIALIZED_NOTIFICATION, self._session_id)

    async def _post_with_retry(
        self, message: dict[str, Any], session_id: str | None
    ) -> httpx.Response:
        """POST once in ``session_id``; on ``401`` force a token refresh and retry once.

        Captures the ``mcp-session-id`` the response carries, so the next
        message goes out in the session an ``initialize`` just opened.
        """
        headers = mcp_session_header(session_id)
        resp = await self._http.post(self._mcp_url, json=message, headers=headers)
        if resp.status_code == 401:
            # The per-request KaguraOAuth refresh only fires inside the skew
            # window; a 401 here means the token was rejected anyway, so force
            # a refresh and retry the single request.
            await self._oauth.force_refresh()
            resp = await self._http.post(self._mcp_url, json=message, headers=headers)
        new_session_id = resp.headers.get("mcp-session-id")
        if new_session_id:
            self._session_id = new_session_id
        return resp


def _downstream_response(message: dict[str, Any], resp: httpx.Response) -> dict[str, Any] | None:
    """Turn the upstream reply to ``message`` into the line to write back, if any.

    A non-2xx carrying a JSON-RPC error body is forwarded unchanged — only a
    ``null`` id (the server could not read one) is replaced with the
    downstream's. A ``401`` always raises so :func:`_error_response` can
    attach the re-login hint.

    Raises:
        httpx.HTTPStatusError: A non-2xx with nothing to forward.
    """
    if not resp.is_success and resp.status_code != 401:
        body = jsonrpc_error_body(resp)
        if body is not None:
            msg_id = message.get("id")
            if msg_id is None:  # a notification never gets a reply, not even an error
                return None
            if body.get("id") is None:
                body["id"] = msg_id
            return body
    resp.raise_for_status()
    if resp.status_code == 202 or not resp.content:
        return None
    return resp.json()


def _error_response(message: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
    """Build a JSON-RPC error response for ``message``, or ``None`` if it had no id.

    A notification (no ``id``) gets no response even on failure — replying
    would violate JSON-RPC. The error text is made actionable for the common
    refresh-failure case so Claude Code surfaces the re-login hint to the user;
    any other HTTP error carries the server's own explanation when it sent one
    (e.g. the OAuth-style ``error_description`` of a workspace-URL ``403``).
    """
    msg_id = message.get("id")
    if msg_id is None:
        return None
    detail = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            detail = (
                "Kagura authentication failed and the token could not be refreshed. "
                "Run `kagura auth login` to re-authenticate."
            )
        elif server_detail := extract_detail(exc.response):
            detail = f"HTTP {status}: {server_detail}"
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
        if not isinstance(message, dict):
            # Nor has a batch array (the upstream rejects batches) or a scalar.
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
