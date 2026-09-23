"""REST client for the memory guardrail endpoints (#253, server v0.74.0+).

Covers the ``/api/v1/memory`` tool-guardrail routes (memory-cloud #1619,
#1621), authenticated with ``APIKeyOrSessionUser`` — so a client hook or a
setup script holding only an API key (including an agent-bound key, which
may read guardrails but never author them) can load them without opening an
MCP session:

- ``POST /guardrails`` — the REST twin of the MCP ``load_guardrails`` tool
  (:meth:`KaguraClient.load_guardrails`); same body, same response.
- ``GET /guardrails/digest`` — the rendered tool-triggered set for clients
  without tool hooks: the ``AGENTS.md`` export block or a preview of the MCP
  server ``instructions``, with the ``tool_triggered_version`` in a header.

Construction, credential resolution, lifecycle, and the base error mapping
live in :class:`~kagura_memory._rest_base.KaguraRestClient` (#229); this
module keeps only the wire calls.
"""

from __future__ import annotations

from typing import Any

from ._http import normalize_uuid
from ._rest_base import KaguraRestClient
from .models import GuardrailDigest, GuardrailDigestTarget, GuardrailSet

GUARDRAIL_VERSION_HEADER = "X-Kagura-Guardrails-Tool-Triggered-Version"


class MemoryClient(KaguraRestClient):
    """REST API client for tool guardrails (memory-cloud v0.74.0+).

    Every method works with an API key; a workspace-scoped key is confined to
    its workspace and an agent-bound key to its bindings. An OAuth token's
    REST scope follows the HTTP method instead, so :meth:`load_guardrails` —
    a ``POST``, although it only reads — needs ``memory:write`` (a
    ``kagura auth login --read-only`` token gets a 403), while
    :meth:`get_guardrail_digest` needs only ``memory:read``. A read-only
    OAuth caller loads the set with the MCP
    :meth:`KaguraClient.load_guardrails <kagura_memory.client.KaguraClient.load_guardrails>`,
    where memory reads and writes otherwise live.

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraNotFoundError: Context not found (404). The 404 is uniform
            (CWE-639) — unknown, other-workspace and not-yours contexts are
            indistinguishable by design. A server older than v0.74.0 also
            answers :meth:`get_guardrail_digest` with this 404.
        KaguraConnectionError: Invalid arguments (422), an OAuth token
            without the scope (403), a server older than v0.74.0 answering
            :meth:`load_guardrails` (405), or any other HTTP/connection error
        KaguraResponseError: A 2xx body that does not parse as the result
            model (e.g. a ``GuardrailSet`` missing a truncation flag)
        ValueError: ``context_id`` is not a UUID (raised before any request)
    """

    async def load_guardrails(self, context_id: str, *, cap: int | None = None) -> GuardrailSet:
        """Load a context's guardrail set (``POST /api/v1/memory/guardrails``).

        REST twin of :meth:`KaguraClient.load_guardrails` — same two
        independently capped lanes, same truncation flags (see
        :class:`~kagura_memory.models.GuardrailSet`). Use this surface from a
        hook or script that holds an API key but no MCP session.

        Args:
            context_id: Context UUID whose guardrail set to load.
            cap: Max tool-triggered memories returned (1-1000; the REST route
                rejects anything else with a 422). Bounds the
                ``tool_triggered`` lane only. Omit for the server default (50).

        Returns:
            :class:`~kagura_memory.models.GuardrailSet` (``context_id`` /
            ``context_name`` are ``None`` on this surface).
        """
        body: dict[str, Any] = {"context_id": normalize_uuid(context_id, label="context_id")}
        if cap is not None:
            body["cap"] = cap
        resp = await self._request("POST", "/api/v1/memory/guardrails", json=body)
        return self._parse(GuardrailSet, self._json(resp), "load_guardrails")

    async def get_guardrail_digest(
        self,
        context_id: str,
        *,
        target: GuardrailDigestTarget = "export",
        profile: str | None = None,
        tools: str | None = None,
    ) -> GuardrailDigest:
        """Render a context's tool guardrails (``GET /api/v1/memory/guardrails/digest``).

        For clients without tool hooks (Codex cloud, ChatGPT, Claude Desktop):
        summaries only — never content, details or patterns — from the same
        trusted-only, binding-filtered read as :meth:`load_guardrails`.

        Args:
            context_id: Context UUID whose tool guardrails to render.
            target: ``"export"`` (default) — the ``AGENTS.md`` block as
                ``text/markdown`` (up to 20 entries, ≤ 12,000 characters,
                empty when the context has none); ``"instructions"`` — the
                exact MCP server ``instructions`` string this credential
                would receive for this context.
            profile: ``target="instructions"`` only — the MCP URL's
                ``?profile=`` value (``full`` | ``core``), so the truncation
                suffix names the tool that URL lists.
            tools: ``target="instructions"`` only — the MCP URL's ``?tools=``
                allowlist.

        Returns:
            :class:`~kagura_memory.models.GuardrailDigest` with the body text
            and the ``X-Kagura-Guardrails-Tool-Triggered-Version`` header.
        """
        ctx = normalize_uuid(context_id, label="context_id")
        params: dict[str, Any] = {"context_id": ctx, "target": target}
        if profile is not None:
            params["profile"] = profile
        if tools is not None:
            params["tools"] = tools
        resp = await self._request("GET", "/api/v1/memory/guardrails/digest", params=params)
        return GuardrailDigest(
            context_id=ctx,
            target=target,
            text=resp.text,
            tool_triggered_version=resp.headers.get(GUARDRAIL_VERSION_HEADER),
            content_type=resp.headers.get("content-type"),
        )
