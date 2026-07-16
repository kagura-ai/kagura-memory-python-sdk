"""REST client for the agent surface (#231, server v0.49.0+).

The ``get_agent_bootstrap`` MCP tool has a REST companion —
``POST /api/v1/agents/{agent_id}/bootstrap`` (RFC-0002 P0-3,
memory-cloud [#1276](https://github.com/kagura-ai/memory-cloud/issues/1276))
— authenticated with ``APIKeyOrSessionUser``, so it accepts agent-bound
member keys: a deployed agent can rehydrate its cognitive state without
opening an MCP session. POST-for-read follows the server's
``POST /api/v1/memory/pinned`` precedent.

Construction, credential resolution, lifecycle, and the base error
mapping live in :class:`~kagura_memory._rest_base.KaguraRestClient`
(#229); this module keeps only the bootstrap wire call.
"""

from __future__ import annotations

import uuid
from typing import Any

from ._rest_base import KaguraRestClient
from .models import AgentBootstrapComponentName, AgentBootstrapResponse


def _normalize_agent_id(agent_id: str) -> str:
    """Return the canonical UUID string, rejecting non-UUIDs before the URL.

    ``uuid.UUID`` tolerates non-canonical spellings (``{braces}``,
    ``urn:uuid:`` prefix, dashless 32-hex); interpolating the RAW input
    would send those to the server and surface as a misleading uniform
    404 — normalize instead of just validating.
    """
    try:
        return str(uuid.UUID(str(agent_id)))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"agent_id must be a UUID, got {agent_id!r}") from exc


class AgentsClient(KaguraRestClient):
    """REST API client for agent bootstrap (memory-cloud v0.49.0+).

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraConnectionError: Invalid arguments (400) or any other
            HTTP/connection error
        KaguraNotFoundError: Agent or context not found (404). The 404 is
            uniform (CWE-639) — nonexistent and not-yours are
            indistinguishable by design, so a 404 does NOT prove the
            agent is absent.
        KaguraQuotaError: Rate limit exceeded (429)
    """

    async def bootstrap(
        self,
        agent_id: str,
        *,
        context_id: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        recall_k: int | None = None,
        pinned_cap: int | None = None,
        upcoming_until: str | None = None,
        include: list[AgentBootstrapComponentName] | None = None,
    ) -> AgentBootstrapResponse:
        """Rehydrate an agent's cognitive state at session start.

        REST companion to :meth:`KaguraClient.get_agent_bootstrap` — same
        arguments, same composed envelope, same fail-soft component
        semantics (see that method for the full contract). Use this
        surface when the caller holds an API key but no MCP session,
        e.g. an agent-bound member key minted for a deployed agent.

        Args:
            agent_id: Agent UUID from the registry (required; sent in the
                URL path).
            context_id: Target context UUID. Omit to use the agent's
                default binding.
            session_id: Opaque correlation id (max 128 chars,
                ``[A-Za-z0-9._-]``).
            query: Recall query (max 1024 chars). Supplying it enables the
                trusted-only recall component; omit to skip recall.
            recall_k: Number of recall results.
            pinned_cap: Override for the pinned-set cap (clamped server-side
                to [1, 1000]).
            upcoming_until: ISO upper bound for upcoming time memories.
            include: Component selector — a subset of ``"pinned"``,
                ``"recall"``, ``"upcoming"``, ``"state"``, ``"policy"``.
                Omit for all components.

        Returns:
            :class:`AgentBootstrapResponse` — the composed envelope.
        """
        body: dict[str, Any] = {}
        if context_id is not None:
            body["context_id"] = context_id
        if session_id is not None:
            body["session_id"] = session_id
        if query is not None:
            body["query"] = query
        if recall_k is not None:
            body["recall_k"] = recall_k
        if pinned_cap is not None:
            body["pinned_cap"] = pinned_cap
        if upcoming_until is not None:
            body["upcoming_until"] = upcoming_until
        if include is not None:
            body["include"] = include
        resp = await self._request(
            "POST",
            f"/api/v1/agents/{_normalize_agent_id(agent_id)}/bootstrap",
            json=body,
        )
        return AgentBootstrapResponse.model_validate(self._json(resp))
