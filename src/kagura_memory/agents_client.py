"""REST client for the agent surface (#231/#235, server v0.49.0+).

Covers the ``/api/v1/agents`` namespace (RFC-0002):

- **Bootstrap** (P0-3, memory-cloud
  [#1276](https://github.com/kagura-ai/memory-cloud/issues/1276)):
  ``POST /{agent_id}/bootstrap``, authenticated with
  ``APIKeyOrSessionUser`` — accepts agent-bound member keys, so a
  deployed agent can rehydrate its cognitive state without opening an
  MCP session. POST-for-read follows the server's
  ``POST /api/v1/memory/pinned`` precedent.
- **Registry CRUD** (P0-1, #1274) and **context bindings** (P0-2,
  #1275): owner/admin-gated (``require_workspace_admin``); every
  mutation writes an audit row, with ``enforce``→``shadow`` transitions
  recorded as the distinct privilege-widening event.

Construction, credential resolution, lifecycle, and the base error
mapping live in :class:`~kagura_memory._rest_base.KaguraRestClient`
(#229); this module keeps only the wire calls.
"""

from __future__ import annotations

from typing import Any, Literal

from ._http import normalize_uuid
from ._rest_base import KaguraRestClient
from .models import (
    Agent,
    AgentBinding,
    AgentBootstrapComponentName,
    AgentBootstrapResponse,
    _agent_update_payload,
    _binding_scope_payload,
    _bootstrap_payload,
)


class AgentsClient(KaguraRestClient):
    """REST API client for the agent control plane (memory-cloud v0.49.0+).

    :meth:`bootstrap` works with any ``APIKeyOrSessionUser`` credential
    (including agent-bound member keys); the registry and binding
    methods are **owner/admin-gated** server-side.

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraConnectionError: Invalid arguments (400), insufficient
            role (403), or any other HTTP/connection error
        KaguraNotFoundError: Agent / binding / context not found (404).
            The 404 is uniform (CWE-639) — nonexistent and not-yours are
            indistinguishable by design, so a 404 does NOT prove the
            resource is absent.
        KaguraQuotaError: Rate limit or agent quota exceeded (429)
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
        body = _bootstrap_payload(
            context_id=context_id,
            session_id=session_id,
            query=query,
            recall_k=recall_k,
            pinned_cap=pinned_cap,
            upcoming_until=upcoming_until,
            include=include,
        )
        resp = await self._request(
            "POST",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}/bootstrap",
            json=body,
        )
        return AgentBootstrapResponse.model_validate(self._json(resp))

    # -------------------------------------------------------------------
    # Agent registry (#235, RFC-0002 P0-1 — owner/admin only)
    # -------------------------------------------------------------------

    async def register_agent(
        self,
        name: str,
        *,
        description: str | None = None,
        framework: str | None = None,
        environment: str | None = None,
        version: str | None = None,
    ) -> Agent:
        """Register an agent (``POST /api/v1/agents``, 201; owner/admin only).

        REST companion to :meth:`KaguraClient.register_agent` — same
        contract: workspace-unique name, new agents start
        ``status="active"`` / ``enforcement_mode="enforce"``.
        """
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if framework is not None:
            body["framework"] = framework
        if environment is not None:
            body["environment"] = environment
        if version is not None:
            body["version"] = version
        resp = await self._request("POST", "/api/v1/agents", json=body)
        return Agent.model_validate(self._json(resp))

    async def list_agents(self) -> list[Agent]:
        """List the workspace's agents, newest first (owner/admin only)."""
        resp = await self._request("GET", "/api/v1/agents")
        return [Agent.model_validate(row) for row in self._expect_wrapped_list(resp, "agents")]

    async def get_agent(self, agent_id: str) -> Agent:
        """Fetch one agent by id (owner/admin only; uniform 404)."""
        resp = await self._request(
            "GET", f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}"
        )
        return Agent.model_validate(self._json(resp))

    async def update_agent(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        framework: str | None = None,
        environment: str | None = None,
        version: str | None = None,
        status: Literal["active", "suspended", "retired"] | None = None,
        enforcement_mode: Literal["shadow", "enforce"] | None = None,
    ) -> Agent:
        """Partially update an agent (``PATCH``; owner/admin only).

        ``status`` is the fail-closed kill switch (suspended/retired
        agents get every bound key rejected at verify time);
        ``enforce``->``shadow`` is audited as privilege widening.
        Set-only wrapper: omitted fields stay untouched — the server's
        null-clears semantics is not expressible here (clear via the web
        UI or the raw API).
        """
        body = _agent_update_payload(
            name=name,
            description=description,
            framework=framework,
            environment=environment,
            version=version,
            status=status,
            enforcement_mode=enforcement_mode,
        )
        resp = await self._request(
            "PATCH",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}",
            json=body,
        )
        return Agent.model_validate(self._json(resp))

    async def delete_agent(self, agent_id: str) -> None:
        """Hard-delete an agent (``DELETE``, 204; owner/admin only).

        Permanent; cascades every API key bound to the agent
        (fail-closed). Prefer ``update_agent(status="retired")`` for
        operational retirement.
        """
        await self._request(
            "DELETE", f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}"
        )

    # -------------------------------------------------------------------
    # Context bindings (#235, RFC-0002 P0-2 — owner/admin only)
    # -------------------------------------------------------------------

    async def bind_context(
        self,
        agent_id: str,
        context_id: str,
        *,
        can_read: bool | None = None,
        write_policy: Literal["deny", "direct"] | None = None,
        is_default: bool | None = None,
    ) -> AgentBinding:
        """Bind an agent to a context (``POST .../bindings``, 201).

        Purely subtractive scoping: effective permission = RBAC AND
        binding. Under ``enforcement_mode="enforce"`` unbound contexts
        are default-denied for the agent. Server defaults:
        ``can_read=True``, ``write_policy="deny"``, ``is_default=False``
        (max one default per agent). ``allowed_memory_types`` /
        ``allowed_source_types`` are reserved (#1286) and not exposed.
        """
        body: dict[str, Any] = {
            "context_id": normalize_uuid(context_id, label="context_id"),
            **_binding_scope_payload(
                can_read=can_read, write_policy=write_policy, is_default=is_default
            ),
        }
        resp = await self._request(
            "POST",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}/bindings",
            json=body,
        )
        return AgentBinding.model_validate(self._json(resp))

    async def list_bindings(self, agent_id: str) -> list[AgentBinding]:
        """List an agent's context bindings (owner/admin only)."""
        resp = await self._request(
            "GET",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}/bindings",
        )
        return [
            AgentBinding.model_validate(row) for row in self._expect_wrapped_list(resp, "bindings")
        ]

    async def update_binding(
        self,
        agent_id: str,
        binding_id: str,
        *,
        can_read: bool | None = None,
        write_policy: Literal["deny", "direct"] | None = None,
        is_default: bool | None = None,
    ) -> AgentBinding:
        """Update a binding's scoping fields (``PATCH``; owner/admin only).

        ``context_id`` is immutable — :meth:`unbind_context` and
        re-:meth:`bind_context` to re-target. Changes are audited with
        old->new values.
        """
        body = _binding_scope_payload(
            can_read=can_read, write_policy=write_policy, is_default=is_default
        )
        resp = await self._request(
            "PATCH",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}"
            f"/bindings/{normalize_uuid(binding_id, label='binding_id')}",
            json=body,
        )
        return AgentBinding.model_validate(self._json(resp))

    async def unbind_context(self, agent_id: str, binding_id: str) -> None:
        """Delete a binding (``DELETE``, 204; owner/admin only).

        Under ``enforcement_mode="enforce"`` the agent's requests against
        the unbound context are denied afterwards (uniform 404).
        """
        await self._request(
            "DELETE",
            f"/api/v1/agents/{normalize_uuid(agent_id, label='agent_id')}"
            f"/bindings/{normalize_uuid(binding_id, label='binding_id')}",
        )
