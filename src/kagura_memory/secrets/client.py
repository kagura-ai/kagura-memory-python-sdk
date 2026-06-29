"""REST client for the Kagura zero-knowledge secret store (Issue #216).

Talks to ``/api/v1/config/secrets`` on memory-cloud (v0.39.0). This client is a
pure wire layer — it sends/receives **only armored ciphertext** and public
metadata; it never holds the age private key (that is
:class:`~kagura_memory.secrets.keymanager.KeyManager`'s job) and never sees
plaintext. Construction mirrors :class:`~kagura_memory.resource_client.ResourceClient`
and :class:`~kagura_memory.files_client.FilesClient` so the three REST clients
share one shape.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from .._auth import _AuthSource, _OAuthAuth, _resolve_auth, _StaticAuth
from .._http import SDK_VERSION, base_url_from_mcp, extract_detail, validate_https_url
from ..auth.credentials import KaguraOAuth
from ..exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraSecretError,
    _exc_message,
)
from . import crypto
from .models import (
    AuditVerifyResponse,
    PubkeyResponse,
    SecretMetaResponse,
    SecretPutResponse,
    SecretValueResponse,
)

_BASE = "/api/v1/config/secrets"


class SecretClient:
    """REST client for the secret store's pubkey-registry and secret endpoints.

    All methods may raise:
        KaguraAuthError: Authentication failed (401).
        KaguraNotFoundError: Resource not found (404).
        KaguraConnectionError: Other HTTP/network errors (incl. the 400 the
            server returns when a put's grant set is inconsistent).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        *,
        _oauth: KaguraOAuth | None = None,
        _auth_source: _AuthSource | None = None,
    ) -> None:
        """Initialize with a static API key, or via :meth:`from_mcp_url` for OAuth.

        Args:
            api_key: Kagura API key (Bearer). Required unless ``_oauth`` is given.
            base_url: REST API base URL (without path).
            timeout: Request timeout in seconds.
            _oauth: Private. ``from_mcp_url`` supplies a ``KaguraOAuth`` handler.
            _auth_source: Private. Provenance tag from ``_resolve_auth``.

        Raises:
            ValueError: If neither ``api_key`` nor ``_oauth`` is provided.
        """
        if api_key is None and _oauth is None:
            raise ValueError(
                "SecretClient requires api_key, or use SecretClient.from_mcp_url(...) "
                "to resolve credentials from environment, OAuth profile, or .kagura.json."
            )

        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")
        self.base_url = stripped_url
        self._oauth = _oauth
        self._auth_source = _auth_source
        if _oauth is not None:
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
                auth=_oauth,
            )
        else:
            # ``api_key`` is baked into the header, not stored as an attribute
            # (python.md: "Never store API keys as instance attributes").
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"kagura-memory-sdk/{SDK_VERSION}",
                },
            )

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        *,
        profile: str | None = None,
    ) -> SecretClient:
        """Create a client by resolving credentials from the SDK auth chain.

        Same precedence as the other REST clients: explicit ``api_key`` >
        ``KAGURA_API_KEY`` > OAuth profile > ``.kagura.json``. The REST
        ``base_url`` is derived from the resolved ``mcp_url``.
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)
        return cls._from_resolved_auth(resolved, timeout=timeout)

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
    ) -> SecretClient:
        """Construct from a pre-resolved auth — internal CLI helper."""
        base_url = base_url_from_mcp(resolved.mcp_url.rstrip("/"))
        if isinstance(resolved, _StaticAuth):
            return cls(
                api_key=resolved.api_key,
                base_url=base_url,
                timeout=timeout,
                _auth_source=resolved.source,
            )
        return cls(
            base_url=base_url,
            timeout=timeout,
            _oauth=resolved.oauth,
            _auth_source="oauth",
        )

    async def _request(
        self,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with standard Kagura error mapping."""
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, url, json=json, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                hint = (
                    "Re-run `kagura auth login` or inspect ~/.kagura/credentials.json."
                    if self._oauth is not None
                    else "Check your API key."
                )
                raise KaguraAuthError(f"Authentication failed. {hint}") from e
            if status == 404:
                raise KaguraNotFoundError(extract_detail(e.response) or "Not found") from e
            if status == 403:
                # The server returns 403 (not 404) for a secret the caller can't
                # read, to avoid leaking whether it exists. Give an actionable
                # message rather than a bare "HTTP 403".
                detail = extract_detail(e.response)
                msg = (
                    "Access denied (HTTP 403): you may not have a grant on this secret, "
                    "it may not exist, or you lack permission for this operation."
                )
                if detail:
                    msg = f"{msg} ({detail})"
                raise KaguraConnectionError(msg) from e
            detail = extract_detail(e.response)
            msg = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
            raise KaguraConnectionError(msg) from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e

    # -- pubkey registry -----------------------------------------------------

    async def register_pubkey(self, pubkey: str, label: str | None = None) -> PubkeyResponse:
        """Register a public age recipient (lands in ``pending`` until approved)."""
        body: dict[str, Any] = {"pubkey": pubkey}
        if label is not None:
            body["label"] = label
        response = await self._request("POST", f"{_BASE}/pubkeys", json=body)
        return PubkeyResponse.model_validate(response.json())

    async def list_pubkeys(self) -> list[PubkeyResponse]:
        """List all pubkeys in the workspace (owner/admin view)."""
        response = await self._request("GET", f"{_BASE}/pubkeys")
        return [PubkeyResponse.model_validate(p) for p in response.json()]

    async def list_my_pubkeys(self) -> list[PubkeyResponse]:
        """List the caller's own pubkeys."""
        response = await self._request("GET", f"{_BASE}/pubkeys/me")
        return [PubkeyResponse.model_validate(p) for p in response.json()]

    async def approve_pubkey(self, pubkey_id: str) -> PubkeyResponse:
        """Approve a pending pubkey (owner only; TOFU attestation)."""
        response = await self._request("POST", f"{_BASE}/pubkeys/{pubkey_id}/approve")
        return PubkeyResponse.model_validate(response.json())

    async def revoke_pubkey(self, pubkey_id: str) -> PubkeyResponse:
        """Revoke a pubkey (owner only)."""
        response = await self._request("POST", f"{_BASE}/pubkeys/{pubkey_id}/revoke")
        return PubkeyResponse.model_validate(response.json())

    # -- secrets -------------------------------------------------------------

    async def put_secret(
        self,
        name: str,
        ciphertext: str,
        recipients_snapshot: list[str],
        grant_pubkey_ids: list[str],
    ) -> SecretPutResponse:
        """Store a new ciphertext version (low-level).

        The server enforces ``set(recipients_snapshot) ==
        {fingerprint(pk) for pk in grant_pubkey_ids}`` and returns 400 on
        mismatch. Prefer :meth:`put_secret_for_recipients`, which builds both
        lists consistently from a recipient set.
        """
        body = {
            "name": name,
            "ciphertext": ciphertext,
            "recipients_snapshot": recipients_snapshot,
            "grant_pubkey_ids": grant_pubkey_ids,
        }
        response = await self._request("POST", _BASE, json=body)
        return SecretPutResponse.model_validate(response.json())

    async def list_secrets(self) -> list[SecretMetaResponse]:
        """List secret metadata (never includes values)."""
        response = await self._request("GET", _BASE)
        return [SecretMetaResponse.model_validate(s) for s in response.json()]

    async def fetch_secret(
        self, name: str, version_number: int | None = None
    ) -> SecretValueResponse:
        """Fetch a secret's armored ciphertext (name carried in body for '/' names)."""
        body: dict[str, Any] = {"name": name}
        if version_number is not None:
            body["version_number"] = version_number
        response = await self._request("POST", f"{_BASE}/fetch", json=body)
        return SecretValueResponse.model_validate(response.json())

    async def revoke_grant(self, name: str, recipient_pubkey_id: str) -> SecretMetaResponse:
        """Revoke one recipient's grant on a secret (flags ``rotation_needed``).

        Note: revoke does NOT invalidate copies already fetched — follow with
        :meth:`put_secret_for_recipients` (re-encrypt to the remaining
        recipients) and rotate the upstream credential to truly contain a leak.
        """
        body = {"name": name, "recipient_pubkey_id": recipient_pubkey_id}
        response = await self._request("POST", f"{_BASE}/revoke-grant", json=body)
        return SecretMetaResponse.model_validate(response.json())

    async def verify_audit(self) -> AuditVerifyResponse:
        """Verify the tamper-evident audit chain (owner/admin)."""
        response = await self._request("GET", f"{_BASE}/audit/verify")
        return AuditVerifyResponse.model_validate(response.json())

    # -- high-level orchestration -------------------------------------------

    async def put_secret_for_recipients(
        self,
        name: str,
        plaintext: bytes,
        recipients: list[PubkeyResponse],
    ) -> SecretPutResponse:
        """Encrypt ``plaintext`` to ``recipients`` and store it in one call.

        Enforces the server's grant-consistency invariant client-side (fail
        fast, before the network): every recipient must be ``active`` and carry
        a fingerprint matching its pubkey. ``recipients_snapshot`` and
        ``grant_pubkey_ids`` are derived 1:1 from ``recipients`` so the sets
        match by construction.

        Raises:
            KaguraSecretError: empty recipients, a non-active recipient, or a
                pubkey whose advertised fingerprint != ``sha256(pubkey)``.
        """
        if not recipients:
            raise KaguraSecretError("at least one recipient is required to put a secret")
        for r in recipients:
            if r.status != "active":
                raise KaguraSecretError(
                    f"recipient {r.pubkey} is not active (status={r.status!r}); "
                    "grants require an owner-approved (active) pubkey"
                )
            if crypto.fingerprint(r.pubkey) != r.fingerprint:
                raise KaguraSecretError(
                    f"pubkey/fingerprint mismatch for recipient {r.id} — refusing to "
                    "encrypt to a pubkey whose advertised fingerprint is inconsistent"
                )
        ciphertext = crypto.encrypt(plaintext, [r.pubkey for r in recipients])
        return await self.put_secret(
            name=name,
            ciphertext=ciphertext,
            recipients_snapshot=[r.fingerprint for r in recipients],
            grant_pubkey_ids=[r.id for r in recipients],
        )

    # -- lifecycle -----------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> SecretClient:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        await self.close()
