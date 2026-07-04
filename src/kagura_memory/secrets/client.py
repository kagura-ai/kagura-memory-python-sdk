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

from typing import Any
from urllib.parse import quote

import httpx

from .._http import extract_detail
from .._rest_base import KaguraRestClient
from ..exceptions import (
    KaguraConnectionError,
    KaguraError,
    KaguraSecretError,
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


class SecretClient(KaguraRestClient):
    """REST client for the secret store's pubkey-registry and secret endpoints.

    All methods may raise:
        KaguraAuthError: Authentication failed (401).
        KaguraNotFoundError: Resource not found (404).
        KaguraConnectionError: Other HTTP/network errors (incl. the 400 the
            server returns when a put's grant set is inconsistent).
    """

    # ---- KaguraRestClient hooks -----------------------------------------

    def _error_403(
        self,
        e: httpx.HTTPStatusError,
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> KaguraError:
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
        return KaguraConnectionError(msg)

    def _error_429(self, e: httpx.HTTPStatusError) -> KaguraError:
        # Historical mapping preserved (#229 is zero-behavior-change): the
        # secret surface has always rendered 429 through the generic branch.
        return self._generic_error(e)

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

    async def delete_secret(self, name: str) -> None:
        """Hard-delete a secret and all its versions + grants (owner only).

        Wraps ``DELETE /api/v1/config/secrets/{name}`` (memory-cloud v0.41.0).
        The server appends a ``delete`` entry to the tamper-evident audit chain
        **before** removal, so :meth:`verify_audit` still passes afterwards.

        **Cleanup, NOT a security control.** Removing the stored ciphertext does
        not un-share a value a recipient already fetched, nor rotate the live
        upstream credential. To contain a leak, rotate the upstream credential
        first, then delete.

        Args:
            name: Secret name. Slash-containing names (e.g.
                ``cloudflare/api-token``) are addressable — each path segment is
                percent-encoded individually so the ``/`` separators stay
                structural in the URL.

        Raises:
            KaguraNotFoundError: No such secret (404).
            KaguraConnectionError: Caller is not the workspace owner (403, the
                delete is owner-only) or another HTTP/network error.
        """
        # Encode each segment but keep the '/' separators so a name like
        # ``cloudflare/api token`` maps to ``cloudflare/api%20token`` and the
        # server's ``{name:path}`` converter still routes on the slashes.
        encoded = "/".join(quote(segment, safe="") for segment in name.split("/"))
        await self._request("DELETE", f"{_BASE}/{encoded}")

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
