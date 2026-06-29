"""Pydantic models for the secret-store REST contract (memory-cloud v0.39.0).

Field names mirror the server's OpenAPI schema for ``/api/v1/config/secrets``.
These are pure data models (no crypto), so importing them does not require the
``[secret]`` extra. ``status`` fields are kept as plain strings rather than
enums so a future server-side status value does not break deserialization.
"""

from __future__ import annotations

from pydantic import BaseModel


class PubkeyResponse(BaseModel):
    """Recipient pubkey metadata (the pubkey is public; no private material)."""

    id: str
    identity_id: str
    pubkey: str
    fingerprint: str
    label: str | None = None
    status: str  # "pending" | "active" | "revoked"
    created_at: str
    attested_at: str | None = None
    revoked_at: str | None = None


class SecretPutResponse(BaseModel):
    """Result of a put (store a new ciphertext version)."""

    name: str
    version_number: int
    status: str
    rotation_needed: bool


class SecretMetaResponse(BaseModel):
    """Secret metadata — never includes the value."""

    name: str
    status: str
    rotation_needed: bool
    current_version: int | None = None
    grant_count: int
    # Timestamps are nullable: the live server returns null for these on some
    # secrets (the OpenAPI marks them required but untyped). Tolerate null
    # rather than failing deserialization.
    created_at: str | None = None
    updated_at: str | None = None


class SecretValueResponse(BaseModel):
    """Opaque ciphertext returned to a granted caller. The server cannot read it."""

    name: str
    version_number: int
    alg: str
    ciphertext: str
    blob_ref: str | None = None
    recipients_snapshot: list[str]
    rotation_needed: bool
    created_at: str


class AuditVerifyResponse(BaseModel):
    """Tamper-evidence check over the secret-store audit chain."""

    valid: bool
    entries: int | None = None
    head: str | None = None
    broken_at: int | None = None
    reason: str | None = None
