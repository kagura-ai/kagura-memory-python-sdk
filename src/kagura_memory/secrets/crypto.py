"""age (X25519) crypto primitives for the zero-knowledge secret client.

All cryptography is delegated to the audited ``pyrage`` binding (Rust ``age``);
the only thing this module implements itself is the RFC 7468 PEM *armor* codec
(base64 + ``-----BEGIN AGE ENCRYPTED FILE-----`` framing), because ``pyrage``
emits binary age files but the server contract (sdk#216 / memory-cloud#1128)
stores **armored** ciphertext. Armoring is a transport encoding, not crypto.

Contract invariants enforced here:

- ``fingerprint(pubkey) == sha256_hex(pubkey)`` over the ``age1`` string.
- recipients match :data:`RECIPIENT_RE` (``^age1[0-9a-z]{20,110}$``, plain X25519).
- armored ciphertext is ``<=`` :data:`MAX_CIPHERTEXT_BYTES` (262144).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import textwrap
from typing import Any

import pyrage as _pyrage  # type: ignore[import-not-found]

from ..exceptions import KaguraCryptoError

# pyrage (the Rust `age` binding) ships incomplete type stubs that don't declare
# its submodules (``x25519``) or error classes, so static checkers choke on
# otherwise-correct attribute access. Treat the module as ``Any`` — runtime
# behavior is unchanged; only type visibility is relaxed.
pyrage: Any = _pyrage

#: Plain X25519 age recipient (no plugin recipients in MVP). Anchored with
#: ``\Z`` (not ``$``) so a trailing newline does NOT match — ``$`` matches just
#: before a final ``\n``, which would let ``age1...\n`` slip past the validator.
RECIPIENT_RE = re.compile(r"^age1[0-9a-z]{20,110}\Z")

#: Server cap on stored ciphertext (256 KiB). ``blob_ref`` (R2) is future work.
MAX_CIPHERTEXT_BYTES = 262144

_ARMOR_BEGIN = "-----BEGIN AGE ENCRYPTED FILE-----"
_ARMOR_END = "-----END AGE ENCRYPTED FILE-----"


def generate_keypair() -> tuple[str, str]:
    """Generate a fresh X25519 age keypair.

    Returns:
        ``(identity, recipient)`` where ``identity`` is the
        ``AGE-SECRET-KEY-1...`` private key string (to be placed in custody by
        :class:`~kagura_memory.secrets.keymanager.KeyManager`) and ``recipient``
        is the public ``age1...`` string (registered with the server).
    """
    identity = pyrage.x25519.Identity.generate()
    return str(identity), str(identity.to_public())


def recipient_from_identity(identity: str) -> str:
    """Derive the public ``age1`` recipient from an ``AGE-SECRET-KEY-1`` identity.

    Raises:
        KaguraCryptoError: if ``identity`` is not a valid age identity string.
    """
    try:
        return str(pyrage.x25519.Identity.from_str(identity).to_public())
    except pyrage.IdentityError as e:
        raise KaguraCryptoError(f"invalid age identity: {e}") from e


def fingerprint(pubkey: str) -> str:
    """Return the server fingerprint: ``sha256`` hex of the recipient string.

    Mirrors ``hashlib.sha256(pubkey.encode("utf-8")).hexdigest()`` exactly so a
    locally computed value can be checked against ``PubkeyResponse.fingerprint``.
    """
    return hashlib.sha256(pubkey.encode("utf-8")).hexdigest()


def armor_encode(binary: bytes) -> str:
    """Wrap a binary age file in RFC 7468 PEM armor (64-column base64)."""
    body = "\n".join(textwrap.wrap(base64.standard_b64encode(binary).decode("ascii"), 64))
    return f"{_ARMOR_BEGIN}\n{body}\n{_ARMOR_END}\n"


def armor_decode(armored: str) -> bytes:
    """Reverse :func:`armor_encode`. Tolerant of CRLF and surrounding blanks.

    Raises:
        KaguraCryptoError: if the input is not a well-formed armored age file.
    """
    lines = [ln.strip() for ln in armored.strip().splitlines()]
    if len(lines) < 2 or lines[0] != _ARMOR_BEGIN or lines[-1] != _ARMOR_END:
        raise KaguraCryptoError("not an armored age file (missing PEM header/footer)")
    try:
        # b64decode (not standard_b64decode) so we can pass validate=True —
        # non-base64 chars are then rejected, not silently discarded.
        return base64.b64decode("".join(lines[1:-1]), validate=True)
    except (binascii.Error, ValueError) as e:
        raise KaguraCryptoError(f"corrupt armored age body: {e}") from e


def encrypt(plaintext: bytes, recipients: list[str]) -> str:
    """Encrypt ``plaintext`` to one or more age recipients; return armored age.

    Args:
        plaintext: raw secret bytes.
        recipients: public ``age1...`` strings (e.g. granted recipients).

    Raises:
        KaguraCryptoError: empty/malformed recipients, an ``age`` failure, or an
            armored result exceeding :data:`MAX_CIPHERTEXT_BYTES`.
    """
    if not recipients:
        raise KaguraCryptoError("at least one recipient is required to encrypt")
    parsed = []
    for r in recipients:
        if not RECIPIENT_RE.match(r):
            raise KaguraCryptoError(f"malformed age recipient: {r!r}")
        try:
            parsed.append(pyrage.x25519.Recipient.from_str(r))
        except pyrage.RecipientError as e:
            raise KaguraCryptoError(f"invalid age recipient {r!r}: {e}") from e

    try:
        binary = pyrage.encrypt(plaintext, parsed)
    except pyrage.EncryptError as e:
        raise KaguraCryptoError(f"age encryption failed: {e}") from e

    armored = armor_encode(binary)
    size = len(armored.encode("ascii"))
    if size > MAX_CIPHERTEXT_BYTES:
        raise KaguraCryptoError(
            f"ciphertext is {size} bytes, exceeds the {MAX_CIPHERTEXT_BYTES}-byte cap"
        )
    return armored


def decrypt(armored: str, identity: str) -> bytes:
    """Decrypt an armored age ciphertext with an ``AGE-SECRET-KEY-1`` identity.

    Raises:
        KaguraCryptoError: input over the size cap, malformed armor, a bad
            identity string, or a decrypt failure (wrong identity / corrupt
            ciphertext).
    """
    # Inbound cap (symmetric with encrypt) so an oversized blob is rejected
    # before it is fully base64-decoded into memory.
    if len(armored) > MAX_CIPHERTEXT_BYTES:
        raise KaguraCryptoError(
            f"ciphertext is {len(armored)} bytes, exceeds the {MAX_CIPHERTEXT_BYTES}-byte cap"
        )
    binary = armor_decode(armored)
    try:
        ident = pyrage.x25519.Identity.from_str(identity)
    except pyrage.IdentityError as e:
        raise KaguraCryptoError(f"invalid age identity: {e}") from e
    try:
        return pyrage.decrypt(binary, [ident])
    except pyrage.DecryptError as e:
        raise KaguraCryptoError(f"age decryption failed: {e}") from e
