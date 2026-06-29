"""Tests for kagura_memory.secrets.crypto — age wrap/unwrap, armor, fingerprint.

The crypto layer is server-independent: it wraps the audited `pyrage` (Rust
`age`/X25519) binding plus a pure-Python RFC 7468 armor codec. No home-grown
crypto — only base64+PEM framing is ours.
"""

import hashlib

import pytest

from kagura_memory.exceptions import KaguraCryptoError
from kagura_memory.secrets import crypto


@pytest.fixture
def keypair() -> tuple[str, str]:
    """A fresh (identity, recipient) age keypair."""
    return crypto.generate_keypair()


# --- keygen + fingerprint ---------------------------------------------------


def test_generate_keypair_returns_age_strings(keypair):
    identity, recipient = keypair
    assert crypto.RECIPIENT_RE.match(recipient), recipient
    assert identity.startswith("AGE-SECRET-KEY-1")


def test_fingerprint_is_sha256_hex_of_pubkey_string(keypair):
    _, recipient = keypair
    expected = hashlib.sha256(recipient.encode("utf-8")).hexdigest()
    fp = crypto.fingerprint(recipient)
    assert fp == expected
    assert len(fp) == 64


def test_recipient_from_identity_recovers_public_key(keypair):
    identity, recipient = keypair
    assert crypto.recipient_from_identity(identity) == recipient


def test_recipient_from_identity_rejects_bad_identity():
    with pytest.raises(KaguraCryptoError):
        crypto.recipient_from_identity("not-an-identity")


# --- armor codec ------------------------------------------------------------


def test_armor_roundtrip_is_binary_equal():
    binary = b"age-encryption.org/v1\n\x00\x01\x02\xff payload"
    armored = crypto.armor_encode(binary)
    assert armored.startswith("-----BEGIN AGE ENCRYPTED FILE-----")
    assert "-----END AGE ENCRYPTED FILE-----" in armored
    assert crypto.armor_decode(armored) == binary


def test_armor_decode_is_crlf_tolerant():
    binary = b"hello-age"
    armored = crypto.armor_encode(binary).replace("\n", "\r\n")
    assert crypto.armor_decode(armored) == binary


def test_armor_decode_rejects_non_armored():
    with pytest.raises(KaguraCryptoError):
        crypto.armor_decode("just some random text, not PEM")


def test_armor_decode_rejects_non_base64_body():
    # `aGVs!!!@@@bG8=` would silently decode to b"hello" under validate=False
    # (the non-alphabet chars are discarded); validate=True must reject it.
    bad = "-----BEGIN AGE ENCRYPTED FILE-----\naGVs!!!@@@bG8=\n-----END AGE ENCRYPTED FILE-----\n"
    with pytest.raises(KaguraCryptoError):
        crypto.armor_decode(bad)


def test_recipient_re_rejects_trailing_newline(keypair):
    _, recipient = keypair
    assert crypto.RECIPIENT_RE.match(recipient)
    assert not crypto.RECIPIENT_RE.match(recipient + "\n")  # $ would have matched; \Z must not


def test_decrypt_rejects_oversize_input(keypair):
    identity, _ = keypair
    huge = (
        "-----BEGIN AGE ENCRYPTED FILE-----\n"
        + "A" * crypto.MAX_CIPHERTEXT_BYTES
        + "\n-----END AGE ENCRYPTED FILE-----\n"
    )
    # Must be rejected by the inbound size cap (before base64-decoding it all),
    # not merely as a downstream age failure.
    with pytest.raises(KaguraCryptoError, match="cap|exceed"):
        crypto.decrypt(huge, identity)


# --- encrypt / decrypt ------------------------------------------------------


def test_encrypt_returns_armored_age(keypair):
    _, recipient = keypair
    armored = crypto.encrypt(b"db-password", [recipient])
    assert armored.startswith("-----BEGIN AGE ENCRYPTED FILE-----")
    # The armored body must decode to a real age file header.
    assert crypto.armor_decode(armored).startswith(b"age-encryption.org/v1")


def test_encrypt_decrypt_roundtrip(keypair):
    identity, recipient = keypair
    plaintext = b"super-secret-\xf0\x9f\x94\x91-value"
    armored = crypto.encrypt(plaintext, [recipient])
    assert crypto.decrypt(armored, identity) == plaintext


def test_encrypt_multi_recipient_each_decrypts():
    id1, r1 = crypto.generate_keypair()
    id2, r2 = crypto.generate_keypair()
    plaintext = b"shared-secret"
    armored = crypto.encrypt(plaintext, [r1, r2])
    assert crypto.decrypt(armored, id1) == plaintext
    assert crypto.decrypt(armored, id2) == plaintext


def test_encrypt_rejects_malformed_recipient():
    with pytest.raises(KaguraCryptoError):
        crypto.encrypt(b"x", ["not-an-age-recipient"])


def test_encrypt_rejects_empty_recipients():
    with pytest.raises(KaguraCryptoError):
        crypto.encrypt(b"x", [])


def test_decrypt_with_wrong_identity_raises(keypair):
    _, recipient = keypair
    other_identity, _ = crypto.generate_keypair()
    armored = crypto.encrypt(b"secret", [recipient])
    with pytest.raises(KaguraCryptoError):
        crypto.decrypt(armored, other_identity)


def test_encrypt_rejects_oversize_ciphertext(keypair):
    _, recipient = keypair
    # 256 KiB of plaintext armors well past the 262144-byte ciphertext cap.
    big = b"A" * crypto.MAX_CIPHERTEXT_BYTES
    with pytest.raises(KaguraCryptoError):
        crypto.encrypt(big, [recipient])
