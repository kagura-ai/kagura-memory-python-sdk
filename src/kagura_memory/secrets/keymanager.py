"""age private-key custody (Issue #216).

The age private key (``AGE-SECRET-KEY-1...``) is the root of trust for the
zero-knowledge secret store: whoever holds it can decrypt every ciphertext ever
shared with the matching recipient. It therefore lives in the **OS keychain**
(``keyring``), never in a repo dotfile, and the SDK is **fail-closed** — if no
secure keychain backend is available, enrollment refuses rather than silently
falling back to a plaintext file on disk.

Custody is abstracted behind the :class:`KeyStore` protocol so callers (and
tests) can inject an alternative backend; the default is :class:`KeyringStore`.

Note: a passphrase-encrypted file tier (``age -p`` / scrypt) for hosts without
a keychain is a deliberate follow-on; the MVP is keychain-or-refuse.
"""

from __future__ import annotations

from typing import Protocol

from ..exceptions import KaguraKeyCustodyError
from . import crypto


class KeyStore(Protocol):
    """Minimal secret-at-rest backend: get / set / delete a named string."""

    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class KeyringStore:
    """:class:`KeyStore` backed by the OS keychain via ``keyring``.

    Refuses to write when the active backend is the no-op ``fail`` keyring
    (e.g. a headless Linux host with no Secret Service) — storing the private
    key only to have ``keyring`` drop it would defeat custody.
    """

    SERVICE = "kagura-secret"

    def get(self, name: str) -> str | None:
        import keyring  # type: ignore[import-not-found]
        import keyring.errors  # type: ignore[import-not-found]

        try:
            return keyring.get_password(self.SERVICE, name)
        except keyring.errors.KeyringError as e:
            # A locked / uninitialized keychain must surface as a custody error,
            # not a bare keyring exception — same contract as set().
            raise KaguraKeyCustodyError(f"failed to read key from keychain: {e}") from e

    def set(self, name: str, value: str) -> None:
        import keyring  # type: ignore[import-not-found]
        import keyring.backends.fail  # type: ignore[import-not-found]
        import keyring.errors  # type: ignore[import-not-found]

        if isinstance(keyring.get_keyring(), keyring.backends.fail.Keyring):
            raise KaguraKeyCustodyError(
                "no OS keychain backend available; refusing to store the age "
                "private key insecurely. Install a keyring backend (e.g. "
                "Secret Service / libsecret on Linux) and retry."
            )
        try:
            keyring.set_password(self.SERVICE, name, value)
        except keyring.errors.KeyringError as e:
            raise KaguraKeyCustodyError(f"failed to store key in keychain: {e}") from e

    def delete(self, name: str) -> None:
        import keyring  # type: ignore[import-not-found]
        import keyring.errors  # type: ignore[import-not-found]

        try:
            keyring.delete_password(self.SERVICE, name)
        except keyring.errors.PasswordDeleteError:
            # Idempotent delete — absent key is success.
            pass


class KeyManager:
    """Generate and custody the per-profile age keypair.

    The private key never leaves the keychain; callers only ever receive the
    public recipient and its fingerprint (for server registration and
    out-of-band TOFU verification).
    """

    def __init__(self, *, profile: str = "default", store: KeyStore | None = None) -> None:
        self._profile = profile
        self._store: KeyStore = store if store is not None else KeyringStore()

    @property
    def _key_name(self) -> str:
        return f"identity:{self._profile}"

    def has_key(self) -> bool:
        """True when a private key is already in custody for this profile."""
        return self._store.get(self._key_name) is not None

    def enroll(self) -> tuple[str, str]:
        """Generate a keypair, store the private key, return ``(recipient, fingerprint)``.

        Raises:
            KaguraKeyCustodyError: if a key already exists for this profile
                (refuses to overwrite) or no secure keychain is available.
        """
        if self.has_key():
            raise KaguraKeyCustodyError(
                f"a key already exists for profile {self._profile!r}; refusing to "
                "overwrite. Use `kagura secret rotate` to roll, or delete first."
            )
        identity, recipient = crypto.generate_keypair()
        self._store.set(self._key_name, identity)
        return recipient, crypto.fingerprint(recipient)

    def get_identity(self) -> str:
        """Return the custodied private key.

        Raises:
            KaguraKeyCustodyError: if no key is enrolled for this profile.
        """
        identity = self._store.get(self._key_name)
        if identity is None:
            raise KaguraKeyCustodyError(
                f"no age key in custody for profile {self._profile!r}; "
                "run `kagura secret keygen` first."
            )
        return identity

    def get_recipient(self) -> str:
        """Return the public ``age1`` recipient derived from the custodied key."""
        return crypto.recipient_from_identity(self.get_identity())

    def fingerprint(self) -> str:
        """Return the sha256 fingerprint of this profile's public recipient."""
        return crypto.fingerprint(self.get_recipient())

    def delete(self) -> None:
        """Remove this profile's private key from custody (idempotent)."""
        self._store.delete(self._key_name)
