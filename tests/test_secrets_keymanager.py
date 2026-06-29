"""Tests for kagura_memory.secrets.keymanager — keygen + private-key custody.

Custody is exercised through an injected in-memory ``KeyStore`` so tests never
touch the real OS keychain. The KeyringStore availability guard is tested by
monkeypatching ``keyring.get_keyring`` to the fail backend.
"""

import keyring
import keyring.backends.fail
import keyring.errors
import pytest

from kagura_memory.exceptions import KaguraKeyCustodyError
from kagura_memory.secrets import crypto
from kagura_memory.secrets.keymanager import KeyManager, KeyringStore


class FakeStore:
    """In-memory KeyStore for tests."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.data.get(name)

    def set(self, name: str, value: str) -> None:
        self.data[name] = value

    def delete(self, name: str) -> None:
        self.data.pop(name, None)


class UnavailableStore(FakeStore):
    """Simulates a host with no usable keychain (fail-closed)."""

    def set(self, name: str, value: str) -> None:
        raise KaguraKeyCustodyError("no secure keychain available")


def test_enroll_returns_recipient_and_fingerprint_and_stores_identity():
    store = FakeStore()
    km = KeyManager(profile="default", store=store)

    recipient, fp = km.enroll()

    assert crypto.RECIPIENT_RE.match(recipient)
    assert fp == crypto.fingerprint(recipient)
    assert km.has_key()
    # The *private* key is what's in custody, not the public recipient.
    assert next(iter(store.data.values())).startswith("AGE-SECRET-KEY-1")


def test_enroll_refuses_to_overwrite_existing_key():
    store = FakeStore()
    km = KeyManager(profile="default", store=store)
    km.enroll()
    before = dict(store.data)

    with pytest.raises(KaguraKeyCustodyError) as exc:
        km.enroll()

    # Must not suggest `secret rotate` (that rotates a value, not the keypair).
    assert "rotate" not in str(exc.value).lower()
    assert store.data == before  # original key untouched


def test_get_identity_raises_when_absent():
    km = KeyManager(profile="default", store=FakeStore())
    with pytest.raises(KaguraKeyCustodyError):
        km.get_identity()


def test_get_recipient_matches_enrolled_recipient():
    km = KeyManager(profile="default", store=FakeStore())
    recipient, _ = km.enroll()
    assert km.get_recipient() == recipient


def test_fingerprint_matches_enrolled():
    km = KeyManager(profile="default", store=FakeStore())
    recipient, fp = km.enroll()
    assert km.fingerprint() == fp


def test_delete_removes_key():
    km = KeyManager(profile="default", store=FakeStore())
    km.enroll()
    km.delete()
    assert not km.has_key()


def test_profiles_are_isolated():
    store = FakeStore()
    a = KeyManager(profile="alice", store=store)
    b = KeyManager(profile="bob", store=store)
    ra, _ = a.enroll()
    rb, _ = b.enroll()
    assert ra != rb
    assert a.get_recipient() == ra
    assert b.get_recipient() == rb


def test_enroll_fails_closed_when_no_keychain():
    km = KeyManager(profile="default", store=UnavailableStore())
    with pytest.raises(KaguraKeyCustodyError):
        km.enroll()
    assert not km.has_key()


def test_keyringstore_set_refuses_fail_backend(monkeypatch):
    monkeypatch.setattr(keyring, "get_keyring", lambda: keyring.backends.fail.Keyring())
    store = KeyringStore()
    with pytest.raises(KaguraKeyCustodyError):
        store.set("identity:default", "AGE-SECRET-KEY-1-whatever")


def test_keyringstore_get_wraps_keyring_error(monkeypatch):
    def boom(*_a, **_k):
        raise keyring.errors.KeyringError("keychain is locked")

    monkeypatch.setattr(keyring, "get_password", boom)
    store = KeyringStore()
    with pytest.raises(KaguraKeyCustodyError):
        store.get("identity:default")


def test_keyringstore_roundtrip_with_usable_backend(monkeypatch):
    """get/set/delete against a mocked usable backend (no real OS keychain)."""
    backing: dict[tuple[str, str], str] = {}

    class _DummyBackend:  # anything that is not keyring.backends.fail.Keyring
        pass

    def _set(service, name, value):
        backing[(service, name)] = value

    def _get(service, name):
        return backing.get((service, name))

    def _delete(service, name):
        if (service, name) not in backing:
            raise keyring.errors.PasswordDeleteError("not found")
        del backing[(service, name)]

    monkeypatch.setattr(keyring, "get_keyring", lambda: _DummyBackend())
    monkeypatch.setattr(keyring, "set_password", _set)
    monkeypatch.setattr(keyring, "get_password", _get)
    monkeypatch.setattr(keyring, "delete_password", _delete)

    store = KeyringStore()
    assert store.get("identity:default") is None
    store.set("identity:default", "AGE-SECRET-KEY-1-abc")
    assert store.get("identity:default") == "AGE-SECRET-KEY-1-abc"
    store.delete("identity:default")
    assert store.get("identity:default") is None
    store.delete("identity:default")  # idempotent: PasswordDeleteError swallowed


def test_keyringstore_set_wraps_keyring_error(monkeypatch):
    class _DummyBackend:  # usable backend (not keyring.backends.fail.Keyring)
        pass

    def boom(*_a, **_k):
        raise keyring.errors.KeyringError("write failed")

    monkeypatch.setattr(keyring, "get_keyring", lambda: _DummyBackend())
    monkeypatch.setattr(keyring, "set_password", boom)
    store = KeyringStore()
    with pytest.raises(KaguraKeyCustodyError):
        store.set("identity:default", "AGE-SECRET-KEY-1-x")
