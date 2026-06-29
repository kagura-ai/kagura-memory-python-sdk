"""Tests for the `kagura secret` CLI subgroup.

Focus: the security-critical DX guards (get's TTY refusal, put-from-stdin,
keygen never printing the private key, revoke's rotate warning, exec's env
injection without mutating os.environ) plus a representative happy path per
command. Network and keychain are replaced via the module's testable seams
(`_get_secret_client`, `_make_key_manager`, `_stdout_isatty`, `_exec_child`).
"""

import os
import subprocess
import uuid

import click
import pytest
from click.testing import CliRunner

from kagura_memory.exceptions import KaguraCryptoError, KaguraKeyCustodyError
from kagura_memory.secrets import cli as secret_cli
from kagura_memory.secrets import crypto
from kagura_memory.secrets.cli import secret
from kagura_memory.secrets.keymanager import KeyManager
from kagura_memory.secrets.models import (
    AuditVerifyResponse,
    PubkeyResponse,
    SecretMetaResponse,
    SecretPutResponse,
    SecretValueResponse,
)


class _FakeStore:
    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, name):
        return self.data.get(name)

    def set(self, name, value):
        self.data[name] = value

    def delete(self, name):
        self.data.pop(name, None)


class _UnavailableStore(_FakeStore):
    """Simulates a host with no usable keychain (fail-closed)."""

    def set(self, name, value):
        raise KaguraKeyCustodyError("no secure keychain available")


class FakeSecretClient:
    """Async-context-manager stand-in for SecretClient."""

    def __init__(self):
        self.pubkeys: list[PubkeyResponse] = []
        self.value: SecretValueResponse | None = None
        self.audit = AuditVerifyResponse(valid=True, entries=3, head="abc")
        self.secrets: list[SecretMetaResponse] = []
        self.calls: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def register_pubkey(self, pubkey, label=None):
        self.calls["register"] = (pubkey, label)
        return PubkeyResponse(
            id=str(uuid.uuid4()),
            identity_id=str(uuid.uuid4()),
            pubkey=pubkey,
            fingerprint=crypto.fingerprint(pubkey),
            label=label,
            status="pending",
            created_at="t",
        )

    async def approve_pubkey(self, pubkey_id):
        self.calls["approve"] = pubkey_id
        pub = self.pubkeys[0].pubkey if self.pubkeys else "age1x"
        return PubkeyResponse(
            id=pubkey_id,
            identity_id="i",
            pubkey=pub,
            fingerprint=crypto.fingerprint(pub),
            label=None,
            status="active",
            created_at="t",
        )

    async def list_pubkeys(self):
        return self.pubkeys

    async def list_my_pubkeys(self):
        return self.pubkeys

    async def fetch_secret(self, name, version_number=None):
        self.calls["fetch"] = (name, version_number)
        return self.value

    async def put_secret_for_recipients(self, name, plaintext, recipients):
        self.calls["put"] = (name, plaintext, recipients)
        return SecretPutResponse(
            name=name, version_number=1, status="active", rotation_needed=False
        )

    async def revoke_grant(self, name, recipient_pubkey_id):
        self.calls["revoke"] = (name, recipient_pubkey_id)
        return SecretMetaResponse(
            name=name,
            status="active",
            rotation_needed=True,
            current_version=2,
            grant_count=1,
            created_at="t",
            updated_at="t",
        )

    async def verify_audit(self):
        return self.audit

    async def list_secrets(self):
        return self.secrets


@pytest.fixture
def km():
    """A real KeyManager with an in-memory store, enrolled."""
    manager = KeyManager(profile="default", store=_FakeStore())
    manager.enroll()
    return manager


@pytest.fixture
def wired(monkeypatch, km):
    """Wire the CLI seams to a FakeSecretClient + the enrolled km."""
    fake = FakeSecretClient()
    monkeypatch.setattr(secret_cli, "_get_secret_client", lambda: fake)
    monkeypatch.setattr(secret_cli, "_make_key_manager", lambda profile="default": km)
    return fake, km


def _active_pubkey(recipient: str) -> PubkeyResponse:
    return PubkeyResponse(
        id=str(uuid.uuid4()),
        identity_id="i",
        pubkey=recipient,
        fingerprint=crypto.fingerprint(recipient),
        label="laptop",
        status="active",
        created_at="t",
    )


# --- keygen -----------------------------------------------------------------


def test_keygen_never_prints_private_key(monkeypatch):
    fake = FakeSecretClient()
    monkeypatch.setattr(secret_cli, "_get_secret_client", lambda: fake)
    monkeypatch.setattr(
        secret_cli,
        "_make_key_manager",
        lambda profile="default": KeyManager(profile=profile, store=_FakeStore()),
    )
    result = CliRunner().invoke(secret, ["keygen", "--label", "laptop"])
    assert result.exit_code == 0, result.output
    assert "AGE-SECRET-KEY" not in result.output  # private key never leaks
    assert "age1" in result.output  # public recipient shown
    assert "Fingerprint" in result.output
    assert fake.calls["register"][1] == "laptop"  # registered with label


def test_keygen_no_register_skips_server(monkeypatch):
    fake = FakeSecretClient()
    monkeypatch.setattr(secret_cli, "_get_secret_client", lambda: fake)
    monkeypatch.setattr(
        secret_cli,
        "_make_key_manager",
        lambda profile="default": KeyManager(profile=profile, store=_FakeStore()),
    )
    result = CliRunner().invoke(secret, ["keygen", "--no-register"])
    assert result.exit_code == 0, result.output
    assert "AGE-SECRET-KEY" not in result.output
    assert "register" not in fake.calls  # never hit the server


def test_keygen_is_idempotent_reuses_existing_key(monkeypatch):
    """If a key is already custodied, keygen reuses it and re-registers (recoverable)."""
    fake = FakeSecretClient()
    km = KeyManager(profile="default", store=_FakeStore())
    recipient, _ = km.enroll()
    monkeypatch.setattr(secret_cli, "_get_secret_client", lambda: fake)
    monkeypatch.setattr(secret_cli, "_make_key_manager", lambda profile="default": km)
    result = CliRunner().invoke(secret, ["keygen"])
    assert result.exit_code == 0, result.output
    assert fake.calls["register"][0] == recipient  # re-registered the SAME key
    assert km.get_recipient() == recipient  # key unchanged (not regenerated)


# --- get: the centerpiece DX guard -----------------------------------------


def test_get_refuses_to_print_to_tty(monkeypatch, wired):
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: True)
    result = CliRunner().invoke(secret, ["get", "db"])
    assert result.exit_code != 0
    assert "refusing" in result.output.lower()


def test_get_pipes_plaintext_without_trailing_newline(monkeypatch, wired):
    fake, km = wired
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: False)
    armored = crypto.encrypt(b"hunter2", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["get", "db"])
    assert result.exit_code == 0, result.output
    assert result.output == "hunter2"  # exact, no trailing newline


def test_get_reveal_bypasses_tty_guard(monkeypatch, wired):
    fake, km = wired
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: True)
    armored = crypto.encrypt(b"shown", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["get", "db", "--reveal"])
    assert result.exit_code == 0, result.output
    assert "shown" in result.output


def test_get_writes_file_0600(monkeypatch, wired, tmp_path):
    fake, km = wired
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: True)  # would refuse, but -o given
    armored = crypto.encrypt(b"filesecret", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    out = tmp_path / "secret.key"
    result = CliRunner().invoke(secret, ["get", "db", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == b"filesecret"


def test_get_pins_version(monkeypatch, wired):
    fake, km = wired
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: False)
    armored = crypto.encrypt(b"v5data", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=5,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["get", "db", "--version", "5"])
    assert result.exit_code == 0, result.output
    assert fake.calls["fetch"] == ("db", 5)


# --- put --------------------------------------------------------------------


def test_put_reads_stdin_and_encrypts(monkeypatch, wired):
    fake, km = wired
    fake.pubkeys = [_active_pubkey(km.get_recipient())]
    result = CliRunner().invoke(secret, ["put", "db"], input="hunter2\n")
    assert result.exit_code == 0, result.output
    name, plaintext, recipients = fake.calls["put"]
    assert name == "db"
    assert plaintext == b"hunter2"  # one trailing newline stripped
    assert len(recipients) == 1


def test_put_has_no_value_option():
    """A secret value must never be passable via argv."""
    result = CliRunner().invoke(secret, ["put", "db", "--value", "leak"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_put_with_explicit_to(wired):
    fake, km = wired
    _, r2 = crypto.generate_keypair()
    pk2 = _active_pubkey(r2)
    fake.pubkeys = [_active_pubkey(km.get_recipient()), pk2]
    result = CliRunner().invoke(secret, ["put", "db", "--to", pk2.id], input="v\n")
    assert result.exit_code == 0, result.output
    _, _, recipients = fake.calls["put"]
    assert [r.id for r in recipients] == [pk2.id]


def test_put_rejects_unknown_recipient(wired):
    fake, km = wired
    fake.pubkeys = [_active_pubkey(km.get_recipient())]
    result = CliRunner().invoke(secret, ["put", "db", "--to", "nope"], input="v\n")
    assert result.exit_code != 0
    assert "not an active recipient" in result.output.lower()


# --- revoke: the rotate warning ---------------------------------------------


def test_revoke_warns_revoke_is_not_invalidation(wired):
    fake, _ = wired
    pk_id = str(uuid.uuid4())
    result = CliRunner().invoke(secret, ["revoke", "db", "--to", pk_id])
    assert result.exit_code == 0, result.output
    assert fake.calls["revoke"] == ("db", pk_id)
    low = result.output.lower()
    assert "rotate" in low
    assert "not" in low and "invalidate" in low


# --- exec: env injection without mutating os.environ ------------------------


def test_exec_injects_env_and_does_not_touch_os_environ(monkeypatch, wired):
    fake, km = wired
    armored = crypto.encrypt(b"db-pass", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    captured = {}

    def fake_exec(argv, env):
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.setattr(secret_cli, "_exec_child", fake_exec)
    assert "DBPASS" not in os.environ
    result = CliRunner().invoke(secret, ["exec", "--as", "DBPASS=db", "--", "myserver", "--flag"])
    assert result.exit_code == 0, result.output
    assert captured["argv"] == ["myserver", "--flag"]
    assert captured["env"]["DBPASS"] == "db-pass"
    assert "DBPASS" not in os.environ  # parent env untouched


# --- approve / list / pubkeys / audit-verify --------------------------------


def test_pubkeys_lists_active(wired):
    fake, km = wired
    fake.pubkeys = [_active_pubkey(km.get_recipient())]
    result = CliRunner().invoke(secret, ["pubkeys"])
    assert result.exit_code == 0, result.output
    assert "active" in result.output


def test_exec_rejects_malformed_as_spec(wired):
    result = CliRunner().invoke(secret, ["exec", "--as", "BROKEN", "--", "cmd"])
    assert result.exit_code != 0
    assert "ENV_NAME=secret_name" in result.output


def test_approve_shows_fingerprint_and_tofu_note(wired):
    fake, _ = wired
    fake.pubkeys = [_active_pubkey("age1qqqqqqqqqqqqqqqqqqqqqqqqqqqq")]
    pk_id = str(uuid.uuid4())
    result = CliRunner().invoke(secret, ["approve", pk_id])
    assert result.exit_code == 0, result.output
    assert fake.calls["approve"] == pk_id
    assert "fingerprint" in result.output.lower()
    assert "verify" in result.output.lower()  # TOFU prompt


def test_audit_verify_reports_valid(wired):
    result = CliRunner().invoke(secret, ["audit-verify"])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_audit_verify_fails_on_broken_chain(monkeypatch, wired):
    fake, _ = wired
    fake.audit = AuditVerifyResponse(valid=False, broken_at=7, reason="hash mismatch")
    result = CliRunner().invoke(secret, ["audit-verify"])
    assert result.exit_code != 0
    assert "broken" in result.output.lower() or "7" in result.output


def test_list_secrets(wired):
    fake, _ = wired
    fake.secrets = [
        SecretMetaResponse(
            name="db",
            status="active",
            rotation_needed=True,
            current_version=2,
            grant_count=3,
            created_at="t",
            updated_at="t",
        )
    ]
    result = CliRunner().invoke(secret, ["list"])
    assert result.exit_code == 0, result.output
    assert "db" in result.output


# --- rotate -----------------------------------------------------------------


def test_grant_adds_recipient_and_reencrypts(wired):
    fake, km = wired
    r1 = km.get_recipient()
    _, r2 = crypto.generate_keypair()
    pk1 = _active_pubkey(r1)
    pk2 = _active_pubkey(r2)
    fake.pubkeys = [pk1, pk2]
    old = crypto.encrypt(b"old-value", [r1])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=old,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["grant", "db", "--to", pk2.id])
    assert result.exit_code == 0, result.output
    _, plaintext, recipients = fake.calls["put"]
    assert plaintext == b"old-value"  # decrypted with own key, then re-encrypted
    assert {r.id for r in recipients} == {pk1.id, pk2.id}


def test_rotate_reencrypts_to_remaining_recipients(monkeypatch, wired):
    fake, km = wired
    recipient = km.get_recipient()
    fake.pubkeys = [_active_pubkey(recipient)]
    old = crypto.encrypt(b"old", [recipient])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=old,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=True,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["rotate", "db"], input="newsecret\n")
    assert result.exit_code == 0, result.output
    name, plaintext, recipients = fake.calls["put"]
    assert plaintext == b"newsecret"
    assert len(recipients) == 1


# --- error mapping: SDK errors must surface as clean ClickExceptions --------


def test_exec_non_utf8_secret_is_clean_error_no_leak(monkeypatch, wired):
    """A non-UTF-8 secret must NOT leak via a raw UnicodeDecodeError traceback."""
    fake, km = wired
    armored = crypto.encrypt(b"\xff\xfe\x00binary", [km.get_recipient()])
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[km.fingerprint()],
        rotation_needed=False,
        created_at="t",
    )
    monkeypatch.setattr(secret_cli, "_exec_child", lambda argv, env: None)
    result = CliRunner().invoke(secret, ["exec", "--as", "TOKEN=db", "--", "cmd"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, UnicodeDecodeError)  # mapped, not raw
    assert "utf-8" in result.output.lower()  # value-free explanation


def test_get_undecryptable_is_clean_error(monkeypatch, wired):
    fake, km = wired
    monkeypatch.setattr(secret_cli, "_stdout_isatty", lambda: False)
    _, other = crypto.generate_keypair()
    armored = crypto.encrypt(b"x", [other])  # NOT encrypted to km's key
    fake.value = SecretValueResponse(
        name="db",
        version_number=1,
        alg="age",
        ciphertext=armored,
        blob_ref=None,
        recipients_snapshot=[crypto.fingerprint(other)],
        rotation_needed=False,
        created_at="t",
    )
    result = CliRunner().invoke(secret, ["get", "db"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, KaguraCryptoError)  # mapped to ClickException


def test_keygen_no_keychain_is_clean_error(monkeypatch):
    fake = FakeSecretClient()
    monkeypatch.setattr(secret_cli, "_get_secret_client", lambda: fake)
    monkeypatch.setattr(
        secret_cli,
        "_make_key_manager",
        lambda profile="default": KeyManager(profile=profile, store=_UnavailableStore()),
    )
    result = CliRunner().invoke(secret, ["keygen"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, KaguraKeyCustodyError)  # mapped, no traceback
    assert "keychain" in result.output.lower()


def test_put_value_read_error_is_clean_error(monkeypatch, wired):
    def boom(_from_file):
        raise OSError("permission denied")

    monkeypatch.setattr(secret_cli, "_read_secret_value", boom)
    result = CliRunner().invoke(secret, ["put", "db"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, OSError)  # mapped to ClickException, no traceback


def test_rotate_value_read_error_is_clean_error(monkeypatch, wired):
    def boom(_from_file):
        raise OSError("permission denied")

    monkeypatch.setattr(secret_cli, "_read_secret_value", boom)
    result = CliRunner().invoke(secret, ["rotate", "db"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, OSError)


# --- coverage: helpers, exec_child branches, empty-list paths ----------------


def test_exec_child_requires_a_command():
    with pytest.raises(click.ClickException):
        secret_cli._exec_child([], {"A": "b"})


def test_exec_child_posix_replaces_process(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    captured = {}
    monkeypatch.setattr(
        os, "execvpe", lambda file, argv, env: captured.update(file=file, argv=argv, env=env)
    )
    secret_cli._exec_child(["echo", "hi"], {"A": "b"})
    assert captured == {"file": "echo", "argv": ["echo", "hi"], "env": {"A": "b"}}


def test_exec_child_windows_runs_subprocess(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    class _Completed:
        returncode = 3

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())
    with pytest.raises(SystemExit) as exc:
        secret_cli._exec_child(["cmd"], {"A": "b"})
    assert exc.value.code == 3


def test_read_secret_value_from_file_is_verbatim(tmp_path):
    f = tmp_path / "s.txt"
    f.write_bytes(b"file-value\n")  # trailing newline preserved for file input
    assert secret_cli._read_secret_value(str(f)) == b"file-value\n"


def test_read_secret_value_prompts_on_tty(monkeypatch):
    import getpass

    monkeypatch.setattr(secret_cli, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "typed-secret")
    assert secret_cli._read_secret_value(None) == b"typed-secret"


def test_pubkeys_empty(wired):
    fake, _ = wired
    fake.pubkeys = []
    result = CliRunner().invoke(secret, ["pubkeys"])
    assert result.exit_code == 0, result.output
    assert "no pubkeys" in result.output.lower()


def test_pubkeys_mine(wired):
    fake, km = wired
    fake.pubkeys = [_active_pubkey(km.get_recipient())]
    result = CliRunner().invoke(secret, ["pubkeys", "--mine"])
    assert result.exit_code == 0, result.output
    assert "active" in result.output


def test_list_empty(wired):
    fake, _ = wired
    fake.secrets = []
    result = CliRunner().invoke(secret, ["list"])
    assert result.exit_code == 0, result.output
    assert "no secrets" in result.output.lower()
