"""``kagura secret`` CLI — zero-knowledge secret store (Issue #216).

Composes the three library layers (crypto / keymanager / client). Security-
critical DX guards live here:

- ``get`` refuses to print a secret to a TTY unless ``--reveal``/``-o`` is given.
- ``put`` reads the value from stdin/``--from-file`` only — never from argv.
- ``keygen`` never prints the private key; only the public recipient + fingerprint.
- ``revoke`` warns that revoke is not cryptographic invalidation → rotate.
- ``exec`` injects secrets into the child's environment via process replacement
  and never mutates the parent ``os.environ``.

Module-level helpers (``_get_secret_client``, ``_make_key_manager``,
``_stdout_isatty``, ``_exec_child``) are deliberately overridable so the CLI is
unit-testable without a network, a keychain, or an actual ``exec``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
from collections.abc import Awaitable, Iterator
from pathlib import Path
from typing import TypeVar

import click

from .._auth import _resolve_auth
from ..config import load_config
from ..exceptions import KaguraAuthError, KaguraSecretError, _exc_message
from . import crypto
from .client import SecretClient
from .keymanager import KeyManager
from .models import (
    AuditVerifyResponse,
    PubkeyResponse,
    SecretMetaResponse,
    SecretPutResponse,
)

T = TypeVar("T")


# --------------------------------------------------------------------------
# Testable seams + IO helpers
# --------------------------------------------------------------------------


def _get_secret_client() -> SecretClient:
    """Construct a SecretClient via the canonical SDK auth chain."""
    config = load_config()
    try:
        resolved = _resolve_auth(api_key=None, mcp_url=None, profile=None, config=config)
    except KaguraAuthError as e:
        raise click.ClickException(_exc_message(e)) from e
    return SecretClient._from_resolved_auth(resolved)


def _make_key_manager(profile: str = "default") -> KeyManager:
    """Construct a KeyManager (OS-keychain custody)."""
    return KeyManager(profile=profile)


def _stdout_isatty() -> bool:
    return sys.stdout.isatty()


def _stdin_isatty() -> bool:
    return sys.stdin.isatty()


def _disable_core_dumps() -> None:
    """Best-effort: prevent plaintext from landing in a core dump on crash.

    POSIX-only; a silent no-op on Windows (no ``resource`` module / core dumps)
    or where the limit cannot be set.
    """
    if sys.platform == "win32":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        # Best-effort hardening: if the limit can't be set (sandboxed/already
        # lowered), proceed anyway rather than blocking the command.
        pass


def _read_secret_value(from_file: str | None) -> bytes:
    """Read a secret value from ``--from-file``, piped stdin, or a no-echo prompt.

    A single trailing newline (``\\n`` or ``\\r\\n``) from piped input is stripped
    so ``echo secret | kagura secret put ...`` stores ``secret``, not ``secret\\n``.
    The value is never taken from argv (it would leak via ``ps``/shell history).
    """
    if from_file is not None:
        return Path(from_file).read_bytes()
    if _stdin_isatty():
        import getpass

        return getpass.getpass("Secret value (input hidden): ").encode("utf-8")
    data = click.get_binary_stream("stdin").read()
    if data.endswith(b"\r\n"):
        return data[:-2]
    if data.endswith(b"\n"):
        return data[:-1]
    return data


def _write_secret_file(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` with mode 0600.

    Mirrors the temp+fsync+chmod+replace sequence of
    :func:`kagura_memory.auth.credentials._atomic_write_json` (kept as a separate
    helper here because that one is dict→JSON and concurrency-locked for the
    credentials hot path; this one writes raw secret bytes).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        # fsync the parent dir so the rename is durable on crash/power-loss
        # (parity with _atomic_write_json). Best-effort: unsupported on Windows
        # and some filesystems, where the atomic rename above already happened.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is unsupported on Windows / some filesystems; the
            # atomic rename above already landed, so skip the durability extra.
            pass
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _exec_child(argv: list[str], env: dict[str, str]) -> None:
    """Replace this process with ``argv`` carrying ``env``.

    On POSIX, ``execvpe`` replaces the address space so the decrypted plaintext
    in this process dies atomically. On Windows (no real exec), run as a child
    and propagate its exit code.
    """
    if not argv:
        raise click.ClickException("no command given to `kagura secret exec`")
    if os.name == "posix":
        os.execvpe(argv[0], argv, env)
    else:
        import subprocess

        completed = subprocess.run(argv, env=env)  # noqa: S603
        raise SystemExit(completed.returncode)


@contextlib.contextmanager
def _click_errors() -> Iterator[None]:
    """Map SDK/runtime errors to a clean ClickException.

    ClickException renders as ``Error: <message>`` with no traceback, so wrapping
    fallible SDK calls (crypto, keychain custody) prevents a raw Python traceback
    — whose exception ``repr`` could include sensitive bytes — from reaching the
    terminal. Use it around any synchronous SDK work done outside :func:`_run`.
    """
    try:
        yield
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(_exc_message(e)) from e


def _run(coro: Awaitable[T]) -> T:
    """Run an async operation, mapping SDK errors to ClickException."""
    with _click_errors():
        return asyncio.run(coro)  # type: ignore[arg-type]


async def _resolve_put_recipients(
    client: SecretClient, km: KeyManager, to_ids: tuple[str, ...]
) -> list[PubkeyResponse]:
    """Resolve ``--to`` ids to active recipients, defaulting to the caller's own key."""
    pubkeys = await client.list_pubkeys()
    active = {p.id: p for p in pubkeys if p.status == "active"}
    if to_ids:
        recipients = []
        for tid in to_ids:
            p = active.get(tid)
            if p is None:
                raise click.ClickException(f"pubkey {tid} is not an active recipient")
            recipients.append(p)
        return recipients
    my_fp = km.fingerprint()
    for p in active.values():
        if p.fingerprint == my_fp:
            return [p]
    raise click.ClickException(
        "no active recipient found for your key. Run `kagura secret keygen` and have an "
        "owner approve it, or pass --to <pubkey-id>."
    )


# --------------------------------------------------------------------------
# Command group
# --------------------------------------------------------------------------


@click.group()
def secret() -> None:
    """Zero-knowledge secret store — age recipient encryption, local decrypt."""


@secret.command()
@click.option("--profile", default="default", help="Key profile (default: default).")
@click.option("--label", default=None, help="Human-readable label for this pubkey.")
@click.option(
    "--no-register", is_flag=True, help="Generate + custody only; skip server registration."
)
def keygen(profile: str, label: str | None, no_register: bool) -> None:
    """Generate an age keypair, custody the private key, and register the public key.

    Idempotent: if a key already exists for this profile it is reused (not
    regenerated), so a run whose server registration previously failed can be
    safely retried.
    """
    km = _make_key_manager(profile)
    # Custody calls can fail-closed (no keychain) — map to a clean error, not a
    # traceback, so a headless host sees the actionable "install a keyring" hint.
    with _click_errors():
        if km.has_key():
            recipient, fp = km.get_recipient(), km.fingerprint()
            existed = True
        else:
            recipient, fp = km.enroll()
            existed = False
    if existed:
        click.echo(f"A key already exists for profile '{profile}'; reusing it.")
    else:
        click.echo(f"✓ age keypair generated and stored in your OS keychain (profile '{profile}').")
    click.echo(f"  Public key:  {recipient}")
    click.echo(f"  Fingerprint: {fp}")
    if no_register:
        click.echo("  (--no-register: did not register with the server.)")
        return

    async def _register() -> PubkeyResponse:
        async with _get_secret_client() as c:
            return await c.register_pubkey(recipient, label=label)

    resp = _run(_register())
    click.echo(f"  Registered with the server (status: {resp.status}).")
    click.echo(f"  An owner approves it with:  kagura secret approve {resp.id}")
    click.echo("  Share the fingerprint out-of-band so the owner can verify it (TOFU).")


@secret.command()
@click.argument("pubkey_id")
def approve(pubkey_id: str) -> None:
    """Approve a pending pubkey (owner only)."""

    async def _approve() -> PubkeyResponse:
        async with _get_secret_client() as c:
            return await c.approve_pubkey(pubkey_id)

    resp = _run(_approve())
    click.echo(f"✓ Approved pubkey {resp.id} (status: {resp.status}).")
    click.echo(f"  Fingerprint: {resp.fingerprint}")
    click.echo("  Verify this fingerprint matches what the member reported out-of-band (TOFU).")


@secret.command()
@click.option("--mine", is_flag=True, help="Only your own pubkeys.")
def pubkeys(mine: bool) -> None:
    """List recipient pubkeys."""

    async def _list() -> list[PubkeyResponse]:
        async with _get_secret_client() as c:
            return await (c.list_my_pubkeys() if mine else c.list_pubkeys())

    items = _run(_list())
    if not items:
        click.echo("(no pubkeys)")
        return
    for p in items:
        label = f"  {p.label}" if p.label else ""
        click.echo(f"{p.status:8} {p.fingerprint[:16]}…  {p.id}{label}")


@secret.command()
@click.argument("name")
@click.option(
    "--to", "to_ids", multiple=True, help="Recipient pubkey id (repeatable). Default: yourself."
)
@click.option(
    "--from-file",
    "from_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Read the value from a file instead of stdin.",
)
@click.option("--profile", default="default", help="Your key profile (default: default).")
def put(name: str, to_ids: tuple[str, ...], from_file: str | None, profile: str) -> None:
    """Store a secret. The value is read from stdin or --from-file (never from argv)."""
    with _click_errors():  # a --from-file read error must not surface as a traceback
        value = _read_secret_value(from_file)
    km = _make_key_manager(profile)

    async def _put() -> tuple[SecretPutResponse, list[PubkeyResponse]]:
        async with _get_secret_client() as c:
            recipients = await _resolve_put_recipients(c, km, to_ids)
            resp = await c.put_secret_for_recipients(name, value, recipients)
            return resp, recipients

    resp, recipients = _run(_put())
    click.echo(
        f"✓ Stored '{resp.name}' v{resp.version_number} for {len(recipients)} recipient(s).",
        err=True,
    )


@secret.command()
@click.argument("name")
@click.option(
    "-o",
    "--output",
    "output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Write to FILE (mode 0600) instead of stdout.",
)
@click.option(
    "--reveal", is_flag=True, help="Allow printing to a terminal (overrides the TTY guard)."
)
@click.option(
    "--version", "version_number", type=int, default=None, help="Pin a version (default: latest)."
)
@click.option("--profile", default="default", help="Your key profile (default: default).")
def get(
    name: str, output: str | None, reveal: bool, version_number: int | None, profile: str
) -> None:
    """Fetch a secret and decrypt it locally with your key."""
    if output is None and not reveal and _stdout_isatty():
        raise click.ClickException(
            "refusing to print a secret to a terminal.\n"
            f"  Pipe it:    kagura secret get {name} | your-tool\n"
            f"  To a file:  kagura secret get {name} -o FILE   (written 0600)\n"
            f"  Force show: kagura secret get {name} --reveal"
        )
    _disable_core_dumps()
    km = _make_key_manager(profile)

    # Decrypt inside the _run scope so a custody/decrypt failure (no key
    # enrolled, not a recipient, corrupt ciphertext) surfaces as a clean error.
    async def _fetch() -> bytes:
        async with _get_secret_client() as c:
            sv = await c.fetch_secret(name, version_number=version_number)
        return crypto.decrypt(sv.ciphertext, km.get_identity())

    plaintext = _run(_fetch())
    if output is not None:
        _write_secret_file(Path(output), plaintext)
        click.echo(f"✓ wrote '{name}' to {output} (mode 0600).", err=True)
    else:
        click.echo(plaintext, nl=False)


@secret.command(name="list")
def list_() -> None:
    """List secret metadata (never the values)."""

    async def _list() -> list[SecretMetaResponse]:
        async with _get_secret_client() as c:
            return await c.list_secrets()

    items = _run(_list())
    if not items:
        click.echo("(no secrets)")
        return
    for s in items:
        flag = "  ⚠ rotation needed" if s.rotation_needed else ""
        click.echo(f"{s.name}  v{s.current_version}  grants={s.grant_count}  {s.status}{flag}")


@secret.command()
@click.argument("name")
@click.option("--to", "to_id", required=True, help="Pubkey id to grant.")
@click.option("--profile", default="default", help="Your key profile (default: default).")
def grant(name: str, to_id: str, profile: str) -> None:
    """Grant a recipient access (re-encrypts the secret to the expanded set).

    You must be a current recipient — only a recipient can decrypt the value to
    re-encrypt it for the new set.
    """
    km = _make_key_manager(profile)

    async def _grant() -> tuple[SecretPutResponse, list[PubkeyResponse]]:
        async with _get_secret_client() as c:
            # The fetch and the pubkey listing are independent — run concurrently.
            sv, pubkeys_ = await asyncio.gather(c.fetch_secret(name), c.list_pubkeys())
            plaintext = crypto.decrypt(sv.ciphertext, km.get_identity())
            active = [p for p in pubkeys_ if p.status == "active"]
            snapshot = set(sv.recipients_snapshot)
            by_id = {p.id: p for p in active if p.fingerprint in snapshot}
            new = next((p for p in active if p.id == to_id), None)
            if new is None:
                raise click.ClickException(f"pubkey {to_id} is not an active recipient")
            by_id[new.id] = new
            recipients = list(by_id.values())
            resp = await c.put_secret_for_recipients(name, plaintext, recipients)
            return resp, recipients

    resp, recipients = _run(_grant())
    click.echo(
        f"✓ granted {to_id} on '{name}'; re-encrypted to {len(recipients)} recipient(s) "
        f"(v{resp.version_number})."
    )


@secret.command()
@click.argument("name")
@click.option("--to", "to_id", required=True, help="Recipient pubkey id to revoke.")
def revoke(name: str, to_id: str) -> None:
    """Revoke one recipient's grant. Does NOT invalidate already-fetched copies."""

    async def _revoke() -> SecretMetaResponse:
        async with _get_secret_client() as c:
            return await c.revoke_grant(name, to_id)

    _run(_revoke())
    click.echo(f"✓ revoked {to_id} on '{name}'.")
    click.echo("  ⚠ revoke does NOT invalidate copies already fetched by that recipient.")
    click.echo(f"     To contain a leak, rotate now:  kagura secret rotate {name}")
    click.echo("     (rotate re-encrypts a new value to the remaining recipients; also")
    click.echo("      regenerate the upstream provider credential.)")


@secret.command()
@click.argument("name")
@click.option(
    "--from-file",
    "from_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Read the new value from a file instead of stdin.",
)
@click.option("--profile", default="default", help="Your key profile (default: default).")
def rotate(name: str, from_file: str | None, profile: str) -> None:
    """Rotate a secret: encrypt a NEW value to the remaining recipients."""
    with _click_errors():  # a --from-file read error must not surface as a traceback
        new_value = _read_secret_value(from_file)

    async def _rotate() -> tuple[SecretPutResponse, list[PubkeyResponse]]:
        async with _get_secret_client() as c:
            sv, pubkeys_ = await asyncio.gather(c.fetch_secret(name), c.list_pubkeys())
            snapshot = set(sv.recipients_snapshot)
            recipients = [p for p in pubkeys_ if p.status == "active" and p.fingerprint in snapshot]
            if not recipients:
                raise click.ClickException(f"no active recipients remain for '{name}'")
            resp = await c.put_secret_for_recipients(name, new_value, recipients)
            return resp, recipients

    resp, recipients = _run(_rotate())
    click.echo(f"✓ rotated '{name}' to v{resp.version_number} for {len(recipients)} recipient(s).")
    click.echo("  Remember to regenerate the upstream provider credential (the real token).")


@secret.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def delete(name: str, yes: bool) -> None:
    """Hard-delete a secret and all its versions + grants (owner only).

    Delete is CLEANUP, not a security control: it removes the stored ciphertext
    but does NOT un-share a value a recipient already fetched, nor rotate the
    live upstream credential. Rotate the upstream credential first, then delete.
    """
    if not yes:
        click.echo(f"About to permanently delete secret '{name}' (all versions + grants).")
        click.echo("  ⚠ Delete is CLEANUP, not a security control. It removes stored")
        click.echo("    ciphertext but does NOT un-share a value already fetched, nor")
        click.echo("    rotate the live upstream credential.")
        click.echo("    To contain a leak, rotate the upstream credential FIRST, then delete.")
        click.confirm("Proceed?", abort=True)

    async def _delete() -> None:
        async with _get_secret_client() as c:
            await c.delete_secret(name)

    _run(_delete())
    click.echo(f"✓ deleted secret '{name}'.")


@secret.command(name="audit-verify")
def audit_verify() -> None:
    """Verify the tamper-evident audit chain (owner/admin)."""

    async def _verify() -> AuditVerifyResponse:
        async with _get_secret_client() as c:
            return await c.verify_audit()

    r = _run(_verify())
    if r.valid:
        click.echo(f"✓ audit chain valid ({r.entries} entries, head {r.head}).")
    else:
        raise click.ClickException(f"audit chain BROKEN at entry {r.broken_at}: {r.reason}")


@secret.command(
    name="exec",
    context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False},
)
@click.option(
    "--as", "as_specs", multiple=True, required=True, help="ENV_NAME=secret_name (repeatable)."
)
@click.option("--profile", default="default", help="Your key profile (default: default).")
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
def exec_(as_specs: tuple[str, ...], profile: str, command: tuple[str, ...]) -> None:
    """Run COMMAND with secrets injected into its environment (no disk, no scrollback).

    Example:  kagura secret exec --as DATABASE_URL=db-prod -- ./server
    """
    _disable_core_dumps()
    km = _make_key_manager(profile)
    specs: list[tuple[str, str]] = []
    for spec in as_specs:
        env_name, sep, secret_name = spec.partition("=")
        if not sep or not env_name or not secret_name:
            raise click.ClickException(f"--as expects ENV_NAME=secret_name (got {spec!r})")
        specs.append((env_name, secret_name))

    async def _build_child_env() -> dict[str, str]:
        # Read the identity once (not once per secret) and fetch concurrently.
        identity = km.get_identity()
        async with _get_secret_client() as c:
            fetched = await asyncio.gather(*(c.fetch_secret(n) for _, n in specs))
        # Build on a COPY — never mutate os.environ (python.md). The decode runs
        # INSIDE this _run-wrapped coroutine and raises a value-free error on
        # non-UTF-8 bytes, so the plaintext never reaches a traceback.
        env = dict(os.environ)
        for (env_name, secret_name), sv in zip(specs, fetched, strict=True):
            try:
                env[env_name] = crypto.decrypt(sv.ciphertext, identity).decode("utf-8")
            except UnicodeDecodeError:
                raise KaguraSecretError(
                    f"secret {secret_name!r} is not valid UTF-8 and cannot be used as an "
                    "environment variable"
                ) from None  # `from None`: do not chain the exc (its repr holds the bytes)
        return env

    child_env = _run(_build_child_env())
    _exec_child(list(command), child_env)
