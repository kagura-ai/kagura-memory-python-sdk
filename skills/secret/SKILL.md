---
name: secret
description: Store and retrieve zero-knowledge secrets with the `kagura secret` CLI — age recipient encryption, local decrypt (memory-cloud never sees plaintext). Use for keygen/enroll, put/get, grant/revoke/rotate, owner-only delete, exec-with-injected-env, and audit verification.
---

# kagura secret — zero-knowledge secret store

Drive the `kagura secret` command group: secrets are encrypted to recipients'
`age`/X25519 public keys and decrypted **locally**, so memory-cloud only ever
stores armored ciphertext. Thin wrapper around the installed `kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install 'kagura-memory[secret]'`
  (or `pip install 'kagura-memory[secret]'`), then stop. The **`[secret]` extra
  is required** — it pulls in `pyrage` (age crypto) and `keyring` (key custody).
- Requires memory-cloud **0.39.0+** and authentication. Run `kagura auth status`;
  if not authed, run the `auth` skill first.
- First use on a machine: `kagura secret keygen` creates an age keypair, stores
  the **private key in the OS keychain**, and registers the public key (it lands
  `pending` until a workspace owner approves it).

## Run (choose by intent)

```bash
kagura secret keygen --label laptop          # enroll: keypair → keychain, register pubkey
kagura secret approve <pubkey-id>            # owner: approve a pending key (verify fingerprint OOB)
kagura secret put db-prod < secret.txt       # store; value from stdin/--from-file (never argv)
kagura secret get db-prod | your-tool        # fetch + local decrypt (see guardrails below)
kagura secret exec --as DATABASE_URL=db-prod -- ./server   # inject into a child env, no disk/scrollback
kagura secret grant db-prod --to <pubkey-id> # re-encrypt to the expanded recipient set
kagura secret revoke db-prod --to <pubkey-id># revoke a grant — then rotate
kagura secret rotate db-prod                 # encrypt a NEW value to the remaining recipients
kagura secret list                           # secret metadata (never the values)
kagura secret delete db-prod                 # owner-only hard delete (cleanup, not invalidation — rotate first; needs memory-cloud 0.41.0+)
kagura secret audit-verify                   # verify the tamper-evident audit chain
```

## Consume the result

- **Never read a secret value aloud or paste it back.** `get` refuses to print to
  a terminal by design — keep it piped or write to a file (`-o FILE`, mode 0600).
  Do **not** pass `--reveal` on the user's behalf.
- **Never put a value on the command line** (`ps`/shell-history leak): always pipe
  it to `put`/`rotate` via stdin or `--from-file`.
- **Private keys never leave the keychain.** `keygen` prints only the public key +
  fingerprint — relay those, never any `AGE-SECRET-KEY...` material.
- **revoke ≠ cryptographic invalidation.** A revoked recipient may still hold a
  fetched copy — after `revoke`, run `kagura secret rotate` *and* regenerate the
  upstream provider credential to actually contain a leak.
- **delete ≠ invalidation either.** `kagura secret delete` is owner-only cleanup:
  it removes the stored ciphertext + all versions + grants, but does **not**
  un-share a value a recipient already fetched, nor rotate the live upstream
  credential. Confirm with the user before deleting, and steer them to rotate the
  upstream credential **first**, then delete — never delete on their behalf
  without that warning.
- On `keygen`/`approve`, surface the **fingerprint** so the user can verify it
  out-of-band (TOFU) before trusting the key.
