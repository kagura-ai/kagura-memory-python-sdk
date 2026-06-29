"""Zero-knowledge secret client for Kagura Memory Cloud (Issue #216).

Client side of the secret store (server: memory-cloud#1128, v0.39.0):
``age``/X25519 recipient encryption with **local decryption** — the private
key never leaves the client and memory-cloud only ever sees armored ciphertext.

Submodules:

- :mod:`~kagura_memory.secrets.crypto` — age wrap/unwrap, armor codec,
  fingerprint (wraps the audited ``pyrage`` binding).
"""
