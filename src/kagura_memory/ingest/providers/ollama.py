"""Ollama provider via litellm (local or Ollama Cloud).

Uses ``qwen3:30b`` by default for text and ``qwen2.5vl:7b`` for vision.
Model names use the ``ollama_chat/`` prefix so litellm dispatches via the
native ``/api/chat`` endpoint (system messages preserved as-is). The
legacy ``ollama/`` prefix routes through ``/api/generate``, which flattens
system + user into a single prompt — explicit callers passing the legacy
prefix are auto-migrated with a ``DeprecationWarning``.

Environment variables read by litellm directly:

- ``OLLAMA_API_BASE`` — base URL. Defaults to ``http://localhost:11434``;
  set to ``https://ollama.com`` for Ollama Cloud.
- ``OLLAMA_API_KEY`` — Bearer token for Ollama Cloud. Leave unset for
  local Ollama (no auth needed).

Two-step recipe for Ollama Cloud::

    export OLLAMA_API_KEY="..."          # or use `ollama signin`
    export OLLAMA_API_BASE="https://ollama.com"
    kagura ingest --text-provider ollama ...

For sensitive documents, leave both unset — nothing leaves the machine.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

from ._litellm import LiteLLMProvider


def _migrate_legacy_ollama_prefix(model: str | None) -> str | None:
    """Rewrite ``ollama/<tag>`` to ``ollama_chat/<tag>`` with a DeprecationWarning.

    The legacy ``ollama/`` prefix flattens system + user prompts via litellm's
    ``/api/generate`` path; ``ollama_chat/`` preserves the message structure
    through ``/api/chat``. Callers passing the legacy form are auto-migrated
    so summarization quality is consistent.
    """
    if model and model.startswith("ollama/"):
        new_model = "ollama_chat/" + model.removeprefix("ollama/")
        warnings.warn(
            (
                f"OllamaProvider: model prefix 'ollama/' is deprecated — "
                f"auto-rewriting '{model}' to '{new_model}'. Pass "
                "'ollama_chat/...' explicitly to silence this warning."
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        return new_model
    return model


class OllamaProvider(LiteLLMProvider):
    name: ClassVar[str] = "ollama"
    default_text_model: ClassVar[str] = "ollama_chat/qwen3:30b"
    default_vision_model: ClassVar[str | None] = "ollama_chat/qwen2.5vl:7b"

    def __init__(
        self,
        text_model: str | None = None,
        vision_model: str | None = None,
    ):
        super().__init__(
            text_model=_migrate_legacy_ollama_prefix(text_model),
            vision_model=_migrate_legacy_ollama_prefix(vision_model),
        )
