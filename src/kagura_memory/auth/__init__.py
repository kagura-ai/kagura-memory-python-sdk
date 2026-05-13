"""OAuth2 device-flow authentication for Kagura Memory SDK.

Implements RFC 8628 device authorization grant against memory-cloud's
``/api/v1/oauth2/*`` endpoints for direct CLI / ``KaguraClient`` use.

Integration with ``kagura setup claude`` (Claude Code MCP server config)
is intentionally out of scope here — see follow-up issue #101 for the
``kagura-mcp`` proxy daemon that resolves the static-config × short-lived
token mismatch.
"""

from .credentials import (
    DEFAULT_CREDENTIALS_PATH,
    REFRESH_SKEW_SEC,
    CredentialsFile,
    KaguraOAuth,
    OAuthCredentials,
    get_shared_state,
    load_credentials_file,
    save_credentials_file,
)

__all__ = [
    "CredentialsFile",
    "DEFAULT_CREDENTIALS_PATH",
    "KaguraOAuth",
    "OAuthCredentials",
    "REFRESH_SKEW_SEC",
    "get_shared_state",
    "load_credentials_file",
    "save_credentials_file",
]
