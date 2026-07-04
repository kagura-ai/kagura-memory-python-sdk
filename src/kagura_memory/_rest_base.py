"""Shared construction/auth/lifecycle/error spine for Kagura REST clients.

FilesClient, ResourceClient, SecretClient, and WorkspaceClient carried
four near-verbatim copies of the same scaffolding, and the copies had
already drifted (403 credential scrubbing and the OAuth-aware 401 hint
existed in some clients but not others — found by the #228 review).
:class:`KaguraRestClient` owns the spine once; each client keeps only
its wire-contract differences via the ``_raise_401``/``_raise_403``/
``_raise_429`` hooks (#229).

Invariants the base enforces for every client:

- credentials are required up front (``api_key`` or a ``KaguraOAuth``
  handler) with an actionable ValueError naming the class's factory;
- HTTPS-only base URLs (loopback exempt) via ``validate_https_url``;
- the API key is baked into the ``Authorization`` header exactly once
  and never stored as an instance attribute (python.md);
- one ``httpx.AsyncClient`` per client instance, closed by
  ``close()`` / ``async with``.
"""

from __future__ import annotations

from typing import Any, Literal, NoReturn, Self

import httpx

from ._auth import _AuthSource, _OAuthAuth, _resolve_auth, _StaticAuth
from ._http import (
    SDK_VERSION,
    _retry_after_seconds,
    base_url_from_mcp,
    extract_detail,
    validate_https_url,
)
from .auth.credentials import KaguraOAuth
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
    _exc_message,
)

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class KaguraRestClient:
    """Base class for the Kagura REST API clients.

    Subclasses add their domain methods on top of :meth:`_request` and
    override the ``_raise_*`` hooks where their server contract differs.
    The default hooks implement the majority behavior:

    - 401 → :class:`KaguraAuthError` with an OAuth-aware recovery hint
    - 403 → the generic ``HTTP 403: <detail>`` mapping
    - 404 → :class:`KaguraNotFoundError` (server detail or "Not found")
    - 429 → :class:`KaguraQuotaError` with a tolerant ``Retry-After``
    - other statuses → :class:`KaguraConnectionError`
    - transport errors → :class:`KaguraConnectionError`
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        *,
        _oauth: KaguraOAuth | None = None,
        _auth_source: _AuthSource | None = None,
        _workspace_id_hint: str | None = None,
    ) -> None:
        """Initialize with a static API key or a pre-built OAuth handler.

        For credential resolution (the auto chain env → ``~/.kagura/
        credentials.json`` → ``.kagura.json``), use :meth:`from_mcp_url` —
        a bare constructor deliberately does not read disk state.

        Args:
            api_key: Kagura API key (Bearer token). Required unless
                ``_oauth`` is supplied.
            base_url: REST API base URL (without path).
            timeout: API request timeout in seconds.
            _oauth: Private. ``from_mcp_url`` passes a ``KaguraOAuth``
                httpx.Auth handler when the resolver picked an OAuth
                profile.
            _auth_source: Private. Provenance tag from ``_resolve_auth``;
                clients use it to flavor 403 hints (issue #115).
            _workspace_id_hint: Private. Workspace UUID associated with
                the credential source — hint display only, never sent on
                the wire.

        Raises:
            ValueError: If neither ``api_key`` nor ``_oauth`` is given,
                or the base URL is plain HTTP on a non-loopback host.
        """
        if api_key is None and _oauth is None:
            name = type(self).__name__
            raise ValueError(
                f"{name} requires api_key, or use {name}.from_mcp_url(...) "
                "to resolve credentials from environment, OAuth profile, or .kagura.json."
            )

        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")
        self.base_url = stripped_url

        if _oauth is not None:
            # OAuth path: the handler injects a fresh access_token per
            # request and coordinates refresh via the credentials lock.
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
                auth=_oauth,
            )
        else:
            # Static path: bake the bearer header once. The key is never
            # stored as an instance attribute (python.md "Never store API
            # keys as instance attributes").
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"kagura-memory-sdk/{SDK_VERSION}",
                },
            )
        self._oauth: KaguraOAuth | None = _oauth
        self._auth_source: _AuthSource | None = _auth_source
        self._workspace_id_hint: str | None = _workspace_id_hint

    # -------------------------------------------------------------------
    # Factories
    # -------------------------------------------------------------------

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        *,
        profile: str | None = None,
    ) -> Self:
        """Create a client by resolving credentials from the SDK chain.

        Precedence: explicit ``api_key`` > ``KAGURA_API_KEY`` env > OAuth
        profile from ``~/.kagura/credentials.json`` > ``.kagura.json``.
        The REST ``base_url`` is derived from the resolved ``mcp_url``
        (strips ``/mcp`` and any ``/mcp/w/{workspace_id}`` suffix); a
        static key bakes the Bearer header once, an OAuth profile
        installs a ``KaguraOAuth`` httpx.Auth handler with auto-refresh.

        Args:
            api_key: Explicit Kagura API key. Skips the resolution chain.
            mcp_url: Explicit MCP URL. When omitted, the resolved
                credential source's stored URL is used (default
                ``https://memory.kagura-ai.com/mcp``).
            timeout: API request timeout in seconds.
            profile: Named OAuth profile to load (overrides
                ``KAGURA_PROFILE`` env and the file's default).
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)
        return cls._from_resolved_auth(resolved, timeout=timeout)

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
        workspace_id_hint: str | None = None,
    ) -> Self:
        """Construct from a pre-resolved auth — internal CLI helper.

        Shared by :meth:`from_mcp_url` (SDK entry) and the CLI command
        runners, which resolve once so api_key and workspace can be
        paired from the same credential source (#115).

        ``workspace_id_hint`` threads the workspace bound to a static
        api_key into the client for 403 hint display. On the OAuth
        branch the resolver already carries ``workspace_id``, so the
        hint is taken from there and the parameter is ignored.
        """
        base_url = base_url_from_mcp(resolved.mcp_url.rstrip("/"))
        if isinstance(resolved, _StaticAuth):
            return cls(
                api_key=resolved.api_key,
                base_url=base_url,
                timeout=timeout,
                _auth_source=resolved.source,
                _workspace_id_hint=workspace_id_hint,
            )
        return cls(
            base_url=base_url,
            timeout=timeout,
            _oauth=resolved.oauth,
            _auth_source="oauth",
            _workspace_id_hint=resolved.workspace_id,
        )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        await self.close()

    # -------------------------------------------------------------------
    # Request spine
    # -------------------------------------------------------------------

    async def _request(
        self,
        method: HttpMethod,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make an authenticated request with standard error mapping."""
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(
                method, url, json=json, params=params, headers=extra_headers
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            self._raise_for_status_error(e, request_json=json, request_params=params)
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e

    def _raise_for_status_error(
        self,
        e: httpx.HTTPStatusError,
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> NoReturn:
        """Dispatch an HTTP status error to the per-status hooks."""
        status = e.response.status_code
        if status == 401:
            self._raise_401(e)
        if status == 403:
            self._raise_403(e, request_json=request_json, request_params=request_params)
        if status == 404:
            raise KaguraNotFoundError(extract_detail(e.response) or "Not found") from e
        if status == 429:
            self._raise_429(e)
        self._raise_generic(e)

    # ---- per-status hooks (override where the wire contract differs) ----

    def _raise_401(self, e: httpx.HTTPStatusError) -> NoReturn:
        """401 → auth error with a recovery hint matching the auth mode."""
        hint = (
            "Re-run `kagura auth login` or inspect ~/.kagura/credentials.json."
            if self._oauth is not None
            else "Check your API key."
        )
        raise KaguraAuthError(f"Authentication failed. {hint}") from e

    def _raise_403(
        self,
        e: httpx.HTTPStatusError,
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> NoReturn:
        """403 → generic mapping by default; clients with a richer story
        (workspace hints, secret existence-hiding) override this."""
        self._raise_generic(e)

    def _raise_429(self, e: httpx.HTTPStatusError) -> NoReturn:
        """429 → quota error with a tolerant ``Retry-After`` parse."""
        raise KaguraQuotaError(
            "Quota exceeded. Try again later.",
            retry_after=_retry_after_seconds(e.response),
        ) from e

    def _raise_generic(self, e: httpx.HTTPStatusError) -> NoReturn:
        status = e.response.status_code
        detail = extract_detail(e.response)
        msg = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
        raise KaguraConnectionError(msg) from e

    # -------------------------------------------------------------------
    # Response-body helpers
    # -------------------------------------------------------------------

    def _json(self, resp: httpx.Response) -> Any:
        """Parse a 2xx body as JSON, mapping garbage to a Kagura error.

        A proxy/CDN can 200 with an HTML maintenance page; that must not
        surface as a raw ``json.JSONDecodeError`` traceback.
        """
        try:
            return resp.json()
        except ValueError as exc:
            raise KaguraConnectionError(
                f"Server returned a non-JSON body (HTTP {resp.status_code}) for "
                f"{resp.request.method} {resp.request.url.path}."
            ) from exc

    def _expect_list(self, resp: httpx.Response) -> list[Any]:
        """Parse a 2xx body that the contract says is a JSON array."""
        payload = self._json(resp)
        if not isinstance(payload, list):
            raise KaguraConnectionError(
                f"Unexpected response shape for {resp.request.method} "
                f"{resp.request.url.path}: expected a JSON array, got "
                f"{type(payload).__name__}."
            )
        return payload
