"""Integration tests for FilesClient — gated by environment.

Skipped unless ``KAGURA_INTEGRATION_URL`` is set (and the matching API
key + context UUID are available). Runs a single round-trip against a
local memory-cloud configured with ``R2_CHECKSUM_BINDING_ENABLED=true``
so the SDK's ``x-amz-checksum-sha256`` header is actually validated by
R2 on the wire.

Local run:

    export KAGURA_INTEGRATION_URL=http://localhost:8080
    export KAGURA_API_KEY=kagura_...
    export KAGURA_INTEGRATION_CONTEXT_ID=<workspace_uuid>
    uv run pytest -m integration -v
"""

import hashlib
import os
import uuid

import pytest

from kagura_memory import FilesClient

INTEGRATION_URL = os.environ.get("KAGURA_INTEGRATION_URL", "")
INTEGRATION_API_KEY = os.environ.get("KAGURA_API_KEY", "")
INTEGRATION_CONTEXT_ID = os.environ.get("KAGURA_INTEGRATION_CONTEXT_ID", "")


@pytest.mark.integration
@pytest.mark.skipif(
    not (INTEGRATION_URL and INTEGRATION_API_KEY and INTEGRATION_CONTEXT_ID),
    reason=(
        "KAGURA_INTEGRATION_URL, KAGURA_API_KEY, and KAGURA_INTEGRATION_CONTEXT_ID must all be set"
    ),
)
@pytest.mark.asyncio
async def test_upload_round_trip_against_live_backend():
    """upload → download_url → list → delete against a live backend."""
    body = f"kagura integration test {uuid.uuid4()}".encode()
    sha256_hex = hashlib.sha256(body).hexdigest()

    async with FilesClient(api_key=INTEGRATION_API_KEY, base_url=INTEGRATION_URL) as client:
        # Upload
        file_obj = await client.upload(
            context_id=INTEGRATION_CONTEXT_ID,
            source=body,
            filename=f"integration_{uuid.uuid4().hex[:8]}.txt",
            content_type="text/plain",
        )
        assert file_obj.sha256 == sha256_hex
        assert file_obj.size_bytes == len(body)
        assert file_obj.status in ("uploaded", "confirmed")

        # Download URL — context_id (workspace) required as of server v0.41.0.
        url = await client.download_url(file_obj.id, context_id=INTEGRATION_CONTEXT_ID)
        assert url.startswith("http")

        # List — uploaded file should appear in the workspace
        page = await client.list(context_id=INTEGRATION_CONTEXT_ID, limit=50)
        assert any(f.id == file_obj.id for f in page.files)

        # Clean up
        await client.delete(file_obj.id, context_id=INTEGRATION_CONTEXT_ID)
