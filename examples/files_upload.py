#!/usr/bin/env python3
"""FilesClient usage — upload, download-url, list, and delete.

Files are uploaded to the workspace's object store via short-lived
presigned PUT URLs. The SDK binds the body's sha256 into the PUT
signature so a tamper-aware server (memory-cloud v0.15.1+,
``R2_CHECKSUM_BINDING_ENABLED=true``) can reject corrupted uploads.

Usage:
    export KAGURA_API_KEY="kagura_..."
    export KAGURA_MCP_URL="http://localhost:8080/mcp/w/{workspace_id}"
    uv run python examples/files_upload.py
"""

import asyncio
import os
import sys

from kagura_memory import FilesClient, KaguraClient


async def main():
    api_key = os.getenv("KAGURA_API_KEY")
    mcp_url = os.getenv("KAGURA_MCP_URL", "http://localhost:8080/mcp")

    if not api_key:
        print("Error: KAGURA_API_KEY required")
        sys.exit(1)

    # Pick a context to attach the file to.
    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        contexts = await client.list_contexts()
        if not contexts.get("contexts"):
            print("No contexts found. Create one first.")
            return
        ctx = contexts["contexts"][0]["id"]
        print(f"Using context: {ctx}")

    async with FilesClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) as files:
        # Upload from bytes — filename is required when source is bytes.
        obj = await files.upload(
            context_id=ctx,
            source=b"hello from the Kagura SDK example",
            filename="example.txt",
        )
        print(f"Uploaded: id={obj.id} sha256={obj.sha256[:12]}... size={obj.size_bytes}B")
        # Unbound upload → context_id is None (workspace-scoped, fully listable).
        print(f"Binding context: {obj.context_id}")

        # Optional: bind a file to an owning context for access control
        # (server v0.41.0+). binding_context_id is the wire `context_id` —
        # distinct from `context_id` above (the workspace). The bound context
        # must be a write-accessible context within the workspace (else
        # 403 / 422), so this is shown as a snippet rather than executed:
        #
        #   bound = await files.upload(
        #       context_id=ctx,
        #       source=b"...",
        #       filename="bound.txt",
        #       binding_context_id="<owning-context-uuid>",
        #   )
        #   print(bound.context_id)  # -> "<owning-context-uuid>"

        # Re-uploading identical bytes is idempotent — returns the existing object.
        dup = await files.upload(
            context_id=ctx,
            source=b"hello from the Kagura SDK example",
            filename="example.txt",
        )
        print(f"Dedup happy-path: same id? {dup.id == obj.id}")

        # Short-lived presigned GET URL for the original bytes.
        url = await files.download_url(obj.id)
        print(f"Download URL (expires shortly): {url[:60]}...")

        # List files in the context, newest first.
        page = await files.list(context_id=ctx, limit=10)
        print(f"Files in context: {len(page.files)}")

        # Cleanup.
        await files.delete(obj.id)
        print(f"Deleted: {obj.id}")


if __name__ == "__main__":
    asyncio.run(main())
