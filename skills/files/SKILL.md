---
name: files
description: Upload, list, delete, and get download URLs for files in Kagura Memory's R2 storage (sha256 integrity-bound; an upload can optionally be bound to an owning context for access control). Use when the user wants to attach or manage raw files in their workspace.
---

# kagura files — R2 file storage

Drive the `kagura files` command group (presigned-PUT uploads with sha256
binding). Thin wrapper around the installed `kagura` CLI.

## Preflight

- `kagura --version`; if missing → `uv tool install kagura-memory` (or
  `pip install kagura-memory`), then stop.
- Requires authentication. Run `kagura auth status`; if not authed, run the
  `auth` skill first.

## Run (choose by intent)

```bash
kagura files upload ./data.bin          # upload (sha256-bound presigned PUT)
kagura files upload ./data.bin --binding-context-id <ctx>   # bind the file to an owning context for ACL (server v0.41.0+)
kagura files list                       # list stored files
kagura files download-url <file_id> -c <ctx>   # short-lived GET URL (context required, server v0.41.0+)
kagura files delete <file_id> -c <ctx>         # delete a file (context required, server v0.41.0+)
```

## Consume the result

- Relay the returned file id and download URL. **Download URLs are short-lived**
  — do not cache them; re-run `download-url` when one expires.
- `--binding-context-id` (the file's owning context for ACL) is **distinct from**
  `--context-id` (the workspace). Bound files route read/write/list/delete through
  that context's ACL; an unbound upload stays workspace-scoped. A denied download
  is reported as a 404 (existence-hiding), not a 403.
