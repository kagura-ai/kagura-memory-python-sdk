---
name: files
description: Upload, list, delete, and get download URLs for files in Kagura Memory's R2 storage (sha256 integrity-bound). Use when the user wants to attach or manage raw files in their workspace.
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
kagura files list                       # list stored files
kagura files download-url <file_id>     # short-lived GET URL
kagura files delete <file_id>           # delete a file
```

## Consume the result

- Relay the returned file id and download URL. **Download URLs are short-lived**
  — do not cache them; re-run `download-url` when one expires.
