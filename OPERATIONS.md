# Irminsul Operations

## Runtime

This repository is designed for one machine, one Uvicorn process, one Uvicorn worker, and one in-process background job worker.

Install dependencies:

```bash
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
```

Run:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not run multiple Uvicorn workers against the same SQLite database.

## Admin Auth

Create the credential files configured in `config.toml` before starting the service:

```bash
mkdir -p data
python - <<'PY' > data/admin_password.hash
from core import hash_password
print(hash_password("replace-me"))
PY
python - <<'PY' > data/admin_signing_secret
import secrets
print(secrets.token_urlsafe(48))
PY
chmod 600 data/admin_password.hash data/admin_signing_secret
```

Use your real admin password instead of `replace-me`.

The admin UI is served at `/admin`.

## Data Flow

1. Upload JSONL in the admin Imports page.
2. Run dry-run and confirm the import.
3. Build an index from the Indexes page.
4. Wait for the build job to finish.
5. Activate the built index.
6. Public search uses only the active in-memory or mmap-loaded cache.

JSONL input rows use:

```json
{"id":"CodeForces/1A","title":"Theatre Square","text":"...","url":"https://..."}
```

## Backup

Back up these paths together:

```text
data/app.sqlite3
data/app.sqlite3-wal
data/app.sqlite3-shm
data/uploads/
data/index_cache/
config.toml
```

Do not commit `data/`, `*.sqlite3`, `*.npy`, `frontend/dist`, or secrets.

## Recovery

On startup, the service moves interrupted running jobs back to `queued`, moves running artifacts back to `pending`, removes stale `.building` cache directories, loads the active index from `kv.active_index_key`, and starts the single job worker.

Use the Jobs page to inspect failed or blocked jobs. Blocked index builds keep their original snapshot and can be retried.
