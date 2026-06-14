# Yuantiji Operations

## Runtime

This repository is designed for one machine, one Uvicorn process, one Uvicorn worker, and one in-process background job worker.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
```

Run:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not run multiple Uvicorn workers against the same SQLite database.

## Admin Auth

Set both variables before starting the service:

```powershell
$env:YUANTIJI_ADMIN_PASSWORD_HASH = "<pbkdf2 hash>"
$env:YUANTIJI_ADMIN_SIGNING_SECRET = "<long random secret>"
```

Generate a password hash:

```powershell
@'
from core import hash_password
print(hash_password("replace-me"))
'@ | python -
```

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
