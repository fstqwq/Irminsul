# Irminsul

Competitive programming problem search engine with four-view semantic retrieval. Supports incremental data maintenance -- problems, rewrites, and embeddings are managed independently and composed into immutable indexes on demand.

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

### Create credentials

```bash
mkdir -p data

# Admin password hash
python -c "from core import hash_password; print(hash_password('replace-me'))" > data/admin_password.hash

# Session signing secret
python -c "import secrets; print(secrets.token_urlsafe(48))" > data/admin_signing_secret

# Linux/macOS only
chmod 600 data/admin_password.hash data/admin_signing_secret
```

Use your real admin password instead of `replace-me`.

### Configure API keys

Create a `.env` file:

```env
DEEPSEEK_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
DEEPINFRA_API_KEY=...
```

### Run

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

- Public search: `http://localhost:8000/`
- Admin UI: `http://localhost:8000/admin`

> **Do not** run multiple Uvicorn workers against the same SQLite database.

## Data Flow

```mermaid
flowchart LR
    subgraph SQLite
        P["problems\n+ sources"]
        R["rewrite\nartifacts"]
        E["embedding\nartifacts"]
        I["index_rows"]
    end

    JSONL -->|import| P
    P -->|LLM| R
    R -->|embed| E
    P & R & E -->|build| I

    subgraph Disk Cache
        NPY[".npy x 4 views\nproblems.jsonl\nviews.jsonl"]
    end

    I -->|export| NPY
    NPY -->|activate + search| Client(["Client"])
```

Admin workflow: upload JSONL -> review dry-run -> confirm import -> build index -> activate. Each step is incremental -- re-importing updates only changed problems, and rebuilding reuses existing rewrites and embeddings.

### JSONL format

```json
{"id": "CodeForces/1A", "title": "Theatre Square", "text": "...", "url": "https://..."}
```

| Field   | Required | Description                      |
|---------|----------|----------------------------------|
| `id`    | yes      | Unique problem identifier        |
| `title` | no       | Display title (defaults to `id`) |
| `text`  | yes      | Problem statement                |
| `url`   | no       | Link to original problem         |

## Configuration

All configuration is in `config.toml`. See the file for the full reference. Key sections:

| Section        | Purpose                              |
|----------------|--------------------------------------|
| `[storage]`    | Database and cache paths             |
| `[admin]`      | Session duration, credential files   |
| `[models.*]`   | LLM / embedding / rerank endpoints   |
| `[search]`     | Retrieval limits, beta, rerank       |
| `[index_cache]`| `load_mode` (`mmap` or `ram`)        |
| `[audit]`      | Search audit retention and pricing   |

## Backup

Back up these paths together:

```
data/app.sqlite3
data/app.sqlite3-wal
data/app.sqlite3-shm
data/uploads/
data/index_cache/
config.toml
```

Do not commit `data/`, `*.sqlite3`, `*.npy`, `frontend/dist`, or credential files.

## Recovery

On startup the service automatically:

- Moves interrupted `running` jobs back to `queued`
- Resets `running` artifacts back to `pending`
- Removes stale `.building` cache directories
- Loads the active index from the database
- Starts the background job worker

Use the admin **Jobs** page to inspect failed or blocked jobs.

## Development

### Tests

```bash
python -m pytest tests -q -p no:cacheprovider
```

### Frontend

```bash
cd frontend && npm run build
```

### Architecture

Single process, single SQLite database (WAL mode), single background job worker thread.

```
app.py       - FastAPI routes, admin auth, index activation
core.py      - Settings, DB helpers, schema, CRUD
pipeline.py  - Import, rewrite, embedding, build index, cache export
search.py    - API clients, search pipeline, audit
frontend/    - Vanilla TypeScript + Vite, PicoCSS admin UI
```
