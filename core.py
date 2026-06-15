from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import tomllib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_CONFIG_PATH = SRC_DIR / "config.toml"
SCHEMA_VERSION = 2


class WriterPriorityRwLock:
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._condition:
            while self._writer_active or self._writers_waiting > 0:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._condition.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer_active = False
                self._condition.notify_all()


_DB_RW_LOCK = WriterPriorityRwLock()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    url: str
    api_key_env: str
    identity: str = ""
    provider: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            object.__setattr__(self, "identity", self.model)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "").strip()

    @property
    def resolved_url(self) -> str:
        return self.url.format(model=self.model)


@dataclass(frozen=True)
class StorageConfig:
    db_path: Path
    upload_dir: Path
    index_cache_dir: Path


@dataclass(frozen=True)
class AdminConfig:
    session_hours: int
    password_hash_file: Path
    signing_secret_file: Path


@dataclass(frozen=True)
class LimitsConfig:
    upload_max_bytes: int
    jsonl_max_line_bytes: int
    field_max_text_chars: int


@dataclass(frozen=True)
class JobsConfig:
    poll_seconds: int
    rewrite_concurrency: int
    embedding_concurrency: int
    embedding_batch_size: int


@dataclass(frozen=True)
class SearchConfig:
    top_per_doc_view: int
    top_retrieval: int
    top_display: int
    rerank_top_k: int
    beta: float
    default_rerank: bool
    rerank_range_floor: float
    embedding_range_floor: float


@dataclass(frozen=True)
class IndexCacheConfig:
    keep_retired: int
    load_mode: str
    activation_drain_timeout_seconds: int


@dataclass(frozen=True)
class AuditConfig:
    retention_days: int
    pricing: dict[str, Any]


@dataclass(frozen=True)
class Settings:
    storage: StorageConfig
    admin: AdminConfig
    limits: LimitsConfig
    jobs: JobsConfig
    search: SearchConfig
    index_cache: IndexCacheConfig
    audit: AuditConfig
    request_timeout: int
    rewrite_model: ModelConfig
    embedding_model: ModelConfig
    rerank_model: ModelConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_dotenv(
    paths: tuple[Path, ...] = (
        SRC_DIR / ".env",
        PROJECT_ROOT / ".env",
    ),
) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return SRC_DIR / path


def _model_config(name: str, data: dict[str, Any]) -> ModelConfig:
    missing = [key for key in ("model", "url", "api_key_env") if not data.get(key)]
    if missing:
        raise ValueError(f"models.{name} missing required keys: {', '.join(missing)}")
    return ModelConfig(
        name=name,
        model=str(data["model"]),
        url=str(data["url"]),
        api_key_env=str(data["api_key_env"]),
        identity=str(data.get("identity") or data["model"]),
        provider=dict(data["provider"]) if isinstance(data.get("provider"), dict) else None,
    )


def get_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    load_dotenv()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    storage = raw.get("storage", {})
    admin = raw.get("admin", {})
    limits = raw.get("limits", {})
    jobs = raw.get("jobs", {})
    search = raw.get("search", raw.get("retrieval", {}))
    index_cache = raw.get("index_cache", {})
    audit = raw.get("audit", {})
    models = raw.get("models", {})

    return Settings(
        storage=StorageConfig(
            db_path=resolve_path(str(storage.get("db_path", "data/app.sqlite3"))),
            upload_dir=resolve_path(str(storage.get("upload_dir", "data/uploads"))),
            index_cache_dir=resolve_path(
                str(storage.get("index_cache_dir", "data/index_cache"))
            ),
        ),
        admin=AdminConfig(
            session_hours=int(admin.get("session_hours", 8)),
            password_hash_file=resolve_path(
                str(admin.get("password_hash_file", "data/admin_password.hash"))
            ),
            signing_secret_file=resolve_path(
                str(admin.get("signing_secret_file", "data/admin_signing_secret"))
            ),
        ),
        limits=LimitsConfig(
            upload_max_bytes=int(limits.get("upload_max_bytes", 104_857_600)),
            jsonl_max_line_bytes=int(limits.get("jsonl_max_line_bytes", 1_048_576)),
            field_max_text_chars=int(limits.get("field_max_text_chars", 200_000)),
        ),
        jobs=JobsConfig(
            poll_seconds=int(jobs.get("poll_seconds", 2)),
            rewrite_concurrency=int(jobs.get("rewrite_concurrency", 16)),
            embedding_concurrency=int(jobs.get("embedding_concurrency", 4)),
            embedding_batch_size=int(jobs.get("embedding_batch_size", 128)),
        ),
        search=SearchConfig(
            top_per_doc_view=int(search.get("top_per_doc_view", 50)),
            top_retrieval=int(search.get("top_retrieval", 200)),
            top_display=int(search.get("top_display", 20)),
            rerank_top_k=int(search.get("rerank_top_k", 0)),
            beta=float(search.get("beta", search.get("default_beta", 0.75))),
            default_rerank=bool(search.get("default_rerank", True)),
            rerank_range_floor=float(search.get("rerank_range_floor", 0.1)),
            embedding_range_floor=float(search.get("embedding_range_floor", 0.05)),
        ),
        index_cache=IndexCacheConfig(
            keep_retired=int(index_cache.get("keep_retired", 3)),
            load_mode=str(index_cache.get("load_mode", "mmap")),
            activation_drain_timeout_seconds=int(
                index_cache.get("activation_drain_timeout_seconds", 30)
            ),
        ),
        audit=AuditConfig(
            retention_days=int(audit.get("retention_days", 9999)),
            pricing=(
                dict(audit.get("pricing", {}))
                if isinstance(audit.get("pricing", {}), dict)
                else {}
            ),
        ),
        request_timeout=int(raw.get("api", {}).get("request_timeout", 240)),
        rewrite_model=_model_config("rewrite", models.get("rewrite", {})),
        embedding_model=_model_config("embedding", models.get("embedding", {})),
        rerank_model=_model_config("rerank", models.get("rerank", {})),
    )


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def canonical_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def text_key(text: str) -> str:
    return "t:" + sha256_hex(canonical_text(text))


def method_key(kind: str, config: dict[str, Any] | ModelConfig) -> str:
    if isinstance(config, ModelConfig):
        payload: dict[str, Any] = {
            "kind": kind,
            "model": config.identity,
        }
    else:
        payload = {"kind": kind, **config}
    return "m:" + sha256_hex(json_dumps(payload))


def rewrite_key(parent_text_key: str, rewrite_method_key: str) -> str:
    return "r:" + sha256_hex(f"rewrite\n{parent_text_key}\n{rewrite_method_key}\nrewrite")


def embedding_key(parent_rewrite_key: str, embedding_method_key: str, view: str) -> str:
    return "e:" + sha256_hex(
        f"embedding\n{parent_rewrite_key}\n{embedding_method_key}\n{view}"
    )


def row_hash(
    problem_ord: int,
    problem_key: str,
    view: str,
    row_embedding_key: str,
    title: str,
    url: str,
    row_text_key: str,
    row_rewrite_key: str,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "problem_ord": problem_ord,
        "problem_key": problem_key,
        "view": view,
        "embedding_key": row_embedding_key,
        "title": title,
        "url": url,
        "text_key": row_text_key,
        "rewrite_key": row_rewrite_key,
    }
    return sha256_hex(json_dumps(payload))


def index_key(
    row_hashes: list[str],
    rewrite_method_key: str,
    embedding_method_key: str,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "rewrite_method_key": rewrite_method_key,
        "embedding_method_key": embedding_method_key,
        "row_hashes": sorted(row_hashes),
    }
    return "i:" + sha256_hex(json_dumps(payload))


def hash_password(password: str, iterations: int = 310_000) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_text, salt, expected = password_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode(), iterations
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


@contextmanager
def db_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    conn = connect_db(settings.storage.db_path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def db_read_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    with _DB_RW_LOCK.read_lock():
        conn = connect_db(settings.storage.db_path)
        try:
            yield conn
        finally:
            conn.close()


@contextmanager
def db_write_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    with _DB_RW_LOCK.write_lock():
        conn = connect_db(settings.storage.db_path)
        try:
            yield conn
        finally:
            conn.close()


def migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"Database schema is newer than this application: {version}")
    if version == 0:
        _migrate_0_to_1(conn)
        version = 1
    if version == 1:
        _migrate_1_to_2(conn)
        version = 2
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported database schema version: {version}")


def ensure_database(settings: Settings) -> None:
    with db_connection(settings) as conn:
        migrate(conn)
        ensure_query_indexes(conn)


def ensure_query_indexes(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_artifacts_kind_method_status_parent
              ON artifacts(kind, method_key, status, parent_key);

            CREATE INDEX IF NOT EXISTS idx_artifacts_parent_kind_method_status_role
              ON artifacts(parent_key, kind, method_key, status, role);

            CREATE INDEX IF NOT EXISTS idx_problems_enabled_deleted_text
              ON problems(enabled, deleted, text_key);
            """
        )


def _migrate_0_to_1(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE sources (
              key TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE artifacts (
              key TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK(kind IN ('problem_text','rewrite','embedding')),
              parent_key TEXT REFERENCES artifacts(key),
              method_key TEXT,
              role TEXT,
              text TEXT,
              data TEXT,
              blob BLOB,
              status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
              attempts INTEGER NOT NULL DEFAULT 0,
              error TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(kind, parent_key, method_key, role)
            );

            CREATE TABLE problems (
              key TEXT PRIMARY KEY,
              source_key TEXT NOT NULL REFERENCES sources(key),
              title TEXT NOT NULL,
              url TEXT NOT NULL,
              text_key TEXT NOT NULL REFERENCES artifacts(key),
              enabled INTEGER NOT NULL DEFAULT 1,
              deleted INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE indexes (
              key TEXT PRIMARY KEY,
              status TEXT NOT NULL CHECK(status IN ('building','built','active','retired','failed')),
              meta TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT,
              error TEXT
            );

            CREATE TABLE index_rows (
              index_key TEXT NOT NULL REFERENCES indexes(key),
              problem_ord INTEGER NOT NULL,
              problem_key TEXT NOT NULL,
              view TEXT NOT NULL,
              embedding_key TEXT NOT NULL REFERENCES artifacts(key),
              title TEXT NOT NULL,
              url TEXT NOT NULL,
              text_key TEXT NOT NULL,
              rewrite_key TEXT NOT NULL,
              row_hash TEXT NOT NULL,
              PRIMARY KEY(index_key, problem_ord, view),
              UNIQUE(index_key, problem_key, view)
            );

            CREATE TABLE jobs (
              key TEXT PRIMARY KEY,
              type TEXT NOT NULL CHECK(type IN ('import','build_index','activate_index','cleanup')),
              status TEXT NOT NULL CHECK(status IN ('draft','queued','running','succeeded','blocked','failed')),
              payload TEXT NOT NULL,
              progress TEXT NOT NULL,
              result TEXT,
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE search_audits (
              request_id TEXT PRIMARY KEY,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              status TEXT NOT NULL,
              client_ip TEXT,
              user_agent TEXT,
              query TEXT NOT NULL,
              timings TEXT NOT NULL,
              api_calls TEXT NOT NULL,
              result TEXT NOT NULL,
              cost TEXT NOT NULL,
              error TEXT
            );

            CREATE TABLE kv (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE INDEX idx_problems_filter ON problems(source_key, enabled, deleted);
            CREATE INDEX idx_artifacts_lookup ON artifacts(kind, parent_key, method_key, role, status);
            CREATE INDEX idx_jobs_queue ON jobs(status, created_at);
            CREATE INDEX idx_audits_time ON search_audits(started_at);

            PRAGMA user_version=1;
            """
        )


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("ALTER TABLE artifacts DROP COLUMN attempts")
        conn.executescript(
            """
            CREATE TABLE job_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_key TEXT NOT NULL REFERENCES jobs(key) ON DELETE CASCADE,
              level TEXT NOT NULL CHECK(level IN ('info','warning','error')),
              message TEXT NOT NULL,
              data TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX idx_job_logs_job ON job_logs(job_key, id);

            PRAGMA user_version=2;
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def get_job(settings: Settings, job_key: str) -> dict[str, Any] | None:
    with db_read_connection(settings) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE key = ?", (job_key,)).fetchone()
        return row_to_dict(row) if row else None


def list_import_jobs(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE type = 'import'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def list_indexes(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM indexes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def get_index(settings: Settings, selected_index_key: str) -> dict[str, Any] | None:
    with db_read_connection(settings) as conn:
        row = conn.execute("SELECT * FROM indexes WHERE key = ?", (selected_index_key,)).fetchone()
        return row_to_dict(row) if row else None


def list_problems(
    settings: Settings,
    source_key: str | None = None,
    enabled: bool | None = None,
    deleted: bool | None = None,
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    where = ["1 = 1"]
    params: list[Any] = []
    if source_key:
        where.append("source_key = ?")
        params.append(source_key)
    if enabled is not None:
        where.append("enabled = ?")
        params.append(1 if enabled else 0)
    if deleted is not None:
        where.append("deleted = ?")
        params.append(1 if deleted else 0)
    if q.strip():
        where.append("(key LIKE ? OR title LIKE ? OR url LIKE ?)")
        needle = f"%{q.strip()}%"
        params.extend([needle, needle, needle])

    where_sql = " AND ".join(where)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with db_read_connection(settings) as conn:
        total = conn.execute(
            f"SELECT count(*) FROM problems WHERE {where_sql}",
            params,
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT key, source_key, title, url, text_key, enabled, deleted, updated_at
            FROM problems
            WHERE {where_sql}
            ORDER BY updated_at DESC, key
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    return {"total": total, "items": [row_to_dict(row) for row in rows]}


def get_problem(settings: Settings, problem_key: str) -> dict[str, Any] | None:
    with db_read_connection(settings) as conn:
        row = conn.execute(
            """
            SELECT
              p.key, p.source_key, p.title, p.url, p.text_key,
              p.enabled, p.deleted, p.updated_at,
              a.text,
              a.status AS text_status,
              a.error AS text_error,
              a.updated_at AS text_updated_at
            FROM problems p
            JOIN artifacts a ON a.key = p.text_key AND a.kind = 'problem_text'
            WHERE p.key = ?
            """,
            (problem_key,),
        ).fetchone()
        if row is None:
            return None
        problem = row_to_dict(row)
        rewrites = [
            row_to_dict(rewrite)
            for rewrite in conn.execute(
                """
                SELECT key, method_key, status, error, updated_at
                FROM artifacts
                WHERE kind = 'rewrite'
                  AND parent_key = ?
                ORDER BY updated_at DESC, key
                """,
                (problem["text_key"],),
            ).fetchall()
        ]
        for rewrite in rewrites:
            embeddings = [
                row_to_dict(embedding)
                for embedding in conn.execute(
                    """
                    SELECT key, method_key, role, status, error, updated_at
                    FROM artifacts
                    WHERE kind = 'embedding'
                      AND parent_key = ?
                    ORDER BY role
                    """,
                    (rewrite["key"],),
                ).fetchall()
            ]
            rewrite["embeddings"] = embeddings
        problem["artifacts"] = {
            "problem_text": {
                "key": problem["text_key"],
                "status": problem["text_status"],
                "error": problem["text_error"],
                "updated_at": problem["text_updated_at"],
            },
            "rewrites": rewrites,
        }
    return problem


def patch_problem(settings: Settings, problem_key: str, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = {"title", "url", "enabled", "deleted"}
    assignments: list[str] = []
    params: list[Any] = []
    now = utc_now()
    for key, value in changes.items():
        if key not in allowed or value is None:
            continue
        assignments.append(f"{key} = ?")
        params.append(int(value) if key in {"enabled", "deleted"} else str(value))
    if "text" in changes and changes["text"] is not None:
        text = canonical_text(str(changes["text"]))
        if not text:
            raise ValueError("text is required")
        if len(text) > settings.limits.field_max_text_chars:
            raise ValueError("text is too long")
        new_text_key = text_key(text)
        assignments.append("text_key = ?")
        params.append(new_text_key)
    if not assignments:
        raise ValueError("no valid problem changes")
    assignments.append("updated_at = ?")
    params.extend([now, problem_key])
    with db_write_connection(settings) as conn:
        with conn:
            if "text" in changes and changes["text"] is not None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artifacts(
                      key, kind, parent_key, method_key, role, text, data, blob,
                      status, error, updated_at
                    )
                    VALUES (?, 'problem_text', NULL, NULL, NULL, ?, NULL, NULL,
                            'succeeded', NULL, ?)
                    """,
                    (new_text_key, text, now),
                )
            cursor = conn.execute(
                f"UPDATE problems SET {', '.join(assignments)} WHERE key = ?",
                params,
            )
        if cursor.rowcount == 0:
            raise ValueError("problem not found")
    updated = get_problem(settings, problem_key)
    if updated is None:
        raise ValueError("problem not found")
    return updated


def batch_update_problems(settings: Settings, keys: list[str], action: str) -> dict[str, Any]:
    if not keys:
        raise ValueError("keys are required")
    if action not in {"enable", "disable", "delete", "restore"}:
        raise ValueError("invalid batch action")
    if action == "enable":
        assignments = "enabled = 1"
    elif action == "disable":
        assignments = "enabled = 0"
    elif action == "delete":
        assignments = "deleted = 1"
    else:
        assignments = "deleted = 0"

    placeholders = ",".join("?" for _ in keys)
    with db_write_connection(settings) as conn:
        with conn:
            cursor = conn.execute(
                f"""
                UPDATE problems
                SET {assignments}, updated_at = ?
                WHERE key IN ({placeholders})
                """,
                [utc_now(), *keys],
            )
    return {"updated": cursor.rowcount}


def list_sources(settings: Settings) -> list[dict[str, Any]]:
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT
              s.key, s.name, s.enabled, s.updated_at,
              count(p.key) AS problem_count,
              sum(CASE WHEN p.enabled = 1 AND p.deleted = 0 THEN 1 ELSE 0 END) AS enabled_problem_count,
              sum(CASE WHEN p.deleted = 1 THEN 1 ELSE 0 END) AS deleted_count
            FROM sources s
            LEFT JOIN problems p ON p.source_key = s.key
            GROUP BY s.key, s.name, s.enabled, s.updated_at
            ORDER BY s.key
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def patch_source(settings: Settings, source_key: str, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = {"name", "enabled"}
    assignments: list[str] = []
    params: list[Any] = []
    for key, value in changes.items():
        if key not in allowed or value is None:
            continue
        assignments.append(f"{key} = ?")
        params.append(int(value) if key == "enabled" else str(value))
    if not assignments:
        raise ValueError("no valid source changes")
    assignments.append("updated_at = ?")
    params.extend([utc_now(), source_key])
    with db_write_connection(settings) as conn:
        with conn:
            cursor = conn.execute(
                f"UPDATE sources SET {', '.join(assignments)} WHERE key = ?",
                params,
            )
        if cursor.rowcount == 0:
            raise ValueError("source not found")
        row = conn.execute("SELECT * FROM sources WHERE key = ?", (source_key,)).fetchone()
    return row_to_dict(row)


def list_jobs(
    settings: Settings,
    job_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    where = ["1 = 1"]
    params: list[Any] = []
    if job_type:
        where.append("type = ?")
        params.append(job_type)
    if status:
        where.append("status = ?")
        params.append(status)
    limit = max(1, min(limit, 500))
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def append_job_log(
    settings: Settings,
    job_key: str,
    level: Literal["info", "warning", "error"],
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO job_logs(job_key, level, message, data, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_key, level, message, json_dumps(data) if data is not None else None, utc_now()),
            )


def list_job_logs(settings: Settings, job_key: str, limit: int = 500) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 2000))
    with db_read_connection(settings) as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_logs
            WHERE job_key = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (job_key, limit),
        ).fetchall()
    return list(reversed([row_to_dict(row) for row in rows]))


def retry_job(settings: Settings, job_key: str) -> dict[str, Any]:
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("job not found")
    if job["status"] not in {"blocked", "failed"}:
        raise ValueError("job is not retryable")
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', progress = ?, result = NULL, error = NULL, updated_at = ?
                WHERE key = ?
                """,
                (json_dumps({"phase": "queued"}), utc_now(), job_key),
            )
    append_job_log(settings, job_key, "info", "Job queued for retry")
    return get_job(settings, job_key) or job


def cancel_job(settings: Settings, job_key: str) -> dict[str, Any]:
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("job not found")
    if job["status"] in {"succeeded", "blocked", "failed"}:
        raise ValueError("job is not running or queued")

    progress = job_progress(job)
    progress["cancel_requested"] = True
    progress.setdefault("phase", "canceling")
    result = job_result(job)
    result["canceled"] = True
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            if job["status"] in {"draft", "queued"}:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', progress = ?, result = ?, error = ?, updated_at = ?
                    WHERE key = ?
                    """,
                    (
                        json_dumps(progress),
                        json_dumps(result),
                        "Canceled by admin",
                        now,
                        job_key,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET progress = ?, result = ?, updated_at = ?
                    WHERE key = ? AND status = 'running'
                    """,
                    (json_dumps(progress), json_dumps(result), now, job_key),
                )
    append_job_log(settings, job_key, "warning", "Cancellation requested")
    updated = get_job(settings, job_key)
    if updated is None:
        raise ValueError("job not found after cancel")
    return updated


def create_cleanup_job(settings: Settings) -> dict[str, Any]:
    job_key = "j:" + uuid.uuid4().hex
    now = utc_now()
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs(key, type, status, payload, progress, created_at, updated_at)
                VALUES (?, 'cleanup', 'queued', ?, ?, ?, ?)
                """,
                (job_key, json_dumps({}), json_dumps({"phase": "queued"}), now, now),
            )
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("cleanup job was not created")
    return job


def mark_job_failed(settings: Settings, job_key: str, error: str) -> None:
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, updated_at = ?
                WHERE key = ? AND status IN ('queued', 'running')
                """,
                (error[:4000], utc_now(), job_key),
            )


def job_progress(job: dict[str, Any]) -> dict[str, Any]:
    raw = job.get("progress")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            return {}
    return {}


def job_result(job: dict[str, Any]) -> dict[str, Any]:
    raw = job.get("result")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            return {}
    return {}


def cancel_requested(settings: Settings, job_key: str) -> bool:
    job = get_job(settings, job_key)
    return bool(job and job_progress(job).get("cancel_requested"))


def finish_if_cancel_requested(
    settings: Settings,
    job_key: str,
    result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not cancel_requested(settings, job_key):
        return None
    payload = dict(result or {})
    payload["canceled"] = True
    append_job_log(settings, job_key, "warning", "Job canceled")
    return finish_job(settings, job_key, "failed", payload, "Canceled by admin")


def update_job_progress(
    settings: Settings,
    job_key: str,
    progress: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> None:
    with db_write_connection(settings) as conn:
        with conn:
            row = conn.execute("SELECT progress FROM jobs WHERE key = ?", (job_key,)).fetchone()
            if row is not None:
                current_progress = job_progress(row_to_dict(row))
                if current_progress.get("cancel_requested") and not progress.get("cancel_requested"):
                    progress = dict(progress)
                    progress["cancel_requested"] = True
            if result is None:
                conn.execute(
                    "UPDATE jobs SET progress = ?, updated_at = ? WHERE key = ?",
                    (json_dumps(progress), utc_now(), job_key),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET progress = ?, result = ?, updated_at = ? WHERE key = ?",
                    (json_dumps(progress), json_dumps(result), utc_now(), job_key),
                )


def finish_job(
    settings: Settings,
    job_key: str,
    status: Literal["succeeded", "blocked", "failed"],
    result: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    with db_write_connection(settings) as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE key = ?
                """,
                (status, json_dumps(result), error, utc_now(), job_key),
            )
    job = get_job(settings, job_key)
    if job is None:
        raise ValueError("job not found after finish")
    return job
