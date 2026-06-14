from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROTO_DIR = PROJECT_ROOT / "proto"
DEFAULT_CONFIG_PATH = SRC_DIR / "config.toml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    url: str
    api_key_env: str

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
    legacy_data_dir: Path


@dataclass(frozen=True)
class AdminConfig:
    session_hours: int
    password_hash_env: str
    signing_secret_env: str


@dataclass(frozen=True)
class LimitsConfig:
    upload_max_bytes: int
    jsonl_max_line_bytes: int
    field_max_text_chars: int


@dataclass(frozen=True)
class JobsConfig:
    poll_seconds: int
    rewrite_max_attempts: int
    embedding_max_attempts: int
    embedding_batch_size: int


@dataclass(frozen=True)
class SearchConfig:
    top_per_doc_view: int
    top_retrieval: int
    top_display: int
    rerank_top_k: int
    beta: float
    alpha: float
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

    @property
    def data_dir(self) -> Path:
        return self.storage.legacy_data_dir


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_dotenv(
    paths: tuple[Path, ...] = (
        SRC_DIR / ".env",
        PROJECT_ROOT / ".env",
        PROTO_DIR / ".env",
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


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _model_config(name: str, data: dict[str, Any]) -> ModelConfig:
    missing = [key for key in ("model", "url", "api_key_env") if not data.get(key)]
    if missing:
        raise ValueError(f"models.{name} missing required keys: {', '.join(missing)}")
    return ModelConfig(
        name=name,
        model=str(data["model"]),
        url=str(data["url"]),
        api_key_env=str(data["api_key_env"]),
    )


def get_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    load_dotenv()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    storage = raw.get("storage", {})
    old_data = raw.get("data", {})
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
            legacy_data_dir=_resolve_project_path(
                str(storage.get("legacy_data_dir", old_data.get("dir", "data/cpret/P2Dup")))
            ),
        ),
        admin=AdminConfig(
            session_hours=int(admin.get("session_hours", 8)),
            password_hash_env=str(admin.get("password_hash_env", "YUANTIJI_ADMIN_PASSWORD_HASH")),
            signing_secret_env=str(admin.get("signing_secret_env", "YUANTIJI_ADMIN_SIGNING_SECRET")),
        ),
        limits=LimitsConfig(
            upload_max_bytes=int(limits.get("upload_max_bytes", 104_857_600)),
            jsonl_max_line_bytes=int(limits.get("jsonl_max_line_bytes", 1_048_576)),
            field_max_text_chars=int(limits.get("field_max_text_chars", 200_000)),
        ),
        jobs=JobsConfig(
            poll_seconds=int(jobs.get("poll_seconds", 2)),
            rewrite_max_attempts=int(jobs.get("rewrite_max_attempts", 3)),
            embedding_max_attempts=int(jobs.get("embedding_max_attempts", 3)),
            embedding_batch_size=int(jobs.get("embedding_batch_size", 16)),
        ),
        search=SearchConfig(
            top_per_doc_view=int(search.get("top_per_doc_view", 50)),
            top_retrieval=int(search.get("top_retrieval", 200)),
            top_display=int(search.get("top_display", 20)),
            rerank_top_k=int(search.get("rerank_top_k", 50)),
            beta=float(search.get("beta", search.get("default_beta", 0.75))),
            alpha=float(search.get("alpha", search.get("default_alpha", 0.5))),
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
        audit=AuditConfig(retention_days=int(audit.get("retention_days", 90))),
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
            "model": config.model,
            "url": config.url,
            "api_key_env": config.api_key_env,
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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def db_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
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
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported database schema version: {version}")


def ensure_database(settings: Settings) -> None:
    with db_connection(settings) as conn:
        migrate(conn)


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


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}
