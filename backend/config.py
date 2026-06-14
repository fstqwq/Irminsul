from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
PROTO_DIR = PROJECT_ROOT / "proto"
DEFAULT_CONFIG_PATH = SRC_DIR / "config.toml"


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
class RetrievalConfig:
    top_retrieval: int
    top_display: int
    default_alpha: float
    default_beta: float
    default_rerank: bool


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    request_timeout: int
    rewrite_model: ModelConfig
    embedding_model: ModelConfig
    rerank_model: ModelConfig
    retrieval: RetrievalConfig


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


def _resolve_path(value: str) -> Path:
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
    models = raw.get("models", {})
    retrieval = raw.get("retrieval", {})

    return Settings(
        data_dir=_resolve_path(str(raw.get("data", {}).get("dir", "data/cpret/P2Dup"))),
        request_timeout=int(raw.get("api", {}).get("request_timeout", 240)),
        rewrite_model=_model_config("rewrite", models.get("rewrite", {})),
        embedding_model=_model_config("embedding", models.get("embedding", {})),
        rerank_model=_model_config("rerank", models.get("rerank", {})),
        retrieval=RetrievalConfig(
            top_retrieval=int(retrieval.get("top_retrieval", 200)),
            top_display=int(retrieval.get("top_display", 20)),
            default_alpha=float(retrieval.get("default_alpha", 0.50)),
            default_beta=float(retrieval.get("default_beta", 0.75)),
            default_rerank=bool(retrieval.get("default_rerank", True)),
        ),
    )
