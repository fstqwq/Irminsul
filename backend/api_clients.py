from __future__ import annotations

import re
from typing import Any

import numpy as np
import requests

from .config import ModelConfig
from .models import RewriteResult


REWRITE_PROMPT = """You rewrite competitive programming statements.

Output exactly two sections and nothing else:

[statement]
Strip off all the stories, legends, characters, backgrounds etc. from the statement while still enabling everyone to understand the problem. Also remove the name of the character if possible. This is to say, do not remove anything necessary to understand the full problem and one should feel safe to replace the original statement with your version of the statement. If it is not in English make it English. Provide the simplified statement directly without jargon. Use mathjax ($...$) for math.

[abstract]
Additionally, make it as succinct as possible while still being understandable. Try to avoid formulas and symbols. Abstract freely - for example, if the problem is about buying sushi, you can just phrase it as a knapsack problem. Provide the succinct simplified statement directly without jargon."""


class RewriteFormatError(ValueError):
    pass


def _find_section_marker(text: str, name: str) -> re.Match[str] | None:
    pattern = rf"(?im)^[ \t]*(?:\*\*)?\[?{name}\]?(?:\*\*)?[ \t]*:?[ \t]*$"
    return re.search(pattern, text)


def parse_rewrite_output(text: str) -> RewriteResult:
    statement_match = _find_section_marker(text, "statement")
    abstract_match = _find_section_marker(text, "abstract")

    if statement_match is None:
        raise RewriteFormatError("missing_statement_section")
    if abstract_match is None:
        raise RewriteFormatError("missing_abstract_section")
    if abstract_match.start() <= statement_match.start():
        raise RewriteFormatError("abstract_before_statement")

    statement = text[statement_match.end() : abstract_match.start()].strip()
    abstract = text[abstract_match.end() :].strip()

    if statement.startswith("---"):
        statement = statement[3:].strip()
    if statement.endswith("---"):
        statement = statement[:-3].strip()
    if abstract.startswith("---"):
        abstract = abstract[3:].strip()
    if abstract.endswith("---"):
        abstract = abstract[:-3].strip()

    if not statement:
        raise RewriteFormatError("empty_statement_section")
    if not abstract:
        raise RewriteFormatError("empty_abstract_section")
    return RewriteResult(statement=statement, abstract=abstract, raw=text)


def rewrite_query(model_config: ModelConfig, text: str, timeout: int = 240) -> RewriteResult:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_config.model,
            "messages": [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload["choices"][0]["message"]["content"]
    return parse_rewrite_output(raw)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_texts(model_config: ModelConfig, texts: list[str], timeout: int = 240) -> np.ndarray:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_config.model,
            "input": texts,
            "encoding_format": "float",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    data = sorted(payload["data"], key=lambda item: item["index"])
    vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
    return normalize_matrix(vectors)


def rerank_documents(
    model_config: ModelConfig,
    query_statement: str,
    document_statements: list[str],
    timeout: int = 240,
) -> list[float]:
    if not model_config.api_key:
        raise ValueError(f"{model_config.api_key_env} is not configured")
    if not document_statements:
        return []

    response = requests.post(
        model_config.resolved_url,
        headers={
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json",
        },
        json={"queries": [query_statement], "documents": document_statements},
        timeout=timeout,
    )

    response.raise_for_status()
    payload = response.json()
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(document_statements):
        raise ValueError(f"Unexpected reranker response: {payload}")
    return [float(score) for score in scores]

