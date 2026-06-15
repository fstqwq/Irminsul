# AGENTS.md

This file applies to the entire `yuantiji/src` repository. Read before making code changes.

## Single Source of Truth

`PLAN.md` is the sole source of truth for product and architecture. When code conflicts with `PLAN.md`, migrate toward `PLAN.md` while keeping intermediate states runnable. Do not treat earlier conversation plans as authoritative.

## Prohibitions

```text
Do not introduce SQLAlchemy, Alembic, Celery, Redis, React, MUI, or AntD.
Do not share SQLite connections across threads.
Do not make API calls, assemble matrices, or export cache inside database transactions.
Do not store secrets, API keys, cookies, or authorization headers in Git or logs.
Do not commit frontend/dist, frontend/node_modules, data/, *.npy, or *.sqlite3.
Do not perform unrelated refactors while implementing a phase.
Do not leave two competing implementations for the same endpoint.
```

## Hard Constraints

```text
Repository root: C:\code\yuantiji\src
Deployment model: single machine, single process, single Uvicorn worker, single background job worker.
SQLite connections: always PRAGMA foreign_keys=ON; journal_mode=WAL; busy_timeout=5000;
Schema migration: PRAGMA user_version, not Alembic.
Frontend stack: vanilla TypeScript + Vite, no component frameworks.
Admin UI style: dense and operational (tables, filters, status, logs), not marketing-style.
Language: use English for all code, comments, UI text, and API responses. Do not mix Chinese and English.
```

## Workflow

```text
1. Run git status --short --branch before editing code.
2. Implement one phase at a time; keep the app runnable between phases.
3. Small, reviewable commits.
4. When uncertain, choose the simpler behavior and record assumptions in comments or tests.
5. Do not revert user changes unless the user explicitly asks.
```

## Verification

After backend changes:

```bash
python -m pytest tests -q -p no:cacheprovider
```

After frontend changes:

```bash
cd frontend && npm run build
```

When the backend file layout changes, update the test commands in this file in the same commit.
