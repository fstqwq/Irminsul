# 数据资产管理后台与四视图索引发布系统

## 1. 目标与架构

把文件驱动的 FastAPI 搜索服务改造为 SQLite 管理 + `.npy` cache 加载 + 内存检索的闭环系统。

```text
JSONL 导入 → 管理题目/来源 → 四视图 rewrite → 四视图 embedding
  → 构建不可变 index → 导出 .npy cache → 激活/回滚
  → 搜索服务(内存矩阵) → 查询审计
```

**核心约束**：单机、单进程、单 worker、SQLite、不做 ORM/Celery/多 worker。

### 1.1 数据分层

| 层 | 职责 | 存储 |
|----|------|------|
| Canonical Store | 题目、artifact、index 版本定义 | SQLite |
| 发布层 | 矩阵 + 元数据，加速加载 | `.npy` cache 目录 |
| 运行时 | `LoadedIndex` 四矩阵 + 文本 | 进程内存 / mmap |

搜索请求**不访问 SQLite 向量表**，只用内存索引。SQLite 仅在写 `search_audits` 时访问。

### 1.2 后端文件

```text
app.py       路由、认证、CSRF、中间件、生命周期
core.py      SQLite、迁移、key 计算、配置
pipeline.py  导入、artifact 生成、build job、cache 导出
search.py    LoadedIndex、IndexState、检索、rerank、fusion、审计
```

### 1.3 依赖

```text
fastapi  uvicorn  numpy  requests  pydantic
python-multipart  itsdangerous  pytest  httpx
```

不引入 SQLAlchemy、Alembic、Celery、Redis。

---

## 2. 数据模型

### 2.1 SQLite 策略

每次创建连接执行 `PRAGMA foreign_keys=ON; journal_mode=WAL; busy_timeout=5000;`。

不跨线程共享连接：HTTP 请求用短连接，后台 worker 用独立连接。默认 `check_same_thread=True`。

事务内只做数据库写入；API 调用、矩阵组装、cache 导出在事务外。迁移用 `PRAGMA user_version`，不用 Alembic。

### 2.2 8 张表

#### sources / problems

```sql
CREATE TABLE sources (
  key TEXT PRIMARY KEY,
  name TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);

CREATE TABLE problems (
  key TEXT PRIMARY KEY,
  source_key TEXT NOT NULL REFERENCES sources(key),
  title TEXT NOT NULL, url TEXT NOT NULL,
  text_key TEXT NOT NULL REFERENCES artifacts(key),
  enabled INTEGER NOT NULL DEFAULT 1, deleted INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_problems_filter ON problems(source_key, enabled, deleted);
```

题面变更时插入新 `problem_text` artifact 并更新 `text_key`；删除为软删除 `deleted=1`。

#### artifacts

```sql
CREATE TABLE artifacts (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK(kind IN ('problem_text','rewrite','embedding')),
  parent_key TEXT REFERENCES artifacts(key),
  method_key TEXT, role TEXT,
  text TEXT, data TEXT, blob BLOB,
  status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
  error TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE(kind, parent_key, method_key, role)
);
CREATE INDEX idx_artifacts_lookup ON artifacts(kind, parent_key, method_key, role, status);
```

三种 artifact 的 data/blob 布局：

| kind | key 前缀 | parent_key | role | text | data | blob |
|------|----------|------------|------|------|------|------|
| problem_text | `t:` | null | null | 原始题面 | null | null |
| rewrite | `r:` | text_key | `rewrite` | null | `{clean,statement,abstract,abstract_zh,usage,method_snapshot}` | null |
| embedding | `e:` | rewrite_key | view 名 | null | `{dim,dtype,normalized,usage,method_snapshot}` | float32 向量 |

`method_snapshot` 保存生成时的完整 method 配置（model/prompt/dim/normalize 等），用于事后追溯。

核心语义：**派生产物由 `(kind, parent_key, method_key, role)` 唯一确定**。

#### indexes / index_rows

```sql
CREATE TABLE indexes (
  key TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK(status IN ('building','built','active','retired','failed')),
  meta TEXT NOT NULL, created_at TEXT NOT NULL,
  activated_at TEXT, error TEXT
);
-- status=built 仅在 cache 导出并校验通过后设置。cache 导出失败时 status=failed。

CREATE TABLE index_rows (
  index_key TEXT NOT NULL REFERENCES indexes(key),
  problem_ord INTEGER NOT NULL,
  problem_key TEXT NOT NULL, view TEXT NOT NULL,
  embedding_key TEXT NOT NULL REFERENCES artifacts(key),
  title TEXT NOT NULL, url TEXT NOT NULL,
  text_key TEXT NOT NULL, rewrite_key TEXT NOT NULL,
  row_hash TEXT NOT NULL,
  PRIMARY KEY(index_key, problem_ord, view),
  UNIQUE(index_key, problem_key, view)
);
```

每个 problem 固定 4 行（clean/statement/abstract/abstract_zh），`problem_ord` 连续对齐。

#### jobs / search_audits / kv

```sql
CREATE TABLE jobs (
  key TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK(type IN ('import','build_index','activate_index','cleanup')),
  status TEXT NOT NULL CHECK(status IN ('draft','queued','running','succeeded','blocked','failed')),
  payload TEXT NOT NULL, progress TEXT NOT NULL,
  result TEXT, error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX idx_jobs_queue ON jobs(status, created_at);
CREATE INDEX idx_audits_time ON search_audits(started_at);

CREATE TABLE search_audits (
  request_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
  client_ip TEXT, user_agent TEXT, query TEXT NOT NULL,
  timings TEXT NOT NULL, api_calls TEXT NOT NULL,
  result TEXT NOT NULL, cost TEXT NOT NULL, error TEXT
);

CREATE TABLE kv (
  key TEXT PRIMARY KEY,             -- active_index_key | schema_version | last_cleanup_at
  value TEXT NOT NULL, updated_at TEXT NOT NULL
);
```

`blocked` = 部分 artifact 成功但仍有缺失/失败，不能创建完整 index，可 retry。

### 2.3 Key 规则

所有 key 为短前缀 + SHA-256 hex。

| Key | 公式 |
|-----|------|
| text_key | `t:` + sha256(canonical_text) |
| method_key | `m:` + sha256(kind + model + prompt/config) |
| rewrite_key | `r:` + sha256(`rewrite` + text_key + rewrite_method_key + `rewrite`) |
| embedding_key | `e:` + sha256(`embedding` + rewrite_key + embedding_method_key + view) |
| row_hash | sha256(schema_version + problem_ord + problem_key + view + embedding_key + title + url + text_key + rewrite_key) |
| index_key | `i:` + sha256(schema_version + rewrite_method_key + embedding_method_key + sorted(row_hashes)) |

**`index_key` 不包含 search_config**（beta/rerank_top_k/fusion 等是查询时参数）。

---

## 3. 生成与构建流水线

### 3.1 JSONL 导入

输入 schema: `{id, title, text, url}`。`source_key` 从 `id` 前缀派生。

**dry-run**（`POST /admin/api/import/dry-run`）：校验→统计 new/overwrite/skip/errors→创建 `draft` job→返回 stats。不修改 problems/artifacts。

**confirm**（`POST /admin/api/import/{job_key}/confirm`）：校验 job=draft、文件未变→改 status=queued→后台 worker 执行。

三种 mode：

| Mode | 行为 |
|------|------|
| upsert | 新增或更新 title/url/text_key |
| insert_only | 已存在则跳过 |
| sync_source | 限定 source，缺失题 `enabled=0` |

### 3.2 可恢复 Artifact 生成

**原则**：每个成功 artifact 立即原子写回 SQLite 作为 checkpoint。中断后从 pending/failed 缺口继续。已 succeeded 的永远复用。

状态机：`pending → running → succeeded | failed`。服务重启时 `running → pending`。

**rewrite**（`ensure_rewrite`）：计算 rewrite_key → INSERT OR IGNORE pending → 已 succeeded 直接复用 → 否则调 DeepSeek → 解析四段 → 短事务写回。

**embedding 批量**：收集 pending embedding artifacts → 按 `embedding_batch_size=16` 分批 → 标记 running → 事务外调 `embed_texts()` → L2 normalize → 逐条短事务写回 succeeded。

**失败处理**：单个 artifact 失败不影响其他；artifact 不记录重试次数。失败后由管理员在 job 层手动 retry。

### 3.3 索引构建

`POST /admin/api/index/build` → 创建 queued `build_index` job。

```text
1. 读取 enabled problems → 持久化到 data/builds/<job_key>/snapshot.jsonl
   jobs.payload 保存 snapshot_path / rewrite_method_key / embedding_method_key
2. 逐题 ensure_rewrite + ensure_embedding（复用已有）
3. 有 failed artifact → job=blocked，不创建 index
4. 全部 succeeded → 写 index_rows，逐行算 row_hash
5. hash-of-hashes 算 index_key → 写 indexes(status=building)
6. 导出 .npy cache（用 open_memmap 按行写入，避免全量 RAM 峰值）
7. cache 校验通过 → indexes.status=built → job=succeeded
   cache 导出失败 → indexes.status=failed → job=failed
```

retry blocked build **必须使用原 snapshot**，不重新读取当前 problems。新建 build 才读新状态。

### 3.4 .npy Cache

每个 index 导出到 `data/index_cache/<index_key>/`：

```text
manifest.json          schema/dim/文件清单
problems.jsonl         ord/key/title/url/text_key/rewrite_key
views.jsonl            ord/key/clean/statement/abstract/abstract_zh
clean.npy              float32 [N, dim]
statement.npy          float32 [N, dim]
abstract.npy           float32 [N, dim]
abstract_zh.npy        float32 [N, dim]
```

导出使用 `numpy.lib.format.open_memmap(path, mode='w+', dtype=float32, shape=(N, dim))` 按行写入，避免构建时 ~12 GiB RAM 峰值。写完后导出到临时目录 `.building/`，校验通过后 rename 到正式目录。Cache 不作为公开静态文件暴露。

**启动加载**：读 `kv.active_index_key` → cache hit（manifest.index_key 匹配 + shape/dtype 正确）直接 `np.load` → cache miss 从 SQLite 重建 cache → 重建也失败则 degraded。启动时不做 hash 校验；完整性校验通过管理 API `POST /admin/api/index/{key}/verify` 按需执行。

默认 `mmap_mode="r"` 加载，降低内存峰值。也支持 RAM 全量加载。

---

## 4. 索引激活与搜索

### 4.1 IndexState 与切换策略

```python
class IndexState:
    current: LoadedIndex | None
    switching: bool           # True 时新搜索返回 503
    inflight_searches: int    # drain 到 0 后才释放旧索引
    condition: threading.Condition
```

激活流程按 load_mode 区分：

**mmap 模式**（mmap 不立即占物理 RAM，可以先构造新索引再切换）：

```text
1. 预打开新 cache 的 manifest + 四个 npy memmap，校验 shape/dtype
2. 构造新 LoadedIndex 对象（此时新旧共存但 mmap 不占大量物理内存）
3. switching=True，等待 inflight_searches=0（超时 activation_drain_timeout_seconds=30 后失败）
4. current = new，释放旧对象 → switching=False
```

**RAM 模式**（全量加载，避免 12 GiB 双份峰值）：

```text
1. 预校验目标 cache manifest
2. switching=True，等待 inflight_searches=0（同上超时）
3. 释放旧 LoadedIndex → gc.collect()
4. 从 cache 全量加载新 LoadedIndex
5. current = new → switching=False
```

失败回滚：尝试重新加载旧 cache；若也失败则 degraded（503）。
更新 `kv.active_index_key` 在 current 替换成功后写入。

### 4.2 搜索流程

```text
POST /api/search → 检查 switching/索引可用
  → DeepSeek rewrite query 为 4 view
  → Qwen embedding 4 个 query vector（L2 normalized）
  → all_query_best_doc_top50_union 召回
  → 按 embedding_score 截断到 rerank_top_k
  → Qwen reranker
  → calibrated_floor fusion
  → NDJSON stream 返回
  → 写 search_audits
```

### 4.3 Retrieval: all_query_best_doc_top50_union

对每个 doc view（clean/statement/abstract/abstract_zh）：

```text
score[i] = max(cosine(doc_view[i], q) for q in [q_clean, q_statement, q_abstract, q_abstract_zh])
```

实现时 stack 4 个 query vector 为 `Q = [dim, 4]`，每个 doc view 做一次 GEMM `scores = matrix @ Q`，再 `max(axis=1)`。共 4 次 GEMM 而非 16 次 GEMV，对 mmap page fault 和 CPU cache 更友好。

每个 doc view 取 top 50 → 四个 top50 union → 按 problem_key 去重保留最高 embedding_score → 最多 200 candidates。

### 4.4 Rerank & Fusion

Rerank pair：`query.abstract_zh` vs `candidate.clean + "\n\n" + candidate.statement`。默认 `rerank_top_k=50`。

Calibrated floor fusion：

```text
r, e ∈ [0,1]（clipped）
S_r = max(r_max - r_min, 0.1)
S_e = max(e_max - e_min, 0.05)
λ = ((1-β)/β) × (S_r/S_e),  β=0.75
final = (r + λe) / (1 + λ)
```

排序：final_score desc → rerank_score desc → embedding_score desc → problem_key asc。

### 4.5 搜索审计

每次 `/api/search` 写一行 `search_audits`，记录 query 原文、client_ip、user_agent、timings、api_calls（含 pricing snapshot）、result、cost（microusd）。

Cost 估算：pricing 放 config.toml，按 token usage 或 pair count 计算。每条 api_call 保留当时的 pricing snapshot。

---

## 5. 管理后台

### 5.1 API

| 分组 | 端点 |
|------|------|
| Auth | `POST login` · `POST logout` · `GET me` |
| Dashboard | `GET dashboard` |
| Import | `POST import/dry-run` · `POST import/{key}/confirm` · `GET imports` · `GET imports/{key}` |
| Problems | `GET problems` · `PATCH problems/{key}` · `POST problems/batch-{enable,disable,delete,restore}` |
| Sources | `GET sources` · `PATCH sources/{key}` |
| Indexes | `POST index/build` · `GET indexes` · `GET indexes/{key}` · `POST index/{key}/activate` · `POST index/{key}/cache/rebuild` |
| Jobs | `GET jobs` · `GET jobs/{key}` · `POST jobs/{key}/retry` |
| Audits | `GET audits` · `GET audits/{id}` |
| Settings | `GET settings`（只读，不返回 secrets） |

所有端点在 `/admin/api/*` 下。`cache/rebuild` 从 SQLite 重建 `.npy` cache。

### 5.2 认证与 CSRF

单管理员，密码 hash 来自环境变量。使用 `itsdangerous` 签名 cookie。

登录设两个 cookie：`admin_session`（HttpOnly，含 sub/exp/csrf）+ `admin_csrf`（非 HttpOnly，前端可读）。

非 GET 请求：校验 session 签名 + exp → session 内 csrf = cookie csrf = `X-CSRF-Token` header 三者一致 → 校验 Origin/Referer。Session TTL 8 小时。

### 5.3 前端 UI

继续 vanilla TypeScript。9 个页面：

| 页面 | 核心功能 |
|------|----------|
| Login | 密码登录 |
| Dashboard | 题目数/来源数/active index/当前 job/今日查询统计 |
| Imports | 上传 JSONL → 选 mode → dry-run → 查看 stats/errors → confirm |
| Problems | 分页表格 + 按 source/enabled/deleted/关键词过滤 + 编辑/启用/禁用/软删除/批量操作 |
| Sources | 来源统计 + 启用/禁用 + 跳转该 source 的 problems |
| Indexes | 构建/查看/激活/回滚/手动 rebuild cache |
| Jobs | 分页 + 查看 progress/result/error + blocked build retry |
| Audits | 按时间/状态/query 过滤 + 详情展示 timings/cost/result JSON |
| Settings | 只读展示模型配置/method keys/搜索参数/存储/active index |

### 5.4 上传安全

限制文件大小/单行大小/字段长度/总行数。随机文件名，上传目录不在静态目录下。展示题面/query/错误摘要时 HTML escape。日志不输出 API key/Cookie/Authorization。

---

## 6. 运维

### 6.1 后台 Job Worker

单 worker 串行执行 import → build_index → activate_index → cleanup。不做 heartbeat/lease/多 worker。

服务启动恢复：`running jobs → queued`，`running artifacts → pending`，清理 `.building` 残留，加载 active index cache，启动 worker。

### 6.2 Cleanup Job

```text
1. 删除超过 retention_days 的 search_audits（默认 90 天）
2. 删除过期上传临时文件
3. 清理 .building 残留目录
4. 旧 index cache：保留 active + 最近 3 个 built/retired
```

不删除 succeeded artifacts（canonical store）。

### 6.3 失败恢复

| 场景 | 恢复 |
|------|------|
| Build 中断 | succeeded artifacts 保留，running → pending，retry 继续 |
| Build blocked | 不创建 index，retry 只处理缺失项 |
| Activate 失败 | 回滚旧 cache 或 degraded(503) |
| Import 失败 | 已提交数据保留，用户重新导入 |
| 搜索失败 | 返回 error event，audit status=failed |

### 6.4 备份

必须备份：`data/app.sqlite3{,-wal,-shm}` + `data/uploads/` + `data/index_cache/` + `config.toml`。恢复后启动服务即可（cache hit 直接加载）。

---

## 7. 公开 API

```text
GET  /api/health    → {ok, loaded_index_key, problem_count, embedding_shape, views, switching}
GET  /api/config    → 检索参数
POST /api/search    → NDJSON stream: request → rewrite → embedding → candidates → rerank → results → done | error
```

`switching=true` 或无索引时返回 503。浏览器前端只通过 API 获取数据，不读 SQLite 或 `.npy`。

---

## 8. 配置

```toml
[storage]
db_path = "data/app.sqlite3"
upload_dir = "data/uploads"
index_cache_dir = "data/index_cache"

[admin]
session_hours = 8

[limits]
upload_max_bytes = 104857600
jsonl_max_line_bytes = 1048576
field_max_text_chars = 200000

[jobs]
poll_seconds = 2
embedding_batch_size = 16

[search]
top_per_doc_view = 50
rerank_top_k = 50
beta = 0.75
rerank_range_floor = 0.1
embedding_range_floor = 0.05

[index_cache]
keep_retired = 3
load_mode = "mmap"                # mmap | ram
activation_drain_timeout_seconds = 30

[audit]
retention_days = 90

[audit.pricing.deepseek.deepseek-v4-flash]
input_price_per_1m_tokens_microusd = 100000
output_price_per_1m_tokens_microusd = 300000

[models.rewrite]
model = "deepseek-v4-flash"
url = "https://api.deepseek.com/chat/completions"
api_key_env = "DEEPSEEK_API_KEY"

[models.embedding]
model = "Qwen/Qwen3-Embedding-8B"
url = "https://openrouter.ai/api/v1/embeddings"
api_key_env = "OPENROUTER_API_KEY"

[models.rerank]
model = "Qwen/Qwen3-Reranker-8B"
url = "https://api.deepinfra.com/v1/inference/{model}"
api_key_env = "DEEPINFRA_API_KEY"
```

---

## 9. 实施阶段

| Phase | 内容 | 关键产出 |
|-------|------|----------|
| 1 | 依赖更新 + 文件重组为 4 文件 | 现有 `/api/search` 仍可运行 |
| 2 | SQLite connect/migrate + 8 张表 + kv | `core.py` 数据层 |
| 3 | 认证 + signed cookie + double-submit CSRF | `app.py` 中间件 |
| 4 | JSONL upload/dry-run/confirm + import job | `pipeline.py` 导入 |
| 5 | 四视图 rewrite parser + embedding batch | `pipeline.py` artifact 生成 |
| 6 | 可恢复 build + blocked/retry + hash-of-hashes index_key | `pipeline.py` 构建 |
| 7 | .npy cache export + manifest + startup load + mmap | `pipeline.py` 发布 |
| 8 | IndexState switching + drain + 释放旧 + 加载新 + rollback | `search.py` 激活 |
| 9 | 四视图 query rewrite/embedding + retrieval + rerank + fusion | `search.py` 搜索 |
| 10 | search_audits + pricing config + cost 估算 | `search.py` 审计 |
| 11 | 管理后台 9 个页面 | 前端 vanilla TS |
| 12 | cleanup job + cache 清理 + audit retention + 部署文档 + 完整测试 | 运维闭环 |

---

## 10. 验收标准

```text
可以导入 JSONL 并 dry-run/confirm。
可以查看/启用/禁用/软删除题目和来源。
可以构建四视图索引，rewrite/embedding 中断后可继续。
单个 rewrite 失败不丢弃已完成结果。
不完整索引不被创建/激活。
index 构建后导出 .npy cache，启动从 cache 加载。
索引切换锁定搜索，避免双份内存。
可以激活新索引、回滚旧索引。
搜索使用四视图 top50 union + rerank + calibrated_floor。
每次查询记录 query/IP/UA/API 开销。
index_key 不受 search_config 变化影响。
管理后台可完成全部管理操作闭环。
```
