# Irminsul 数据资产管理后台与四视图索引发布系统

## 1. 当前目标

Irminsul 是一个单机 FastAPI 服务，提供两类能力：

1. 面向用户的题目相似检索前台。
2. 面向管理员的数据资产、模型产物、索引与任务管理后台。

系统从 JSONL 导入题目，生成四种文本视图，生成 embedding，构建不可变索引，导出 `.npy` cache，并把 active index 加载到内存或 mmap 后服务前台检索。

```text
JSONL import
  -> problems / sources / problem_text artifacts
  -> rewrite artifacts: clean / statement / abstract / abstract_zh
  -> embedding artifacts for each view
  -> immutable index rows
  -> .npy cache
  -> active LoadedIndex
  -> streaming search API
```

核心约束：

- 单机、单进程、SQLite。
- 不引入 SQLAlchemy、Alembic、Celery、Redis。
- 不跨线程共享 SQLite connection。
- 公开搜索路径不从 SQLite 读取向量，只读取 active `LoadedIndex`。
- SQLite 是 canonical store；`.npy` cache 是发布层和启动加速层。

## 2. 代码边界

```text
app.py        FastAPI 路由、认证、CSRF、生命周期、静态文件托管
core.py       配置、SQLite schema/migration、连接、key、CRUD
pipeline.py   import、rewrite/embedding artifact 生成、index build/cache/job worker
search.py     LoadedIndex、检索、rerank、fusion、API client、search audit
frontend/     vanilla TypeScript 前台；vanilla TypeScript + PicoCSS classless 管理后台
tests/        API、检索、迁移、pipeline smoke tests
```

当前不再强制“只保留 4 个 Python 文件”之外的所有职责都挤在同一处；但新增文件仍应有明确边界，避免把 pipeline 再继续扩大为通用工具箱。

## 3. SQLite 策略

每次创建连接执行：

```sql
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
```

`connect_db()` 使用 `timeout=5.0`。HTTP 请求和后台 worker 都使用短连接；后台线程不得共享 connection。

当前有一个进程内 writer-priority read/write lock：

- `db_read_connection(settings)`：读连接。
- `db_write_connection(settings)`：写连接。
- 目标是减少“多读者压住少写者”时的 `database is locked`。
- 这不是跨进程锁；部署仍假设单 uvicorn worker。

迁移使用 `PRAGMA user_version`，不用 Alembic。当前 schema version 为 2。

## 4. 数据模型

当前 SQLite 共有 9 张业务表：

| 表 | 作用 |
|----|------|
| `sources` | 来源定义和启用状态 |
| `problems` | 题目元数据，指向当前 `problem_text` artifact |
| `artifacts` | `problem_text`、`rewrite`、`embedding` 产物 |
| `indexes` | 不可变 index 版本及状态 |
| `index_rows` | index 中每个 problem/view 的行映射 |
| `jobs` | import/build/activate 等后台任务 |
| `job_logs` | job 流式日志和错误详情 |
| `search_audits` | 前台查询审计、耗时、API cost |
| `kv` | `active_index_key` 等小型状态 |

### 4.1 Artifacts

`artifacts` 当前不再记录 `attempts`。失败 artifact 可由管理员在 job 层手动 retry，不在 artifact 上自动累加重试次数。

三类 artifact：

| kind | parent | role | content |
|------|--------|------|---------|
| `problem_text` | null | null | 原始题面文本 |
| `rewrite` | `problem_text` | `rewrite` | JSON data: `clean`、`statement`、`abstract`、`abstract_zh`、usage、method snapshot |
| `embedding` | `rewrite` | view name | float32 blob + dim/dtype/usage/method snapshot |

派生产物由 `(kind, parent_key, method_key, role)` 唯一确定。`method_key` 使用模型 identity 和影响输出的配置；不包含 provider endpoint 或 API key env。这样相同模型经不同 provider 路由时不会误判为不同语义版本。

### 4.2 Problems

`problems.text_key` 指向当前题面 artifact。后台 problem detail 可以预览并修改 problem text；修改时插入新的 `problem_text` artifact 并更新 `text_key`，旧 artifact 保留用于追溯。

### 4.3 Jobs

`jobs.progress` 存储当前任务进度、取消请求、计数等结构化状态。`job_logs` 作为流式 result 展示来源；错误应写入 logs，而不是只在任务结束后塞一个 JSON。

取消语义：

- queued job 可直接取消。
- running job 设置 `progress.cancel_requested = true`。
- worker 在 rewrite/embedding/build 循环边界检查取消请求。
- 取消不应把尚未开始的 artifact 计为失败。

## 5. Key 与版本

所有内容 key 为短前缀 + SHA-256 hex。

| Key | 语义 |
|-----|------|
| `text_key` | canonicalized problem text |
| `method_key` | kind + model identity + prompt/config |
| `rewrite_key` | rewrite + text_key + rewrite_method_key |
| `embedding_key` | embedding + rewrite_key + embedding_method_key + view |
| `row_hash` | index row schema + problem/view/embedding/text metadata |
| `index_key` | schema + rewrite method + embedding method + sorted row hashes |

`index_key` 不包含 search-time 参数，例如 beta、rerank、top_display。

## 6. Import

输入 JSONL 推荐字段：

```json
{"id": "CodeForces/1234B", "title": "...", "text": "...", "url": "..."}
```

`source_key` 从 `id` 前缀派生。当前 import mode：

| Mode | 行为 |
|------|------|
| `upsert` | 新增或更新 title/url/text |
| `insert_only` | 已存在题目跳过 |
| `sync_source` | 同步某来源，缺失题目置为 disabled |

流程：

1. `POST /admin/api/import/dry-run` 上传并校验文件，创建 draft job。
2. `POST /admin/api/import/{key}/confirm` 把 draft 入队执行。
3. `DELETE /admin/api/import/{key}` 删除尚未 confirm 的 draft。

当前 imports 页面第一行显示原始文件名，便于识别。

## 7. Artifact 生成与 Index Build

Build job 使用 snapshot，保证 retry blocked job 时仍基于同一批题目，不被后台之后的 problem 修改影响。

当前流程：

```text
1. 读取 enabled problems，写 build snapshot。
2. 批量生成缺失 rewrite；已有 succeeded rewrite 直接复用。
3. 所有 rewrite 处理完后，收集缺失 embedding。
4. embedding 按 batch 调 API，并发处理多个 batch。
5. 所有四视图 embedding 齐全后写 index_rows。
6. 导出 .npy cache 到临时目录。
7. 校验通过后发布 cache 并把 index 标记 built。
```

当前并发配置：

```toml
[jobs]
poll_seconds = 2
rewrite_concurrency = 32
embedding_concurrency = 4
embedding_batch_size = 128
```

rewrite 使用线程池并发逐题请求。embedding 使用 batch API；每个 batch 最多 `embedding_batch_size` 条文本，同时最多 `embedding_concurrency` 个 batch in flight。

失败策略：

- 单个 artifact 失败不阻塞其他 artifact。
- 失败写入 `job_logs`，后台 Job detail 可查看 problem key、phase、error。
- rewrite 输出有结构质量门：`clean`、`statement`、`abstract`、`abstract_zh` 任一 view 超过 `10_000` 字符则判为 rewrite failed。
- 构建结束时如果仍有失败或缺失 artifact，job 进入 blocked/failed，不创建完整 active index。
- retry failed job 时应跳过已 succeeded artifact，只处理缺口。

## 8. Cache 与启动

每个 index 导出到：

```text
data/index_cache/<index_key>/
  manifest.json
  problems.jsonl
  views.jsonl
  clean.npy
  statement.npy
  abstract.npy
  abstract_zh.npy
```

导出使用临时目录，校验通过后原子替换正式 cache 目录。默认 load mode 为 `mmap`。

启动时读取 `kv.active_index_key` 并尝试加载 cache：

- 加载成功：`IndexState.current = LoadedIndex`。
- 加载失败：记录 `indexes.error = "startup load failed: ..."`，服务 degraded 启动，前台搜索返回无索引错误，后台仍可进入并修复。

手动 activate 仍保持严格语义：目标 cache 加载失败时 API 返回错误，不更新 `active_index_key`。

## 9. Active Index 与搜索

`IndexState` 持有当前 active index：

- `current`
- `switching`
- `inflight_searches`
- `condition`

搜索流程：

```text
POST /api/search
  -> optional query rewrite, returns four views
  -> embed four query views
  -> retrieve up to top_retrieval candidates from active index
  -> rerank window
  -> calibrated fusion
  -> NDJSON stream events
  -> write search_audits
```

当前 rerank 规则：

- `rerank_top_k = 0` 表示对全部 `top_retrieval` candidates rerank。
- 正数表示先截断到 topK 再 rerank。

排序规则：

```text
final_score desc
rerank_score desc
embedding_score desc
problem_key asc
```

## 10. 公开 API

```text
GET  /api/health
GET  /api/config
POST /api/search        NDJSON stream
```

`/api/search` 返回 candidates 时包含四个 view：`clean`、`statement`、`abstract`、`abstract_zh`。前端将 `clean` 显示为 `Filtered`，将 `abstract_zh` 显示为 `中文`。

返回结果包含 cost。前端在结果数量旁展示估算费用：英文 view 显示 `$`，中文 view 显示按 `1 USD = 7 CNY` 估算的 `￥`。

## 11. 管理 API

所有管理端点在 `/admin/api/*` 下，需要 session + CSRF。

| 分组 | 端点 |
|------|------|
| Auth | `POST /auth/login` · `POST /auth/logout` |
| Dashboard | `GET /dashboard` |
| Problems | `GET /problems` · `GET /problems/{key}` · `PATCH /problems/{key}` · `POST /problems/batch-{action}` |
| Sources | `GET /sources` · `PATCH /sources/{key}` |
| Import | `POST /import/dry-run` · `POST /import/{key}/confirm` · `DELETE /import/{key}` · `GET /imports` · `GET /imports/{key}` |
| Jobs | `GET /jobs` · `GET /jobs/{key}` · `POST /jobs/{key}/retry` · `POST /jobs/{key}/cancel` |
| Indexes | `POST /index/build` · `GET /indexes` · `GET /indexes/{key}` · `DELETE /indexes/{key}` · `POST /index/{key}/activate` · `POST /index/{key}/cache/rebuild` · `POST /index/{key}/verify` |
| Audits | `GET /audits` · `GET /audits/{id}` |
| Settings | `GET /settings` |

Dashboard 当前展示：

- enabled problem count。
- 有 succeeded rewrite 的 problem count。
- 有完整 succeeded embedding 的 problem count。
- active index 中 problem count。
- 当前 active/running jobs。

## 12. 认证与 Secrets

后台是单管理员模式：

- 密码 hash 文件：`data/admin_password.hash`。
- session signing secret 文件：`data/admin_signing_secret`。

登录后设置：

- `admin_session`：HttpOnly signed cookie。
- `admin_csrf`：非 HttpOnly，用于 `X-CSRF-Token`。

非 GET 请求校验 session、CSRF cookie、`X-CSRF-Token`、Origin/Referer。

模型 API key 仍通过环境变量读取，后台可以展示配置但不返回 secret 值。

## 13. 当前配置

```toml
[storage]
db_path = "data/app.sqlite3"
upload_dir = "data/uploads"
index_cache_dir = "data/index_cache"

[admin]
session_hours = 8
password_hash_file = "data/admin_password.hash"
signing_secret_file = "data/admin_signing_secret"

[jobs]
poll_seconds = 2
rewrite_concurrency = 32
embedding_concurrency = 4
embedding_batch_size = 128

[api]
request_timeout = 240

[search]
top_per_doc_view = 50
top_retrieval = 200
top_display = 20
rerank_top_k = 0
beta = 0.75
default_rerank = true
rerank_range_floor = 0.1
embedding_range_floor = 0.05

[index_cache]
load_mode = "mmap"
activation_drain_timeout_seconds = 30

[audit]
retention_days = 9999

[models.rewrite]
model = "deepseek-v4-flash"
identity = "deepseek-v4-flash"
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

Pricing 当前按 provider/model 存在 `[audit.pricing.*]` 下，search audit 写入当时的 pricing snapshot。

## 14. 前端状态

前台是 vanilla TypeScript，不使用 React/MUI/AntD。当前主要交互：

- 查询输入自适应高度。
- NDJSON streaming search。
- Rewrite 完成后在阶段旁显示编辑按钮；编辑弹窗支持 `clean`、`statement`、`abstract`、`abstract_zh` 四个 view。
- Sort 和 View 使用下拉选择，选项写入 localStorage。
- 页脚展示 active index 题目数、按来源统计入口，以及输入内容会被保留用于 audition 的提示。
- 结果列表显示标题、来源链接、statement/view 内容、主分数、rr/emb 小条。
- LaTeX 使用 Temml/MathML 渲染，并定制复制行为。

后台同样是 vanilla TypeScript，使用 PicoCSS classless 承担基础控件样式；当前实现 Dashboard、Imports、Problems、Sources、Indexes、Jobs、Audits、Settings。

## 15. 已知待修

这些是当前实现和理想状态之间仍然存在的明确差距：

1. RAM load mode 的 activate 仍可能先加载新索引再释放旧索引，存在双份内存峰值风险。
2. Build、Activate、Rebuild 等高风险后台操作还缺少二次 confirm。
3. Problems 页面已有单题编辑，但批量选择 UI 仍不完整。
4. Activate 失败后的回滚策略还需要更严格的端到端测试。
5. `pipeline.py`、`admin.ts`、`core.py` 仍偏大，需要后续按职责继续收敛。
6. 导入、构建和 cache 路径仍在同一 pipeline 模块内，需要后续继续按职责收敛。

## 16. 验收

后端改动：

```powershell
python -m pytest tests -q -p no:cacheprovider
```

前端改动：

```powershell
cd frontend
npm run build
```

涉及前端布局或交互时，还需要在浏览器中检查：

- 初始页、搜索中、搜索完成。
- Rewrite 编辑浮层。
- Sort/View 切换。
- 窄屏结果布局。
- Admin Dashboard、Problems、Jobs、Indexes。
