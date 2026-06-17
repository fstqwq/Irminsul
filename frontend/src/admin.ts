import { escapeHtml } from "./math";

type AdminPage = "dashboard" | "imports" | "problems" | "sources" | "indexes" | "jobs" | "audits" | "settings";
type Row = Record<string, unknown>;
type DetailKey = "import" | "problem" | "index" | "job" | "audit";
type DisplayKind = "text" | "key" | "date" | "status" | "jobType" | "cost";
type Field = [string, unknown, DisplayKind?];
type Column = [string, (row: Row) => string];

type AdminState = {
  page: AdminPage;
  authenticated: boolean;
  loading: boolean;
  error: string;
  notice: string;
  data: Record<string, unknown>;
  draftImport: Row | null;
  details: Record<DetailKey, Row | null>;
  filters: {
    problems: { q: string; source_key: string; enabled: string; deleted: string; limit: number; offset: number };
    jobs: { type: string; status: string; limit: number };
    audits: { status: string; q: string; date_from: string; date_to: string; limit: number };
  };
};

const pages = ["dashboard", "imports", "problems", "sources", "indexes", "jobs", "audits", "settings"].map((id) => ({
  id: id as AdminPage,
  label: titleCase(id)
}));
const importModeLabels: Record<string, string> = { upsert: "Update or add", insert_only: "Add new only (skip existing)", sync_source: "Sync source (disable missing)" };
const jobTypeLabels: Record<string, string> = { import: "Import", build_index: "Build Index", activate_index: "Activate Index" };
const statusLabels: Record<string, string> = {
  draft: "Draft", queued: "Queued", running: "Running", succeeded: "Succeeded", blocked: "Blocked", failed: "Failed", building: "Building", built: "Ready", active: "Active", retired: "Retired"
};
const settingsKeyLabels: Record<string, string> = {
  db_path: "Database path", upload_dir: "Upload directory", index_cache_dir: "Cache directory", top_per_doc_view: "Candidates per view", top_retrieval: "Retrieved candidates", top_display: "Displayed results",
  rerank_top_k: "Rerank candidates (0 = all)", beta: "Beta (rerank weight)", default_rerank: "Rerank by default", rerank_range_floor: "Rerank range floor", embedding_range_floor: "Embedding range floor",
  load_mode: "Load mode", activation_drain_timeout_seconds: "Activation drain timeout (s)", name: "Provider", model: "Model", identity: "Model identity", url: "Endpoint",
  api_key_env: "API key environment variable", provider: "Provider routing"
};

const state: AdminState = {
  page: "dashboard",
  authenticated: false,
  loading: true,
  error: "",
  notice: "",
  data: {},
  draftImport: null,
  details: { import: null, problem: null, index: null, job: null, audit: null },
  filters: {
    problems: { q: "", source_key: "", enabled: "", deleted: "false", limit: 25, offset: 0 },
    jobs: { type: "", status: "", limit: 50 },
    audits: { status: "", q: "", date_from: "", date_to: "", limit: 50 }
  }
};

let rootEl: HTMLElement;
let refreshTimer: number | null = null;

export function startAdmin(root: HTMLElement): void {
  rootEl = root;
  window.addEventListener("hashchange", () => {
    state.page = pageFromHash();
    state.error = "";
    state.notice = "";
    void loadPage();
  });
  void checkSession();
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.method && init.method !== "GET") headers.set("X-CSRF-Token", decodeURIComponent(document.cookie.match(/(?:^|; )admin_csrf=([^;]+)/)?.[1] || ""));
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...init, headers, credentials: "same-origin" });
  if (!response.ok) {
    const text = await response.text();
    let message = text || `Request failed (${response.status})`;
    try { message = String((JSON.parse(text) as Row).detail || message); } catch { /* keep plain text */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

async function checkSession(): Promise<void> {
  state.loading = true;
  render();
  try {
    await api("/admin/api/auth/me");
    state.authenticated = true;
    await loadPage();
  } catch {
    state.authenticated = false;
    state.loading = false;
    render();
  }
}

async function loadPage(options: { silent?: boolean } = {}): Promise<void> {
  if (!state.authenticated) return render();
  if (!options.silent) { state.loading = true; render(); }
  state.error = "";
  try {
    const loaders: Record<AdminPage, () => Promise<unknown>> = {
      dashboard: () => api("/admin/api/dashboard"),
      imports: () => api("/admin/api/imports"),
      problems: () => api(`/admin/api/problems?${query(state.filters.problems)}`),
      sources: () => api("/admin/api/sources"),
      indexes: () => api("/admin/api/indexes"),
      jobs: async () => {
        const jobs = await api<Row>(`/admin/api/jobs?${query(state.filters.jobs)}`);
        const detailKey = String(asRecord(state.details.job).key || "");
        if (detailKey) state.details.job = await api<Row>(`/admin/api/jobs/${encodeURIComponent(detailKey)}`);
        return jobs;
      },
      audits: () => api(`/admin/api/audits?${query(state.filters.audits)}`),
      settings: () => api("/admin/api/settings")
    };
    state.data[state.page] = await loaders[state.page]();
  } catch (error) {
    state.error = (error as Error).message;
  } finally {
    if (!options.silent) state.loading = false;
    render();
  }
}

function render(): void {
  if (!state.authenticated) {
    stopAutoRefresh();
    rootEl.onclick = null;
    rootEl.onsubmit = null;
    rootEl.innerHTML = `<main class="admin-login"><form id="adminLoginForm"><h1>Irminsul Admin</h1><label for="adminPassword">Password</label><input id="adminPassword" type="password" autocomplete="current-password" autofocus><button type="submit">Login</button>${state.error ? `<p>${h(state.error)}</p>` : ""}</form></main>`;
    bindLogin();
    return;
  }
  rootEl.innerHTML = `<main class="admin-app"><nav><ul><li><strong>Irminsul Admin</strong></li></ul><ul>${pages.map((page) => `<li><a ${state.page === page.id ? `aria-current="page"` : ""} href="#${page.id}">${page.label}</a></li>`).join("")}<li><button class="secondary" id="adminLogout" type="button">Logout</button></li></ul></nav><header><h1>${h(pages.find((page) => page.id === state.page)?.label || "Dashboard")}</h1><button class="secondary" id="adminRefresh" type="button">${state.loading ? "Loading" : "Refresh"}</button></header>${state.error ? article("Error", `<p>${h(state.error)}</p>`) : ""}${state.notice ? article("Notice", `<p>${h(state.notice)}</p>`) : ""}${renderPage()}</main>`;
  bindAdmin();
  scheduleAutoRefresh();
}

function renderPage(): string {
  if (state.loading) return `<article>Loading data.</article>`;
  const pagesById = { dashboard: renderDashboard, imports: renderImports, problems: renderProblems, sources: renderSources, indexes: renderIndexes, jobs: renderJobs, audits: renderAudits, settings: renderSettings } satisfies Record<AdminPage, (data: Row) => string>;
  return pagesById[state.page](asRecord(state.data[state.page]));
}

function renderDashboard(data: Row): string {
  const currentJob = asRecord(data.current_job);
  const activeIndexCount = data.active_index_problem_count === null || data.active_index_problem_count === undefined ? "None" : data.active_index_problem_count;
  const metrics: Field[] = [["Enabled problems", data.problem_count], ["With rewrite", data.rewrite_problem_count], ["With embedding", data.embedding_problem_count], ["Active index problems", activeIndexCount], ["Sources", data.source_count], ["Today searches", data.today_searches]];
  return `<section class="grid">${metrics.map(([name, raw]) => `<article><small>${h(name)}</small><h3>${display(raw)}</h3></article>`).join("")}</section>${article("Current job", Object.keys(currentJob).length ? detailFields([["Job ID", currentJob.key, "key"], ["Type", currentJob.type, "jobType"], ["Status", currentJob.status, "status"], ["Updated", currentJob.updated_at, "date"]]) + block("Progress", h(formatProgress(currentJob.type, currentJob.progress))) : `<p>No queued or running job.</p>`)}`;
}

function renderImports(data: Row): string {
  return `${article("Upload JSONL", `<form id="importForm"><input name="file" type="file" accept=".jsonl,application/jsonl">${select("mode", "", Object.entries(importModeLabels))}<button type="submit">Dry run</button></form>${state.draftImport ? draftImport(state.draftImport) : ""}`)}${state.details.import ? importDetail(state.details.import) : ""}${table([
    ["File", (row) => h(importFilename(row))], ["Status", (row) => statusBadge(row.status)], ["Progress", (row) => h(formatProgress(row.type || "import", row.progress))], ["Updated", (row) => shortDate(row.updated_at)], ["", importActions]
  ], asRows(data.items))}`;
}

function draftImport(draft: Row): string {
  const stats = asRecord(draft.stats);
  const errors = Array.isArray(stats.errors) ? stats.errors : [];
  const key = String(draft.job_key || draft.key || "");
  return `<article><p>${h(importFilename(draft))}</p><p>New: ${v(stats.new)} Updated: ${v(stats.overwrite)} Skipped: ${v(stats.skip)} Errors: ${errors.length}</p><button data-confirm-import="${h(key)}" type="button">Confirm</button> <button data-delete-import="${h(key)}" type="button">Delete</button>${errors.length ? jsonBlock("Errors", errors) : ""}</article>`;
}

function importActions(row: Row): string {
  const key = String(row.key || "");
  const draft = String(row.status || "") === "draft";
  return `<button data-import-detail="${h(key)}" type="button">View</button>${draft ? ` <button data-confirm-import="${h(key)}" type="button">Confirm</button> <button data-delete-import="${h(key)}" type="button">Delete</button>` : ""}`;
}

function importDetail(job: Row): string {
  return detailPanel("Import detail", "import", detailFields([["File", importFilename(job)], ["Import ID", job.key, "key"], ["Status", job.status, "status"], ["Created", job.created_at, "date"], ["Updated", job.updated_at, "date"], ["Progress", formatProgress(job.type || "import", job.progress)]]) + jsonBlock("Result", job.result) + jsonBlock("Error", job.error));
}

function renderProblems(data: Row): string {
  const f = state.filters.problems;
  const total = Number(data.total || 0);
  const from = total === 0 ? 0 : f.offset + 1;
  const to = Math.min(f.offset + f.limit, total);
  return `${article("Filters", `<form id="problemFilterForm"><input name="q" value="${h(f.q)}" placeholder="Keyword"><input name="source_key" value="${h(f.source_key)}" placeholder="Source">${select("enabled", f.enabled, [["", "Enabled: any"], ["true", "Enabled only"], ["false", "Disabled only"]])}${select("deleted", f.deleted, [["", "Deleted: any"], ["false", "Not deleted"], ["true", "Deleted only"]])}<button type="submit">Apply</button></form>`)}${state.details.problem ? problemEditor(state.details.problem) : ""}<nav><ul><li>${from}-${to} / ${total}</li></ul><ul><li><button data-page-problems="-1" type="button" ${f.offset <= 0 ? "disabled" : ""}>Prev</button></li><li><button data-page-problems="1" type="button" ${f.offset + f.limit >= total ? "disabled" : ""}>Next</button></li></ul></nav>${table([
    ["ID", (row) => h(row.key)], ["Title", (row) => h(row.title)], ["Source", (row) => h(row.source_key)], ["Enabled", (row) => flag(row.enabled)], ["Deleted", (row) => flag(row.deleted)], ["Updated", (row) => shortDate(row.updated_at)], ["", problemActions]
  ], asRows(data.items))}`;
}

function problemActions(row: Row): string {
  const key = String(row.key || "");
  const enabled = truthy(row.enabled);
  const deleted = truthy(row.deleted);
  return `<button data-edit-problem="${h(key)}" type="button">Detail</button> <button data-problem-patch="${h(key)}" data-enabled="${enabled ? "false" : "true"}" type="button">${enabled ? "Disable" : "Enable"}</button> <button data-problem-patch="${h(key)}" data-deleted="${deleted ? "false" : "true"}" type="button">${deleted ? "Restore" : "Delete"}</button>`;
}

function problemEditor(problem: Row): string {
  const form = `<form id="problemEditForm"><label>ID<input name="key" value="${h(problem.key)}" readonly></label><label>Title<input name="title" value="${h(problem.title)}"></label><label>URL<input name="url" value="${h(problem.url)}"></label><label>Text key<input value="${h(problem.text_key)}" readonly></label><label><input name="enabled" type="checkbox" ${truthy(problem.enabled) ? "checked" : ""}> Enabled</label><label><input name="deleted" type="checkbox" ${truthy(problem.deleted) ? "checked" : ""}> Deleted</label><label>Problem text<textarea name="text" rows="18" spellcheck="false">${h(problem.text)}</textarea></label><button type="submit">Save problem</button> <button data-close-detail="problem" type="button">Cancel</button></form>`;
  return detailPanel("Problem detail", "problem", detailFields([["ID", problem.key, "key"], ["Source", problem.source_key], ["Enabled", truthy(problem.enabled) ? "Yes" : "No"], ["Deleted", truthy(problem.deleted) ? "Yes" : "No"], ["Updated", problem.updated_at, "date"], ["URL", problem.url]]) + artifactStatusBlock(problem) + form);
}

function renderSources(data: Row): string {
  return table([
    ["ID", (row) => h(row.key)], ["Name", (row) => h(row.name)], ["Enabled", (row) => flag(row.enabled)], ["Problems", (row) => v(row.problem_count)], ["Enabled problems", (row) => v(row.enabled_problem_count)], ["Deleted", (row) => v(row.deleted_count)],
    ["", (row) => { const key = String(row.key || ""); const enabled = truthy(row.enabled); return `<button data-source-toggle="${h(key)}" data-enabled="${enabled}" type="button">${enabled ? "Disable" : "Enable"}</button> <button data-source-problems="${h(key)}" type="button">Problems</button>`; }]
  ], asRows(data.items));
}

function renderIndexes(data: Row): string {
  return `${article("", `<button id="buildIndex" type="button">Build new index</button>`)}${state.details.index ? indexDetail(state.details.index) : ""}${table([
    ["ID", (row) => shortKey(row.key)], ["Status", (row) => statusBadge(row.status)], ["Created", (row) => shortDate(row.created_at)], ["Activated", (row) => shortDate(row.activated_at)], ["", indexActions]
  ], asRows(data.items))}`;
}

function indexActions(row: Row): string {
  const key = String(row.key || "");
  const status = String(row.status || "");
  const deleteButton = ["retired", "failed"].includes(status) ? ` <button data-delete-index="${h(key)}" type="button">Delete</button>` : "";
  return `<button data-index-detail="${h(key)}" type="button">View</button> <button data-activate-index="${h(key)}" type="button">Activate</button> <button data-verify-index="${h(key)}" type="button">Check integrity</button> <button data-rebuild-index="${h(key)}" type="button">Rebuild cache</button>${deleteButton}`;
}

function indexDetail(index: Row): string {
  return detailPanel("Index detail", "index", detailFields([["ID", index.key, "key"], ["Status", index.status, "status"], ["Created", index.created_at, "date"], ["Activated", index.activated_at, "date"]]) + renderIndexMeta(index.meta) + jsonBlock("Error", index.error));
}

function renderJobs(data: Row): string {
  const f = state.filters.jobs;
  const filters = `<form id="jobFilterForm">${select("type", f.type, [["", "Type: any"], ["import", jobTypeLabels.import], ["build_index", jobTypeLabels.build_index], ["activate_index", jobTypeLabels.activate_index]])}${select("status", f.status, [["", "Status: any"], ["draft", statusLabels.draft], ["queued", statusLabels.queued], ["running", statusLabels.running], ["succeeded", statusLabels.succeeded], ["blocked", statusLabels.blocked], ["failed", statusLabels.failed]])}<button type="submit">Apply</button></form>`;
  return `${article("Filters", filters)}${state.details.job ? jobDetail(state.details.job) : ""}${table([
    ["Job ID", (row) => shortKey(row.key)], ["Type", (row) => h(label(jobTypeLabels, row.type))], ["Status", (row) => statusBadge(row.status)], ["Progress", (row) => h(formatProgress(row.type, row.progress))], ["Updated", (row) => shortDate(row.updated_at)], ["", jobActions]
  ], asRows(data.items))}`;
}

function jobActions(row: Row): string {
  const key = String(row.key || "");
  const status = String(row.status || "");
  return `<button data-job-detail="${h(key)}" type="button">View</button>${isActiveJobStatus(status) ? ` <button data-cancel-job="${h(key)}" type="button">Cancel</button>` : ""}${["blocked", "failed"].includes(status) ? ` <button data-retry-job="${h(key)}" type="button">Retry</button>` : ""}`;
}

function jobDetail(job: Row): string {
  const status = String(job.status || "");
  return detailPanel("Job detail", "job", detailFields([["Job ID", job.key, "key"], ["Type", job.type, "jobType"], ["Status", job.status, "status"], ["Created", job.created_at, "date"], ["Updated", job.updated_at, "date"], ["Progress", formatProgress(job.type, job.progress)]]) + jsonBlock("Payload", job.payload) + failureBlock(job) + logBlock("Result", asRows(job.logs)) + jsonBlock("Error", job.error) + (isActiveJobStatus(status) ? `<button data-cancel-job="${h(job.key)}" type="button">Cancel</button>` : "") + (["blocked", "failed"].includes(status) ? ` <button data-retry-job="${h(job.key)}" type="button">Retry</button>` : ""));
}

function renderAudits(data: Row): string {
  const f = state.filters.audits;
  return `${article("Filters", `<form id="auditFilterForm"><input name="q" value="${h(f.q)}" placeholder="Query contains"><input name="date_from" value="${h(f.date_from)}" type="date"><input name="date_to" value="${h(f.date_to)}" type="date">${select("status", f.status, [["", "Status: any"], ["succeeded", statusLabels.succeeded], ["failed", statusLabels.failed]])}<button type="submit">Apply</button></form>`)}${state.details.audit ? auditDetail(state.details.audit) : ""}${table([
    ["Started", (row) => shortDate(row.started_at)], ["Status", (row) => statusBadge(row.status)], ["Query", (row) => `<span class="audit-query">${truncateText(row.query, 260)}</span>`], ["Cost", (row) => formatCost(row.cost)], ["", (row) => `<button data-audit-detail="${h(row.request_id)}" type="button">View</button>`]
  ], asRows(data.items))}`;
}

function auditDetail(audit: Row): string {
  return detailPanel("Audit detail", "audit", detailFields([["Request", audit.request_id, "key"], ["Status", audit.status, "status"], ["Started", audit.started_at, "date"], ["Finished", audit.finished_at, "date"], ["Client IP", audit.client_ip], ["User agent", audit.user_agent], ["Cost", audit.cost, "cost"]]) + block("Query", h(audit.query)) + jsonBlock("Timings", audit.timings) + jsonBlock("API calls", audit.api_calls) + jsonBlock("Result", audit.result) + jsonBlock("Error", audit.error));
}

function renderSettings(data: Row): string {
  const models = asRecord(data.models);
  return `<section class="grid">${settingsSection("Storage", data.storage)}${settingsSection("Search", data.search)}${settingsSection("Index cache", data.index_cache)}</section>${article("Models", `<section class="grid">${settingsSection("Rewrite", models.rewrite)}${settingsSection("Embedding", models.embedding)}${settingsSection("Rerank", models.rerank)}</section>`)}`;
}

function settingsSection(title: string, raw: unknown): string {
  const rows = Object.entries(asRecord(raw)).map(([key, value]) => `<tr><th>${h(label(settingsKeyLabels, key))}</th><td>${h(settingsValue(value))}</td></tr>`).join("");
  return article(title, `<table><tbody>${rows || emptyRow(2)}</tbody></table>`);
}

function article(title: string, body: string): string {
  return `<article>${title ? `<h2>${h(title)}</h2>` : ""}${body}</article>`;
}
function table(columns: Column[], rows: Row[]): string {
  return `<table><thead><tr>${columns.map(([title]) => `<th>${h(title)}</th>`).join("")}</tr></thead><tbody>${rows.length ? rows.map((row) => `<tr>${columns.map(([, cell]) => `<td>${cell(row)}</td>`).join("")}</tr>`).join("") : emptyRow(columns.length)}</tbody></table>`;
}
function emptyRow(cols: number): string { return `<tr><td colspan="${cols}">No data.</td></tr>`; }
function detailPanel(title: string, key: DetailKey, body: string): string { return `<article><header><h2>${h(title)}</h2><button data-close-detail="${key}" type="button" class="secondary">Close</button></header>${body}</article>`; }
function detailFields(fields: Field[]): string { return `<dl>${fields.map(([name, raw, kind]) => `<dt>${h(name)}</dt><dd>${display(raw, kind || "text")}</dd>`).join("")}</dl>`; }
function block(title: string, body: string): string { return body ? `<section><h3>${h(title)}</h3>${body}</section>` : ""; }
function jsonBlock(title: string, raw: unknown): string { return raw === null || raw === undefined || raw === "" ? "" : `<section><h3>${h(title)}</h3><pre>${h(formatJson(raw))}</pre></section>`; }
function failureBlock(job: Row): string {
  const failures = jobFailures(job);
  return failures.length ? block(`Failures (${failures.length})`, table([["Problem", (row) => `<code>${h(row.problemKey)}</code>`], ["Phase", (row) => h(row.phase)], ["Error", (row) => `<mark>${h(row.error)}</mark>`]], failures as unknown as Row[])) : "";
}
function artifactStatusBlock(problem: Row): string {
  const artifacts = asRecord(problem.artifacts);
  const problemText = asRecord(artifacts.problem_text);
  const rows: Row[] = [{ name: "Problem text", status: problemText.status, updated_at: problemText.updated_at, error: problemText.error }];
  for (const rewrite of asRows(artifacts.rewrites)) {
    rows.push({ name: `Rewrite ${shortKey(rewrite.key)}`, status: rewrite.status, updated_at: rewrite.updated_at, error: rewrite.error });
    for (const emb of asRows(rewrite.embeddings)) {
      rows.push({ name: `\u00a0\u00a0\u21b3 Embedding (${h(emb.role)})`, status: emb.status, updated_at: emb.updated_at, error: emb.error });
    }
  }
  const tbl = table([["Artifact", (row) => String(row.name || "")], ["Status", (row) => statusBadge(row.status)], ["Updated", (row) => shortDate(row.updated_at)], ["Error", (row) => row.error ? `<mark>${h(row.error)}</mark>` : ""]], rows);
  return block("Artifacts", tbl + rewriteContentBlock(asRows(artifacts.rewrites)));
}
function rewriteContentBlock(rewrites: Row[]): string {
  const succeeded = rewrites.filter((r) => r.status === "succeeded" && r.data);
  if (!succeeded.length) return "";
  return succeeded.map((rewrite) => {
    const data = asRecord(typeof rewrite.data === "string" ? JSON.parse(String(rewrite.data)) : rewrite.data);
    const views = ["clean", "statement", "abstract", "abstract_zh"].filter((v) => data[v]);
    if (!views.length) return "";
    return `<details><summary>Rewrite content (${shortKey(rewrite.key)})</summary>${views.map((v) => `<h4>${h(v)}</h4><pre>${h(data[v])}</pre>`).join("")}</details>`;
  }).join("");
}
function logBlock(title: string, logs: Row[]): string { return logs.length ? `<section><h3>${h(title)}</h3>${logs.map(logLine).join("")}</section>` : ""; }
function logLine(log: Row): string { return `<p><small>${shortDate(log.created_at, true)} ${h(log.level || "info")}</small><br>${h(log.message)} ${formatLogData(log.data)}</p>`; }
function formatLogData(raw: unknown): string {
  if (raw === null || raw === undefined || raw === "") return "";
  const data = asRecord(raw);
  const parts = [data.problem_key ? `<code>${h(data.problem_key)}</code>` : "", data.error ? `<mark>${h(data.error)}</mark>` : ""].filter(Boolean);
  return parts.length ? parts.join(" ") : `<code>${h(formatJson(raw))}</code>`;
}
function renderIndexMeta(raw: unknown): string {
  const meta = asRecord(parseMaybeJson(raw));
  if (!Object.keys(meta).length) return "";
  return `<section><h3>Index metadata</h3>${detailFields([["Problem count", meta.problem_count], ["Schema version", meta.schema_version], ["Rewrite method key", meta.rewrite_method_key, "key"], ["Embedding method key", meta.embedding_method_key, "key"], ["Cache path", meta.cache_path]])}${jsonBlock("Rewrite method", meta.rewrite_method)}${jsonBlock("Embedding method", meta.embedding_method)}</section>`;
}
function jobFailures(job: Row): { problemKey: string; phase: string; error: string }[] {
  const resultFailures = asRows(asRecord(parseMaybeJson(job.result)).failures);
  if (resultFailures.length) return resultFailures.map((failure) => ({ problemKey: String(failure.problem_key || ""), phase: String(failure.phase || ""), error: String(failure.error || "") })).filter((failure) => failure.problemKey || failure.error);
  return asRows(job.logs).flatMap((log) => {
    const data = asRecord(log.data);
    const keys = Array.isArray(data.problem_keys) ? data.problem_keys.map(String) : data.problem_key ? [String(data.problem_key)] : [];
    const phase = String(data.phase || phaseFromMessage(log.message));
    const error = String(data.error || log.message || "");
    return keys.map((problemKey) => ({ problemKey, phase, error }));
  });
}

function handleAdminClick(event: MouseEvent): void {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button");
  if (!button) return;
  const data = button.dataset;
  if (button.id === "adminLogout") return void api("/admin/api/auth/logout", { method: "POST" }).finally(() => { state.authenticated = false; render(); });
  if (button.id === "adminRefresh") return void loadPage();
  if (button.id === "buildIndex") return void runAction(() => api("/admin/api/index/build", { method: "POST" }), "Index build queued.");
  if (data.confirmImport && window.confirm("Start this import job now?")) return void runAction(() => api(`/admin/api/import/${encodeURIComponent(data.confirmImport || "")}/confirm`, { method: "POST" }), "Import queued.", () => { state.draftImport = null; });
  if (data.deleteImport && window.confirm("Delete this draft import?")) return void runAction(() => api(`/admin/api/import/${encodeURIComponent(data.deleteImport || "")}`, { method: "DELETE" }), "Draft import deleted.", () => { if (String(state.draftImport?.job_key || state.draftImport?.key || "") === data.deleteImport) state.draftImport = null; if (String(state.details.import?.key || "") === data.deleteImport) state.details.import = null; });
  if (data.importDetail) return void loadDetail("import", `/admin/api/imports/${encodeURIComponent(data.importDetail)}`);
  if (data.pageProblems) { const f = state.filters.problems; f.offset = Math.max(0, f.offset + Number(data.pageProblems) * f.limit); return void loadPage(); }
  if (data.editProblem) return void loadDetail("problem", `/admin/api/problems/${pathKey(data.editProblem)}`);
  if (data.problemPatch) {
    const body = patchProblemBody(data);
    if (!body) return;
    return void patchProblem(data.problemPatch, body);
  }
  if (data.sourceToggle && (!truthy(data.enabled) || window.confirm("Disable this source?"))) return void runAction(() => api(`/admin/api/sources/${encodeURIComponent(data.sourceToggle || "")}`, { method: "PATCH", body: JSON.stringify({ enabled: !truthy(data.enabled) }) }), "Source updated.");
  if (data.sourceProblems) { state.filters.problems.source_key = data.sourceProblems; state.filters.problems.offset = 0; window.location.hash = "problems"; return; }
  if (data.indexDetail) return void loadDetail("index", `/admin/api/indexes/${encodeURIComponent(data.indexDetail)}`);
  if (data.activateIndex && window.confirm("Activate this index and switch search traffic to it?")) return void runAction(() => api(`/admin/api/index/${encodeURIComponent(data.activateIndex || "")}/activate`, { method: "POST" }), "Index activated.");
  if (data.verifyIndex) return void runAction(() => api(`/admin/api/index/${encodeURIComponent(data.verifyIndex || "")}/verify`, { method: "POST" }), "Cache verified.");
  if (data.rebuildIndex) return void runAction(() => api(`/admin/api/index/${encodeURIComponent(data.rebuildIndex || "")}/cache/rebuild`, { method: "POST" }), "Cache rebuilt.");
  if (data.deleteIndex && window.confirm("Delete this index and its cache?")) return void runAction(() => api(`/admin/api/indexes/${encodeURIComponent(data.deleteIndex || "")}`, { method: "DELETE" }), "Index deleted.", () => { if (String(state.details.index?.key || "") === data.deleteIndex) state.details.index = null; });
  if (data.jobDetail) return void loadDetail("job", `/admin/api/jobs/${encodeURIComponent(data.jobDetail)}`);
  if (data.retryJob && window.confirm("Queue this job for retry?")) return void runAction(() => api(`/admin/api/jobs/${encodeURIComponent(data.retryJob || "")}/retry`, { method: "POST" }), "Job queued for retry.");
  if (data.cancelJob && window.confirm("Cancel this job? Running API calls will stop at the next checkpoint.")) return void runAction(async () => { state.details.job = asRecord(await api(`/admin/api/jobs/${encodeURIComponent(data.cancelJob || "")}/cancel`, { method: "POST" })); }, "Job cancellation requested.");
  if (data.auditDetail) return void loadDetail("audit", `/admin/api/audits/${encodeURIComponent(data.auditDetail)}`);
  if (data.closeDetail) { state.details[data.closeDetail as DetailKey] = null; render(); }
}

function handleAdminSubmit(event: SubmitEvent): void {
  const form = event.target as HTMLFormElement;
  if (!(form instanceof HTMLFormElement)) return;
  event.preventDefault();
  const data = new FormData(form);
  if (form.id === "importForm") return void api<Row>("/admin/api/import/dry-run", { method: "POST", body: data }).then((result) => { state.notice = ""; state.draftImport = result; render(); }).catch(showError);
  if (form.id === "problemFilterForm") return setFilter("problems", data, ["q", "source_key", "enabled", "deleted"], { offset: 0 });
  if (form.id === "jobFilterForm") return setFilter("jobs", data, ["type", "status"]);
  if (form.id === "auditFilterForm") return setFilter("audits", data, ["q", "date_from", "date_to", "status"]);
  if (form.id === "problemEditForm") return void patchProblem(String(data.get("key") || ""), { title: String(data.get("title") || ""), url: String(data.get("url") || ""), text: String(data.get("text") || ""), enabled: data.get("enabled") === "on", deleted: data.get("deleted") === "on" }, { keepDetail: true });
}

function bindLogin(): void {
  rootEl.querySelector<HTMLFormElement>("#adminLoginForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const password = rootEl.querySelector<HTMLInputElement>("#adminPassword")?.value || "";
    void api("/admin/api/auth/login", { method: "POST", body: JSON.stringify({ password }) }).then(() => { state.authenticated = true; state.error = ""; return loadPage(); }).catch(showError);
  });
}
function bindAdmin(): void { rootEl.onclick = handleAdminClick; rootEl.onsubmit = handleAdminSubmit; }
function setFilter(key: keyof AdminState["filters"], data: FormData, fields: string[], extra: Row = {}): void {
  const current = state.filters[key] as unknown as Row;
  const updates = Object.fromEntries(fields.map((field) => [field, String(data.get(field) || "")]));
  (state.filters as unknown as Record<string, Row>)[key] = { ...current, ...updates, ...extra };
  void loadPage();
}
function patchProblemBody(data: DOMStringMap): Row | null {
  const body: Row = {};
  if (data.enabled !== undefined) body.enabled = data.enabled === "true";
  if (data.deleted !== undefined) body.deleted = data.deleted === "true";
  if (body.enabled === false && !window.confirm("Disable this problem?")) return null;
  if (body.deleted === true && !window.confirm("Delete this problem? It can be restored later.")) return null;
  return body;
}
async function loadDetail(key: DetailKey, path: string): Promise<void> { try { state.details[key] = await api<Row>(path); render(); } catch (error) { showError(error); } }
async function runAction(action: () => Promise<unknown>, success: string, beforeLoad?: () => void): Promise<void> { try { await action(); beforeLoad?.(); state.notice = success; await loadPage(); } catch (error) { showError(error); } }
async function patchProblem(key: string, body: Row, options: { keepDetail?: boolean } = {}): Promise<void> {
  try {
    const updated = await api<Row>(`/admin/api/problems/${pathKey(key)}`, { method: "PATCH", body: JSON.stringify(body) });
    state.notice = "Problem updated.";
    if (options.keepDetail) state.details.problem = updated;
    await loadPage();
  } catch (error) {
    if ((error as Error).message !== "Canceled") showError(error);
  }
}
function scheduleAutoRefresh(): void {
  stopAutoRefresh();
  if (state.loading || !state.authenticated || !pageHasActiveJob()) return;
  refreshTimer = window.setTimeout(() => { refreshTimer = null; void loadPage({ silent: true }); }, 2500);
}
function stopAutoRefresh(): void { if (refreshTimer !== null) window.clearTimeout(refreshTimer); refreshTimer = null; }
function pageHasActiveJob(): boolean {
  if (state.page === "dashboard") return isActiveJobStatus(asRecord(asRecord(state.data.dashboard).current_job).status);
  if (state.page === "jobs") return asRows(asRecord(state.data.jobs).items).some((job) => isActiveJobStatus(job.status)) || isActiveJobStatus(asRecord(state.details.job).status);
  return false;
}
function isActiveJobStatus(raw: unknown): boolean { return ["queued", "running"].includes(String(raw || "")); }
function showError(error: unknown): void { state.error = (error as Error).message; state.notice = ""; render(); }
function pageFromHash(): AdminPage { const raw = window.location.hash.replace("#", ""); return pages.some((page) => page.id === raw) ? (raw as AdminPage) : "dashboard"; }
function query(values: Record<string, string | number>): string { const params = new URLSearchParams(); Object.entries(values).forEach(([key, value]) => { if (value !== "" && value !== null && value !== undefined) params.set(key, String(value)); }); return params.toString(); }
function select(name: string, selected: string, options: [string, string][]): string { return `<select name="${h(name)}">${options.map(([value, text]) => `<option value="${h(value)}" ${value === selected ? "selected" : ""}>${h(text)}</option>`).join("")}</select>`; }
function statusBadge(raw: unknown): string { const status = String(raw || "unknown"); return `<mark class="status ${h(status)}">${h(label(statusLabels, status) || "Unknown")}</mark>`; }
function display(raw: unknown, kind: DisplayKind = "text"): string {
  if (kind === "key") return shortKey(raw);
  if (kind === "date") return shortDate(raw);
  if (kind === "status") return statusBadge(raw);
  if (kind === "jobType") return h(label(jobTypeLabels, raw));
  if (kind === "cost") return formatCost(raw);
  return h(v(raw));
}
function formatProgress(type: unknown, progress: unknown): string {
  const data = asRecord(parseMaybeJson(progress));
  if (!Object.keys(data).length) return "";
  const prefix = data.cancel_requested ? "Cancel requested, " : "";
  const stats = asRecord(data.stats);
  if (Object.keys(stats).length) return `${prefix}New: ${Number(stats.new || 0)}, Updated: ${Number(stats.overwrite || 0)}, Skipped: ${Number(stats.skip || 0)}, Errors: ${Array.isArray(stats.errors) ? stats.errors.length : Number(stats.errors || 0)}`;
  const processed = Number(data.processed || 0);
  const total = Number(data.total || 0);
  if (total > 0) return `${prefix}${processed} / ${total} ${String(type || "") === "import" ? "rows" : "problems"}${Number(data.failures || 0) ? `, ${Number(data.failures || 0)} failed` : ""}`;
  const artifactTotal = Number(data.total_rewrites || 0) + Number(data.total_embeddings || 0);
  if (artifactTotal > 0) return `${prefix}${Number(data.succeeded_rewrites || 0) + Number(data.succeeded_embeddings || 0)} / ${artifactTotal} artifacts`;
  const phase = String(data.phase || "");
  return phase ? `${prefix}${titleCase(phase.replace(/_/g, " "))}` : compactJson(progress);
}
function importFilename(item: Row): string { const payload = asRecord(parseMaybeJson(item.payload)); const filename = String(item.filename || payload.filename || ""); return filename || String(payload.path || "").split(/[\\/]/).filter(Boolean).pop() || String(item.key || ""); }
function shortDate(raw: unknown, withSeconds = false): string {
  const text = String(raw || "");
  if (!text) return "";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return h(text);
  return `<span title="${h(text)}">${date.toLocaleDateString("en", { month: "short", day: "numeric" })}, ${date.toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit", second: withSeconds ? "2-digit" : undefined, hour12: false })}</span>`;
}
function formatCost(raw: unknown): string { const parsed = parseMaybeJson(raw); const cost = asRecord(parsed); const micro = Number(typeof parsed === "number" ? parsed : cost.microusd || 0); return `<span title="${Number.isFinite(micro) ? micro : 0} microUSD">$${((Number.isFinite(micro) ? micro : 0) / 1_000_000).toFixed(6)}</span>`; }
function settingsValue(raw: unknown): string { return raw === null || raw === undefined ? "" : typeof raw === "object" ? formatJson(raw) : String(raw); }
function compactJson(raw: unknown): string { const text = formatJson(raw); return text.length > 120 ? `${text.slice(0, 117)}...` : text; }
function formatJson(raw: unknown): string { const parsed = parseMaybeJson(raw); return typeof parsed === "string" ? parsed : JSON.stringify(parsed ?? {}, null, 2); }
function parseMaybeJson(raw: unknown): unknown { if (typeof raw !== "string") return raw; try { return JSON.parse(raw); } catch { return raw; } }
function label(map: Record<string, string>, raw: unknown): string { const key = String(raw || ""); return map[key] || key; }
function shortKey(raw: unknown): string { const text = String(raw || ""); return text.length > 12 ? `<span title="${h(text)}">${h(text.slice(0, 10))}...</span>` : h(text); }
function truncateText(raw: unknown, maxLength: number): string { const text = String(raw || ""); return h(text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text); }
function phaseFromMessage(raw: unknown): string { const message = String(raw || "").toLowerCase(); return message.includes("rewrite") ? "rewrite" : message.includes("embedding") ? "embedding" : message.includes("import") ? "import" : ""; }
function titleCase(text: string): string { return text.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function flag(raw: unknown): string { return truthy(raw) ? "\u2713" : "\u2717"; }
function v(raw: unknown): string { return raw === null || raw === undefined ? "" : String(raw); }
function h(raw: unknown): string { return escapeHtml(String(raw ?? "")); }
function asRecord(raw: unknown): Row { return raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Row) : {}; }
function asRows(raw: unknown): Row[] { return Array.isArray(raw) ? raw.filter((item): item is Row => Boolean(item) && typeof item === "object") : []; }
function truthy(raw: unknown): boolean { return raw === true || raw === "true" || raw === 1 || raw === "1"; }
function pathKey(key: string): string { return key.split("/").map(encodeURIComponent).join("/"); }
