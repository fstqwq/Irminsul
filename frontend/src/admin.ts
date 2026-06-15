import { escapeHtml } from "./math";

type AdminPage =
  | "dashboard"
  | "imports"
  | "problems"
  | "sources"
  | "indexes"
  | "jobs"
  | "audits"
  | "settings";

type Row = Record<string, unknown>;

type ProblemFilters = {
  q: string;
  source_key: string;
  enabled: string;
  deleted: string;
  limit: number;
  offset: number;
};

type JobFilters = {
  type: string;
  status: string;
  limit: number;
};

type AuditFilters = {
  status: string;
  q: string;
  date_from: string;
  date_to: string;
  limit: number;
};

type AdminState = {
  page: AdminPage;
  authenticated: boolean;
  loading: boolean;
  error: string;
  notice: string;
  data: Record<string, unknown>;
  draftImport: Row | null;
  details: {
    import: Row | null;
    problem: Row | null;
    index: Row | null;
    job: Row | null;
    audit: Row | null;
  };
  filters: {
    problems: ProblemFilters;
    jobs: JobFilters;
    audits: AuditFilters;
  };
};

const pages: { id: AdminPage; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "imports", label: "Imports" },
  { id: "problems", label: "Problems" },
  { id: "sources", label: "Sources" },
  { id: "indexes", label: "Indexes" },
  { id: "jobs", label: "Jobs" },
  { id: "audits", label: "Audits" },
  { id: "settings", label: "Settings" }
];

const importModeLabels: Record<string, string> = {
  upsert: "Update or add",
  insert_only: "Add new only (skip existing)",
  sync_source: "Sync source (disable missing)"
};

const jobTypeLabels: Record<string, string> = {
  import: "Import",
  build_index: "Build Index",
  activate_index: "Activate Index",
  cleanup: "Cleanup"
};

const statusLabels: Record<string, string> = {
  draft: "Draft",
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  blocked: "Blocked",
  failed: "Failed",
  building: "Building",
  built: "Ready",
  active: "Active",
  retired: "Retired"
};

const settingsKeyLabels: Record<string, string> = {
  db_path: "Database path",
  upload_dir: "Upload directory",
  index_cache_dir: "Cache directory",
  top_per_doc_view: "Candidates per view",
  top_retrieval: "Retrieved candidates",
  top_display: "Displayed results",
  rerank_top_k: "Rerank candidates",
  alpha: "Alpha",
  beta: "Beta (rerank weight)",
  default_rerank: "Rerank by default",
  rerank_range_floor: "Rerank range floor",
  embedding_range_floor: "Embedding range floor",
  keep_retired: "Retained old indexes",
  load_mode: "Load mode",
  activation_drain_timeout_seconds: "Activation drain timeout (s)",
  name: "Provider",
  model: "Model",
  url: "Endpoint",
  api_key_env: "API key environment variable"
};

const state: AdminState = {
  page: pageFromHash(),
  authenticated: false,
  loading: true,
  error: "",
  notice: "",
  data: {},
  draftImport: null,
  details: {
    import: null,
    problem: null,
    index: null,
    job: null,
    audit: null
  },
  filters: {
    problems: {
      q: "",
      source_key: "",
      enabled: "",
      deleted: "false",
      limit: 25,
      offset: 0
    },
    jobs: {
      type: "",
      status: "",
      limit: 50
    },
    audits: {
      status: "",
      q: "",
      date_from: "",
      date_to: "",
      limit: 50
    }
  }
};

let rootEl: HTMLElement;

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

function pageFromHash(): AdminPage {
  const raw = window.location.hash.replace("#", "");
  return pages.some((page) => page.id === raw) ? (raw as AdminPage) : "dashboard";
}

function csrfToken(): string {
  const match = document.cookie.match(/(?:^|; )admin_csrf=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.method && init.method !== "GET") {
    headers.set("X-CSRF-Token", csrfToken());
  }
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "same-origin"
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text || `Request failed (${response.status})`;
    try {
      const parsed = JSON.parse(text) as Row;
      if (parsed.detail) message = String(parsed.detail);
    } catch {
      // Keep the plain response text.
    }
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

async function loadPage(): Promise<void> {
  if (!state.authenticated) {
    render();
    return;
  }
  state.loading = true;
  state.error = "";
  render();
  try {
    if (state.page === "dashboard") state.data.dashboard = await api("/admin/api/dashboard");
    if (state.page === "imports") state.data.imports = await api("/admin/api/imports");
    if (state.page === "problems") {
      state.data.problems = await api(`/admin/api/problems?${query(state.filters.problems)}`);
    }
    if (state.page === "sources") state.data.sources = await api("/admin/api/sources");
    if (state.page === "indexes") state.data.indexes = await api("/admin/api/indexes");
    if (state.page === "jobs") {
      state.data.jobs = await api(`/admin/api/jobs?${query(state.filters.jobs)}`);
    }
    if (state.page === "audits") {
      state.data.audits = await api(`/admin/api/audits?${query(state.filters.audits)}`);
    }
    if (state.page === "settings") state.data.settings = await api("/admin/api/settings");
  } catch (error) {
    state.error = (error as Error).message;
  } finally {
    state.loading = false;
    render();
  }
}

function query(values: Record<string, string | number>): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== "" && value !== null && value !== undefined) params.set(key, String(value));
  });
  return params.toString();
}

function render(): void {
  if (!state.authenticated) {
    rootEl.innerHTML = renderLogin();
    bindLogin();
    return;
  }
  rootEl.innerHTML = `
    <div class="admin-app">
      <aside class="admin-side">
        <div class="admin-brand">Yuantiji Admin</div>
        <nav>${pages.map(renderNavItem).join("")}</nav>
        <button class="admin-logout" id="adminLogout" type="button">Logout</button>
      </aside>
      <main class="admin-main">
        <header class="admin-head">
          <h1>${escapeHtml(pages.find((page) => page.id === state.page)?.label || "Dashboard")}</h1>
          <button class="admin-refresh" id="adminRefresh" type="button">
            ${state.loading ? "Loading" : "Refresh"}
          </button>
        </header>
        ${state.error ? `<div class="admin-error">${escapeHtml(state.error)}</div>` : ""}
        ${state.notice ? `<div class="admin-notice">${escapeHtml(state.notice)}</div>` : ""}
        ${renderPage()}
      </main>
    </div>`;
  bindAdmin();
}

function renderLogin(): string {
  return `
    <main class="admin-login">
      <form id="adminLoginForm">
        <h1>Yuantiji Admin</h1>
        <label for="adminPassword">Password</label>
        <input id="adminPassword" type="password" autocomplete="current-password" autofocus>
        <button type="submit">Login</button>
        ${state.error ? `<p>${escapeHtml(state.error)}</p>` : ""}
      </form>
    </main>`;
}

function renderNavItem(page: { id: AdminPage; label: string }): string {
  return `<a class="${state.page === page.id ? "active" : ""}" href="#${page.id}">${page.label}</a>`;
}

function renderPage(): string {
  if (state.loading) return `<div class="admin-empty">Loading data.</div>`;
  if (state.page === "dashboard") return renderDashboard(asRecord(state.data.dashboard));
  if (state.page === "imports") return renderImports(asRecord(state.data.imports));
  if (state.page === "problems") return renderProblems(asRecord(state.data.problems));
  if (state.page === "sources") return renderSources(asRecord(state.data.sources));
  if (state.page === "indexes") return renderIndexes(asRecord(state.data.indexes));
  if (state.page === "jobs") return renderJobs(asRecord(state.data.jobs));
  if (state.page === "audits") return renderAudits(asRecord(state.data.audits));
  return renderSettings(asRecord(state.data.settings));
}

function renderDashboard(data: Row = {}): string {
  const currentJob = asRecord(data.current_job);
  return `
    <section class="admin-grid">
      ${metric("Problems", data.problem_count)}
      ${metric("Sources", data.source_count)}
      ${metric("Active index", data.active_index_key || "None", "key")}
      ${metric("Today searches", data.today_searches)}
    </section>
    <section class="admin-panel">
      <h2>Current job</h2>
      ${
        Object.keys(currentJob).length
          ? detailGrid([
              ["Job ID", currentJob.key, "key"],
              ["Type", currentJob.type, "jobType"],
              ["Status", currentJob.status, "status"],
              ["Updated", currentJob.updated_at, "date"]
            ]) + htmlBlock("Progress", escapeHtml(formatProgress(currentJob.type, currentJob.progress)))
          : `<div class="admin-empty">No queued or running job.</div>`
      }
    </section>`;
}

function renderImports(data: Row = {}): string {
  const items = asRows(data.items).map(importRow).join("");
  return `
    <section class="admin-panel">
      <h2>Upload JSONL</h2>
      <form id="importForm" class="admin-inline-form">
        <input name="file" type="file" accept=".jsonl,application/jsonl">
        <select name="mode">
          ${Object.entries(importModeLabels)
            .map(
              ([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`
            )
            .join("")}
        </select>
        <button type="submit">Dry run</button>
      </form>
      ${state.draftImport ? renderDraftImport(state.draftImport) : ""}
    </section>
    ${state.details.import ? renderImportDetail(state.details.import) : ""}
    ${table(["Import ID", "Status", "Progress", "Updated", ""], items || emptyRow(5))}`;
}

function importRow(item: Row): string {
  const key = String(item.key || "");
  const status = String(item.status || "");
  return `
    <tr>
      <td>${shortKey(key)}</td>
      <td>${statusBadge(status)}</td>
      <td>${escapeHtml(formatProgress(item.type || "import", item.progress))}</td>
      <td>${shortDate(item.updated_at)}</td>
      <td class="admin-actions">
        <button data-import-detail="${escapeHtml(key)}" type="button">View</button>
        ${status === "draft" ? `<button data-confirm-import="${escapeHtml(key)}" type="button">Confirm</button>` : ""}
      </td>
    </tr>`;
}

function renderDraftImport(draft: Row): string {
  const stats = asRecord(draft.stats);
  const errors = Array.isArray(stats.errors) ? stats.errors : [];
  return `
    <div class="admin-draft">
      <span>New: ${value(stats.new)}</span>
      <span>Updated: ${value(stats.overwrite)}</span>
      <span>Skipped: ${value(stats.skip)}</span>
      <span>Errors: ${errors.length}</span>
      <button data-confirm-import="${escapeHtml(String(draft.job_key))}" type="button">Confirm</button>
    </div>
    ${errors.length ? jsonBlock("Errors", errors) : ""}`;
}

function renderImportDetail(job: Row): string {
  return detailPanel(
    "Import detail",
    detailGrid([
      ["Import ID", job.key, "key"],
      ["Status", job.status, "status"],
      ["Created", job.created_at, "date"],
      ["Updated", job.updated_at, "date"],
      ["Progress", formatProgress(job.type || "import", job.progress)]
    ]) +
      jsonBlock("Result", job.result) +
      jsonBlock("Error", job.error),
    "import"
  );
}

function renderProblems(data: Row = {}): string {
  const filters = state.filters.problems;
  const total = Number(data.total || 0);
  const items = asRows(data.items).map(problemRow).join("");
  const from = total === 0 ? 0 : filters.offset + 1;
  const to = Math.min(filters.offset + filters.limit, total);
  return `
    <section class="admin-panel">
      <h2>Filters</h2>
      <form id="problemFilterForm" class="admin-inline-form">
        <input name="q" value="${escapeHtml(filters.q)}" placeholder="Keyword">
        <input name="source_key" value="${escapeHtml(filters.source_key)}" placeholder="Source">
        ${select("enabled", filters.enabled, [
          ["", "Enabled: any"],
          ["true", "Enabled only"],
          ["false", "Disabled only"]
        ])}
        ${select("deleted", filters.deleted, [
          ["", "Deleted: any"],
          ["false", "Active only"],
          ["true", "Deleted only"]
        ])}
        <button type="submit">Apply</button>
      </form>
    </section>
    ${state.details.problem ? renderProblemEditor(state.details.problem) : ""}
    <div class="admin-toolbar">
      <span>${from}-${to} / ${total}</span>
      <div>
        <button data-page-problems="-1" type="button" ${filters.offset <= 0 ? "disabled" : ""}>Prev</button>
        <button data-page-problems="1" type="button" ${filters.offset + filters.limit >= total ? "disabled" : ""}>Next</button>
      </div>
    </div>
    ${table(["ID", "Title", "Source", "Enabled", "Deleted", "Updated", ""], items || emptyRow(7))}`;
}

function problemRow(item: Row): string {
  const key = String(item.key || "");
  const enabled = truthy(item.enabled);
  const deleted = truthy(item.deleted);
  return `
    <tr>
      <td>${escapeHtml(key)}</td>
      <td>${escapeHtml(String(item.title || ""))}</td>
      <td>${escapeHtml(String(item.source_key || ""))}</td>
      <td>${flag(enabled)}</td>
      <td>${flag(deleted)}</td>
      <td>${shortDate(item.updated_at)}</td>
      <td class="admin-actions">
        <button data-edit-problem="${escapeHtml(key)}" type="button">Preview</button>
        <button data-problem-patch="${escapeHtml(key)}" data-enabled="${enabled ? "false" : "true"}" type="button">
          ${enabled ? "Disable" : "Enable"}
        </button>
        <button data-problem-patch="${escapeHtml(key)}" data-deleted="${deleted ? "false" : "true"}" type="button">
          ${deleted ? "Restore" : "Delete"}
        </button>
      </td>
    </tr>`;
}

function renderProblemEditor(problem: Row): string {
  return detailPanel(
    "Problem detail",
    `
      <form id="problemEditForm" class="admin-edit-form">
        <label>ID<input name="key" value="${escapeHtml(String(problem.key || ""))}" readonly></label>
        <label>Title<input name="title" value="${escapeHtml(String(problem.title || ""))}"></label>
        <label>URL<input name="url" value="${escapeHtml(String(problem.url || ""))}"></label>
        <label><input name="enabled" type="checkbox" ${truthy(problem.enabled) ? "checked" : ""}> Enabled</label>
        <label><input name="deleted" type="checkbox" ${truthy(problem.deleted) ? "checked" : ""}> Deleted</label>
        <label>
          Statement preview
          <textarea readonly rows="16">${escapeHtml(String(problem.text || ""))}</textarea>
        </label>
        <div class="admin-actions">
          <button type="submit">Save metadata</button>
          <button data-close-detail="problem" type="button">Cancel</button>
        </div>
      </form>`,
    "problem"
  );
}

function renderSources(data: Row = {}): string {
  const items = asRows(data.items)
    .map((item) => {
      const key = String(item.key || "");
      const enabled = truthy(item.enabled);
      return `
        <tr>
          <td>${escapeHtml(key)}</td>
          <td>${escapeHtml(String(item.name || ""))}</td>
          <td>${flag(enabled)}</td>
          <td>${value(item.problem_count)}</td>
          <td>${value(item.active_count)}</td>
          <td>${value(item.deleted_count)}</td>
          <td class="admin-actions">
            <button data-source-toggle="${escapeHtml(key)}" data-enabled="${enabled}" type="button">
              ${enabled ? "Disable" : "Enable"}
            </button>
            <button data-source-problems="${escapeHtml(key)}" type="button">Problems</button>
          </td>
        </tr>`;
    })
    .join("");
  return table(
    ["ID", "Name", "Enabled", "Problems", "Active", "Deleted", ""],
    items || emptyRow(7)
  );
}

function renderIndexes(data: Row = {}): string {
  const items = asRows(data.items).map(indexRow).join("");
  return `
    <section class="admin-panel">
      <button id="buildIndex" type="button">Build new index</button>
      <button id="cleanupJob" type="button">Run cleanup</button>
    </section>
    ${state.details.index ? renderIndexDetail(state.details.index) : ""}
    ${table(["ID", "Status", "Created", "Activated", ""], items || emptyRow(5))}`;
}

function indexRow(item: Row): string {
  const key = String(item.key || "");
  const status = String(item.status || "");
  return `
    <tr>
      <td>${shortKey(key)}</td>
      <td>${statusBadge(status)}</td>
      <td>${shortDate(item.created_at)}</td>
      <td>${shortDate(item.activated_at)}</td>
      <td class="admin-actions">
        <button data-index-detail="${escapeHtml(key)}" type="button">View</button>
        <button data-activate-index="${escapeHtml(key)}" type="button">Activate</button>
        <button data-verify-index="${escapeHtml(key)}" type="button">Check integrity</button>
        <button data-rebuild-index="${escapeHtml(key)}" type="button">Rebuild cache</button>
      </td>
    </tr>`;
}

function renderIndexDetail(index: Row): string {
  return detailPanel(
    "Index detail",
    detailGrid([
      ["ID", index.key, "key"],
      ["Status", index.status, "status"],
      ["Created", index.created_at, "date"],
      ["Activated", index.activated_at, "date"]
    ]) +
      renderIndexMeta(index.meta) +
      jsonBlock("Error", index.error),
    "index"
  );
}

function renderJobs(data: Row = {}): string {
  const filters = state.filters.jobs;
  const items = asRows(data.items).map(jobRow).join("");
  return `
    <section class="admin-panel">
      <h2>Filters</h2>
      <form id="jobFilterForm" class="admin-inline-form">
        ${select("type", filters.type, [
          ["", "Type: any"],
          ["import", jobTypeLabels.import],
          ["build_index", jobTypeLabels.build_index],
          ["activate_index", jobTypeLabels.activate_index],
          ["cleanup", jobTypeLabels.cleanup]
        ])}
        ${select("status", filters.status, [
          ["", "Status: any"],
          ["draft", statusLabels.draft],
          ["queued", statusLabels.queued],
          ["running", statusLabels.running],
          ["succeeded", statusLabels.succeeded],
          ["blocked", statusLabels.blocked],
          ["failed", statusLabels.failed]
        ])}
        <button type="submit">Apply</button>
      </form>
    </section>
    ${state.details.job ? renderJobDetail(state.details.job) : ""}
    ${table(["Job ID", "Type", "Status", "Progress", "Updated", ""], items || emptyRow(6))}`;
}

function jobRow(item: Row): string {
  const key = String(item.key || "");
  const status = String(item.status || "");
  return `
    <tr>
      <td>${shortKey(key)}</td>
      <td>${escapeHtml(label(jobTypeLabels, item.type))}</td>
      <td>${statusBadge(status)}</td>
      <td>${escapeHtml(formatProgress(item.type, item.progress))}</td>
      <td>${shortDate(item.updated_at)}</td>
      <td class="admin-actions">
        <button data-job-detail="${escapeHtml(key)}" type="button">View</button>
        ${["blocked", "failed"].includes(status) ? `<button data-retry-job="${escapeHtml(key)}" type="button">Retry</button>` : ""}
      </td>
    </tr>`;
}

function renderJobDetail(job: Row): string {
  const status = String(job.status || "");
  return detailPanel(
    "Job detail",
    detailGrid([
      ["Job ID", job.key, "key"],
      ["Type", job.type, "jobType"],
      ["Status", job.status, "status"],
      ["Created", job.created_at, "date"],
      ["Updated", job.updated_at, "date"],
      ["Progress", formatProgress(job.type, job.progress)]
    ]) +
      jsonBlock("Payload", job.payload) +
      jsonBlock("Result", job.result) +
      jsonBlock("Error", job.error) +
      (["blocked", "failed"].includes(status)
        ? `<button data-retry-job="${escapeHtml(String(job.key || ""))}" type="button">Retry</button>`
        : ""),
    "job"
  );
}

function renderAudits(data: Row = {}): string {
  const filters = state.filters.audits;
  const items = asRows(data.items).map(auditRow).join("");
  return `
    <section class="admin-panel">
      <h2>Filters</h2>
      <form id="auditFilterForm" class="admin-inline-form">
        <input name="q" value="${escapeHtml(filters.q)}" placeholder="Query contains">
        <input name="date_from" value="${escapeHtml(filters.date_from)}" type="date">
        <input name="date_to" value="${escapeHtml(filters.date_to)}" type="date">
        ${select("status", filters.status, [
          ["", "Status: any"],
          ["succeeded", statusLabels.succeeded],
          ["failed", statusLabels.failed]
        ])}
        <button type="submit">Apply</button>
      </form>
    </section>
    ${state.details.audit ? renderAuditDetail(state.details.audit) : ""}
    ${table(["Started", "Status", "Query", "Cost", ""], items || emptyRow(5))}`;
}

function auditRow(item: Row): string {
  const requestId = String(item.request_id || "");
  return `
    <tr>
      <td>${shortDate(item.started_at)}</td>
      <td>${statusBadge(String(item.status || ""))}</td>
      <td>${truncateText(item.query, 160)}</td>
      <td>${formatCost(item.cost)}</td>
      <td><button data-audit-detail="${escapeHtml(requestId)}" type="button">View</button></td>
    </tr>`;
}

function renderAuditDetail(audit: Row): string {
  return detailPanel(
    "Audit detail",
    detailGrid([
      ["Request", audit.request_id, "key"],
      ["Status", audit.status, "status"],
      ["Started", audit.started_at, "date"],
      ["Finished", audit.finished_at, "date"],
      ["Client IP", audit.client_ip],
      ["User agent", audit.user_agent],
      ["Cost", audit.cost, "cost"]
    ]) +
      `<div class="admin-detail-block"><h3>Query</h3><p>${escapeHtml(String(audit.query || ""))}</p></div>` +
      jsonBlock("Timings", audit.timings) +
      jsonBlock("API calls", audit.api_calls) +
      jsonBlock("Result", audit.result) +
      jsonBlock("Error", audit.error),
    "audit"
  );
}

function renderSettings(data: Row = {}): string {
  const models = asRecord(data.models);
  return `
    <section class="admin-settings-grid">
      ${settingsSection("Storage", asRecord(data.storage))}
      ${settingsSection("Search", asRecord(data.search))}
      ${settingsSection("Index cache", asRecord(data.index_cache))}
    </section>
    <section class="admin-panel">
      <h2>Models</h2>
      <div class="admin-settings-grid">
        ${settingsSection("Rewrite", asRecord(models.rewrite))}
        ${settingsSection("Embedding", asRecord(models.embedding))}
        ${settingsSection("Rerank", asRecord(models.rerank))}
      </div>
    </section>`;
}

function settingsSection(title: string, values: Row): string {
  const rows = Object.entries(values)
    .map(
      ([key, raw]) => `
      <tr>
        <th>${escapeHtml(label(settingsKeyLabels, key))}</th>
        <td>${escapeHtml(String(raw ?? ""))}</td>
      </tr>`
    )
    .join("");
  return `
    <section class="admin-panel">
      <h2>${escapeHtml(title)}</h2>
      <table class="admin-kv"><tbody>${rows || emptyRow(2)}</tbody></table>
    </section>`;
}

type DisplayKind = "text" | "key" | "date" | "status" | "jobType" | "cost";

function metric(labelText: string, raw: unknown, kind: DisplayKind = "text"): string {
  return `<div class="admin-metric"><span>${escapeHtml(labelText)}</span><b>${formatDisplay(raw, kind)}</b></div>`;
}

function table(headers: string[], rows: string): string {
  return `
    <div class="admin-table-wrap">
      <table class="admin-table">
        <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function emptyRow(cols: number): string {
  return `<tr><td colspan="${cols}" class="admin-empty">No data.</td></tr>`;
}

function detailPanel(title: string, body: string, key: keyof AdminState["details"]): string {
  return `
    <section class="admin-detail">
      <div class="admin-detail-head">
        <h2>${escapeHtml(title)}</h2>
        <button data-close-detail="${key}" type="button">Close</button>
      </div>
      ${body}
    </section>`;
}

function detailGrid(rows: [string, unknown, DisplayKind?][]): string {
  return `
    <dl class="admin-detail-grid">
      ${rows
        .map(
          ([key, raw, kind]) => `
          <div>
            <dt>${escapeHtml(key)}</dt>
            <dd>${formatDisplay(raw, kind || "text")}</dd>
          </div>`
        )
        .join("")}
    </dl>`;
}

function htmlBlock(title: string, body: string): string {
  if (!body) return "";
  return `
    <div class="admin-detail-block">
      <h3>${escapeHtml(title)}</h3>
      <p>${body}</p>
    </div>`;
}

function jsonBlock(title: string, raw: unknown): string {
  if (raw === null || raw === undefined || raw === "") return "";
  return `
    <div class="admin-detail-block">
      <h3>${escapeHtml(title)}</h3>
      <pre class="admin-json">${escapeHtml(formatJson(raw))}</pre>
    </div>`;
}

function select(name: string, selected: string, options: [string, string][]): string {
  return `
    <select name="${escapeHtml(name)}">
      ${options
        .map(
          ([value, label]) => `
          <option value="${escapeHtml(value)}" ${selected === value ? "selected" : ""}>
            ${escapeHtml(label)}
          </option>`
        )
        .join("")}
    </select>`;
}

function value(raw: unknown): string {
  return escapeHtml(String(raw ?? "0"));
}

function flag(raw: unknown): string {
  return truthy(raw) ? "&#10003;" : "&#10007;";
}

function statusBadge(status: string): string {
  return `<span class="admin-badge status-${escapeHtml(status)}">${escapeHtml(label(statusLabels, status) || "Unknown")}</span>`;
}

function compactJson(raw: unknown): string {
  const text = formatJson(raw);
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

function formatJson(raw: unknown): string {
  const parsed = parseMaybeJson(raw);
  return typeof parsed === "string" ? parsed : JSON.stringify(parsed ?? {}, null, 2);
}

function parseMaybeJson(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

function label(map: Record<string, string>, raw: unknown): string {
  const key = String(raw || "");
  return map[key] || key;
}

function formatDisplay(raw: unknown, kind: DisplayKind = "text"): string {
  if (kind === "key") return shortKey(raw);
  if (kind === "date") return shortDate(raw);
  if (kind === "status") return statusBadge(String(raw || ""));
  if (kind === "jobType") return escapeHtml(label(jobTypeLabels, raw));
  if (kind === "cost") return formatCost(raw);
  return escapeHtml(String(raw ?? ""));
}

function shortKey(raw: unknown): string {
  const text = String(raw || "");
  if (text.length > 12) {
    return `<span title="${escapeHtml(text)}">${escapeHtml(text.slice(0, 10))}...</span>`;
  }
  return escapeHtml(text);
}

function shortDate(raw: unknown): string {
  const text = String(raw || "");
  if (!text) return "";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return escapeHtml(text);
  const display = `${date.toLocaleDateString("en", {
    month: "short",
    day: "numeric"
  })}, ${date.toLocaleTimeString("en", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  })}`;
  return `<span title="${escapeHtml(text)}">${escapeHtml(display)}</span>`;
}

function formatCost(raw: unknown): string {
  const parsed = parseMaybeJson(raw);
  const cost = asRecord(parsed);
  const micro = typeof parsed === "number" ? parsed : Number(cost.microusd || 0);
  const safeMicro = Number.isFinite(micro) ? micro : 0;
  const dollars = (safeMicro / 1_000_000).toFixed(6);
  return `<span title="${escapeHtml(String(safeMicro))} microUSD">$${escapeHtml(dollars)}</span>`;
}

function formatProgress(type: unknown, progress: unknown): string {
  const parsed = parseMaybeJson(progress);
  const data = asRecord(parsed);
  if (!Object.keys(data).length) return "";

  const stats = asRecord(data.stats);
  if (Object.keys(stats).length) {
    const errors = Array.isArray(stats.errors) ? stats.errors.length : Number(stats.errors || 0);
    return `New: ${Number(stats.new || 0)}, Updated: ${Number(stats.overwrite || 0)}, Skipped: ${Number(
      stats.skip || 0
    )}, Errors: ${errors}`;
  }

  const kind = String(type || "");
  const processed = Number(data.processed || 0);
  const total = Number(data.total || 0);
  if (total > 0) {
    const unit = kind === "import" ? "rows" : "problems";
    const failures = Number(data.failures || 0);
    return `${processed} / ${total} ${unit}${failures ? `, ${failures} failed` : ""}`;
  }

  const succeededRewrites = Number(data.succeeded_rewrites || 0);
  const succeededEmbeddings = Number(data.succeeded_embeddings || 0);
  const totalRewrites = Number(data.total_rewrites || 0);
  const totalEmbeddings = Number(data.total_embeddings || 0);
  const artifactTotal = totalRewrites + totalEmbeddings;
  if (artifactTotal > 0) {
    return `${succeededRewrites + succeededEmbeddings} / ${artifactTotal} artifacts`;
  }

  const phase = String(data.phase || "");
  if (phase) return titleCase(phase.replace(/_/g, " "));
  return compactJson(progress);
}

function titleCase(text: string): string {
  return text.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function truncateText(raw: unknown, maxLength: number): string {
  const text = String(raw || "");
  const clipped = text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
  return escapeHtml(clipped);
}

function renderIndexMeta(raw: unknown): string {
  const meta = asRecord(parseMaybeJson(raw));
  if (!Object.keys(meta).length) return "";
  return `
    <div class="admin-detail-block">
      <h3>Index metadata</h3>
      ${detailGrid([
        ["Problem count", meta.problem_count],
        ["Schema version", meta.schema_version],
        ["Rewrite method", meta.rewrite_method_key, "key"],
        ["Embedding method", meta.embedding_method_key, "key"],
        ["Cache path", meta.cache_path]
      ])}
    </div>`;
}

function asRecord(raw: unknown): Row {
  return raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Row) : {};
}

function asRows(raw: unknown): Row[] {
  return Array.isArray(raw) ? raw.filter((item): item is Row => Boolean(item) && typeof item === "object") : [];
}

function truthy(raw: unknown): boolean {
  return raw === true || raw === "true" || raw === 1 || raw === "1";
}

function pathKey(key: string): string {
  return key.split("/").map(encodeURIComponent).join("/");
}

function bindLogin(): void {
  rootEl.querySelector<HTMLFormElement>("#adminLoginForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const password = rootEl.querySelector<HTMLInputElement>("#adminPassword")?.value || "";
    void api("/admin/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ password })
    })
      .then(() => {
        state.authenticated = true;
        state.error = "";
        return loadPage();
      })
      .catch((error) => {
        state.error = (error as Error).message;
        render();
      });
  });
}

function bindAdmin(): void {
  rootEl.querySelector<HTMLButtonElement>("#adminLogout")?.addEventListener("click", () => {
    void api("/admin/api/auth/logout", { method: "POST" }).finally(() => {
      state.authenticated = false;
      render();
    });
  });

  rootEl.querySelector<HTMLButtonElement>("#adminRefresh")?.addEventListener("click", () => {
    void loadPage();
  });

  bindImports();
  bindProblems();
  bindSources();
  bindIndexes();
  bindJobs();
  bindAudits();
  bindDetailClosers();
}

function bindImports(): void {
  rootEl.querySelector<HTMLFormElement>("#importForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const formData = new FormData(form);
    state.notice = "";
    void api<Row>("/admin/api/import/dry-run", {
      method: "POST",
      body: formData
    })
      .then((result) => {
        state.draftImport = result;
        render();
      })
      .catch(showError);
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-confirm-import]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.confirmImport || "";
      if (!window.confirm("Start this import job now?")) return;
      void api(`/admin/api/import/${encodeURIComponent(key)}/confirm`, { method: "POST" })
        .then(() => {
          state.draftImport = null;
          state.notice = "Import queued.";
          return loadPage();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-import-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.importDetail || "";
      void api<Row>(`/admin/api/imports/${encodeURIComponent(key)}`)
        .then((detail) => {
          state.details.import = detail;
          render();
        })
        .catch(showError);
    });
  });
}

function bindProblems(): void {
  rootEl.querySelector<HTMLFormElement>("#problemFilterForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget as HTMLFormElement);
    state.filters.problems = {
      ...state.filters.problems,
      q: String(form.get("q") || ""),
      source_key: String(form.get("source_key") || ""),
      enabled: String(form.get("enabled") || ""),
      deleted: String(form.get("deleted") || ""),
      offset: 0
    };
    void loadPage();
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-page-problems]").forEach((button) => {
    button.addEventListener("click", () => {
      const direction = Number(button.dataset.pageProblems || 0);
      const filters = state.filters.problems;
      filters.offset = Math.max(0, filters.offset + direction * filters.limit);
      void loadPage();
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-edit-problem]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.editProblem || "";
      void api<Row>(`/admin/api/problems/${pathKey(key)}`)
        .then((detail) => {
          state.details.problem = detail;
          render();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-problem-patch]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.problemPatch || "";
      const body: Row = {};
      if (button.dataset.enabled !== undefined) body.enabled = button.dataset.enabled === "true";
      if (button.dataset.deleted !== undefined) body.deleted = button.dataset.deleted === "true";
      if (body.enabled === false && !window.confirm("Disable this problem?")) return;
      if (body.deleted === true && !window.confirm("Delete this problem? It can be restored later.")) return;
      void patchProblem(key, body);
    });
  });

  rootEl.querySelector<HTMLFormElement>("#problemEditForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget as HTMLFormElement);
    const key = String(form.get("key") || "");
    const body = {
      title: String(form.get("title") || ""),
      url: String(form.get("url") || ""),
      enabled: form.get("enabled") === "on",
      deleted: form.get("deleted") === "on"
    };
    state.details.problem = null;
    void patchProblem(key, body);
  });
}

async function patchProblem(key: string, body: Row): Promise<void> {
  await api(`/admin/api/problems/${pathKey(key)}`, {
    method: "PATCH",
    body: JSON.stringify(body)
  });
  state.notice = "Problem updated.";
  await loadPage();
}

function bindSources(): void {
  rootEl.querySelectorAll<HTMLButtonElement>("[data-source-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sourceToggle || "";
      const enabled = button.dataset.enabled === "true";
      if (enabled && !window.confirm("Disable this source?")) return;
      void api(`/admin/api/sources/${encodeURIComponent(key)}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled: !enabled })
      })
        .then(() => {
          state.notice = "Source updated.";
          return loadPage();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-source-problems]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.problems.source_key = button.dataset.sourceProblems || "";
      state.filters.problems.offset = 0;
      window.location.hash = "problems";
    });
  });
}

function bindIndexes(): void {
  rootEl.querySelector<HTMLButtonElement>("#buildIndex")?.addEventListener("click", () => {
    void api("/admin/api/index/build", { method: "POST" })
      .then(() => {
        state.notice = "Index build queued.";
        return loadPage();
      })
      .catch(showError);
  });

  rootEl.querySelector<HTMLButtonElement>("#cleanupJob")?.addEventListener("click", () => {
    void api("/admin/api/jobs/cleanup", { method: "POST" })
      .then(() => {
        state.notice = "Cleanup queued.";
        return loadPage();
      })
      .catch(showError);
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-index-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.indexDetail || "";
      void api<Row>(`/admin/api/indexes/${encodeURIComponent(key)}`)
        .then((detail) => {
          state.details.index = detail;
          render();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-activate-index]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!window.confirm("Activate this index and switch search traffic to it?")) return;
      void api(`/admin/api/index/${encodeURIComponent(button.dataset.activateIndex || "")}/activate`, {
        method: "POST"
      })
        .then(() => {
          state.notice = "Index activated.";
          return loadPage();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-verify-index]").forEach((button) => {
    button.addEventListener("click", () => {
      void api(`/admin/api/index/${encodeURIComponent(button.dataset.verifyIndex || "")}/verify`, {
        method: "POST"
      })
        .then(() => {
          state.notice = "Cache verified.";
          return loadPage();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-rebuild-index]").forEach((button) => {
    button.addEventListener("click", () => {
      void api(`/admin/api/index/${encodeURIComponent(button.dataset.rebuildIndex || "")}/cache/rebuild`, {
        method: "POST"
      })
        .then(() => {
          state.notice = "Cache rebuilt.";
          return loadPage();
        })
        .catch(showError);
    });
  });
}

function bindJobs(): void {
  rootEl.querySelector<HTMLFormElement>("#jobFilterForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget as HTMLFormElement);
    state.filters.jobs = {
      ...state.filters.jobs,
      type: String(form.get("type") || ""),
      status: String(form.get("status") || "")
    };
    void loadPage();
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-job-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.jobDetail || "";
      void api<Row>(`/admin/api/jobs/${encodeURIComponent(key)}`)
        .then((detail) => {
          state.details.job = detail;
          render();
        })
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-retry-job]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.retryJob || "";
      if (!window.confirm("Queue this job for retry?")) return;
      void api(`/admin/api/jobs/${encodeURIComponent(key)}/retry`, { method: "POST" })
        .then(() => {
          state.notice = "Job queued for retry.";
          return loadPage();
        })
        .catch(showError);
    });
  });
}

function bindAudits(): void {
  rootEl.querySelector<HTMLFormElement>("#auditFilterForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget as HTMLFormElement);
    state.filters.audits = {
      ...state.filters.audits,
      q: String(form.get("q") || ""),
      date_from: String(form.get("date_from") || ""),
      date_to: String(form.get("date_to") || ""),
      status: String(form.get("status") || "")
    };
    void loadPage();
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-audit-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const requestId = button.dataset.auditDetail || "";
      void api<Row>(`/admin/api/audits/${encodeURIComponent(requestId)}`)
        .then((detail) => {
          state.details.audit = detail;
          render();
        })
        .catch(showError);
    });
  });
}

function bindDetailClosers(): void {
  rootEl.querySelectorAll<HTMLButtonElement>("[data-close-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.closeDetail as keyof AdminState["details"];
      state.details[key] = null;
      render();
    });
  });
}

function showError(error: unknown): void {
  state.error = (error as Error).message;
  state.notice = "";
  render();
}
