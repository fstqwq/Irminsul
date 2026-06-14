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

type AdminState = {
  page: AdminPage;
  authenticated: boolean;
  loading: boolean;
  error: string;
  data: Record<string, unknown>;
  draftImport: Record<string, unknown> | null;
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

const state: AdminState = {
  page: pageFromHash(),
  authenticated: false,
  loading: true,
  error: "",
  data: {},
  draftImport: null
};

let rootEl: HTMLElement;

export function startAdmin(root: HTMLElement): void {
  rootEl = root;
  window.addEventListener("hashchange", () => {
    state.page = pageFromHash();
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
    const message = await response.text();
    throw new Error(message || `Request failed (${response.status})`);
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
    if (state.page === "problems") state.data.problems = await api("/admin/api/problems?limit=100");
    if (state.page === "sources") state.data.sources = await api("/admin/api/sources");
    if (state.page === "indexes") state.data.indexes = await api("/admin/api/indexes");
    if (state.page === "jobs") state.data.jobs = await api("/admin/api/jobs");
    if (state.page === "audits") state.data.audits = await api("/admin/api/audits");
    if (state.page === "settings") state.data.settings = await api("/admin/api/settings");
  } catch (error) {
    state.error = (error as Error).message;
  } finally {
    state.loading = false;
    render();
  }
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
          ${state.loading ? `<span class="admin-status">Loading</span>` : ""}
        </header>
        ${state.error ? `<div class="admin-error">${escapeHtml(state.error)}</div>` : ""}
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
        <label>Password</label>
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
  if (state.page === "dashboard") return renderDashboard(state.data.dashboard as Record<string, unknown>);
  if (state.page === "imports") return renderImports(state.data.imports as Record<string, unknown>);
  if (state.page === "problems") return renderProblems(state.data.problems as Record<string, unknown>);
  if (state.page === "sources") return renderSources(state.data.sources as Record<string, unknown>);
  if (state.page === "indexes") return renderIndexes(state.data.indexes as Record<string, unknown>);
  if (state.page === "jobs") return renderJobs(state.data.jobs as Record<string, unknown>);
  if (state.page === "audits") return renderAudits(state.data.audits as Record<string, unknown>);
  return renderSettings(state.data.settings as Record<string, unknown>);
}

function renderDashboard(data: Record<string, unknown> = {}): string {
  return `
    <section class="admin-grid">
      ${metric("Problems", data.problem_count)}
      ${metric("Sources", data.source_count)}
      ${metric("Active index", data.active_index_key || "None")}
      ${metric("Today searches", data.today_searches)}
    </section>`;
}

function renderImports(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || []).map(jobRow).join("");
  return `
    <section class="admin-panel">
      <form id="importForm" class="admin-inline-form">
        <input name="file" type="file" accept=".jsonl,application/jsonl">
        <select name="mode">
          <option value="upsert">upsert</option>
          <option value="insert_only">insert_only</option>
          <option value="sync_source">sync_source</option>
        </select>
        <button type="submit">Dry run</button>
      </form>
      ${state.draftImport ? renderDraftImport(state.draftImport) : ""}
    </section>
    ${table(["Key", "Status", "Progress", "Updated"], items || emptyRow(4))}`;
}

function renderDraftImport(draft: Record<string, unknown>): string {
  const stats = (draft.stats || {}) as Record<string, unknown>;
  return `
    <div class="admin-draft">
      <span>new ${value(stats.new)}</span>
      <span>overwrite ${value(stats.overwrite)}</span>
      <span>skip ${value(stats.skip)}</span>
      <span>errors ${Array.isArray(stats.errors) ? stats.errors.length : 0}</span>
      <button data-confirm-import="${escapeHtml(String(draft.job_key))}" type="button">Confirm</button>
    </div>`;
}

function renderProblems(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || [])
    .map(
      (item) => `
      <tr>
        <td>${escapeHtml(String(item.key))}</td>
        <td>${escapeHtml(String(item.title))}</td>
        <td>${escapeHtml(String(item.source_key))}</td>
        <td>${flag(item.enabled)}</td>
        <td>${flag(item.deleted)}</td>
        <td class="admin-actions">
          <button data-problem-action="disable" data-key="${escapeHtml(String(item.key))}">Disable</button>
          <button data-problem-action="delete" data-key="${escapeHtml(String(item.key))}">Delete</button>
        </td>
      </tr>`
    )
    .join("");
  return table(["Key", "Title", "Source", "Enabled", "Deleted", ""], items || emptyRow(6));
}

function renderSources(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || [])
    .map(
      (item) => `
      <tr>
        <td>${escapeHtml(String(item.key))}</td>
        <td>${escapeHtml(String(item.name))}</td>
        <td>${flag(item.enabled)}</td>
        <td>${value(item.problem_count)}</td>
        <td>${value(item.active_count)}</td>
        <td><button data-source-toggle="${escapeHtml(String(item.key))}" data-enabled="${String(item.enabled)}">Toggle</button></td>
      </tr>`
    )
    .join("");
  return table(["Key", "Name", "Enabled", "Problems", "Active", ""], items || emptyRow(6));
}

function renderIndexes(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || [])
    .map(
      (item) => `
      <tr>
        <td>${escapeHtml(String(item.key))}</td>
        <td>${escapeHtml(String(item.status))}</td>
        <td>${escapeHtml(String(item.created_at))}</td>
        <td><button data-activate-index="${escapeHtml(String(item.key))}">Activate</button></td>
      </tr>`
    )
    .join("");
  return `
    <section class="admin-panel"><button id="buildIndex" type="button">Build index</button></section>
    ${table(["Key", "Status", "Created", ""], items || emptyRow(4))}`;
}

function renderJobs(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || []).map(jobRow).join("");
  return table(["Key", "Status", "Progress", "Updated"], items || emptyRow(4));
}

function renderAudits(data: Record<string, unknown> = {}): string {
  const items = ((data.items as Record<string, unknown>[]) || [])
    .map(
      (item) => `
      <tr>
        <td>${escapeHtml(String(item.started_at))}</td>
        <td>${escapeHtml(String(item.status))}</td>
        <td>${escapeHtml(String(item.query)).slice(0, 120)}</td>
        <td>${value((item.cost as Record<string, unknown>)?.microusd)}</td>
      </tr>`
    )
    .join("");
  return table(["Started", "Status", "Query", "microusd"], items || emptyRow(4));
}

function renderSettings(data: Record<string, unknown> = {}): string {
  return `<pre class="admin-json">${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
}

function metric(label: string, raw: unknown): string {
  return `<div class="admin-metric"><span>${label}</span><b>${escapeHtml(String(raw ?? "0"))}</b></div>`;
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

function jobRow(item: Record<string, unknown>): string {
  return `
    <tr>
      <td>${escapeHtml(String(item.key))}</td>
      <td>${escapeHtml(String(item.status))}</td>
      <td><code>${escapeHtml(JSON.stringify(item.progress || {}))}</code></td>
      <td>${escapeHtml(String(item.updated_at || ""))}</td>
    </tr>`;
}

function value(raw: unknown): string {
  return escapeHtml(String(raw ?? "0"));
}

function flag(raw: unknown): string {
  return Number(raw) === 1 || raw === true ? "yes" : "no";
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

  rootEl.querySelector<HTMLFormElement>("#importForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const formData = new FormData(form);
    void api<Record<string, unknown>>("/admin/api/import/dry-run", {
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
      void api(`/admin/api/import/${encodeURIComponent(key)}/confirm`, { method: "POST" })
        .then(() => {
          state.draftImport = null;
          return loadPage();
        })
        .catch(showError);
    });
  });

  rootEl.querySelector<HTMLButtonElement>("#buildIndex")?.addEventListener("click", () => {
    void api("/admin/api/index/build", { method: "POST" }).then(loadPage).catch(showError);
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-activate-index]").forEach((button) => {
    button.addEventListener("click", () => {
      void api(`/admin/api/index/${encodeURIComponent(button.dataset.activateIndex || "")}/activate`, {
        method: "POST"
      })
        .then(loadPage)
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-problem-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.key || "";
      const action = button.dataset.problemAction || "";
      void api(`/admin/api/problems/batch-${action}`, {
        method: "POST",
        body: JSON.stringify({ keys: [key] })
      })
        .then(loadPage)
        .catch(showError);
    });
  });

  rootEl.querySelectorAll<HTMLButtonElement>("[data-source-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sourceToggle || "";
      const enabled = button.dataset.enabled === "1" || button.dataset.enabled === "true";
      void api(`/admin/api/sources/${encodeURIComponent(key)}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled: !enabled })
      })
        .then(loadPage)
        .catch(showError);
    });
  });
}

function showError(error: unknown): void {
  state.error = (error as Error).message;
  render();
}
