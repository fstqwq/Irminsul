import "misans-vf/lib/MiSans.min.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "temml/dist/Temml-Local.css";
import { fetchConfig, fetchHealth, streamSearch, type StreamEvent } from "./api";
import { renderApp, type Actions } from "./search";
import {
  createInitialState,
  initialStages,
  saveResultView,
  stageOrder,
  type RewritePayload,
  type ResultView,
  type SortMode
} from "./state";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root");
const appRoot = root;

if (window.location.pathname.startsWith("/admin")) {
  void startAdminApp(appRoot);
} else {
  startSearchApp();
}

async function startAdminApp(root: HTMLElement): Promise<void> {
  document.documentElement.dataset.theme = "light";
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "/pico.classless.blue.min.css";
  document.head.append(link);
  const style = document.createElement("style");
  style.textContent = ":root[data-theme=light]{--pico-font-family:var(--sans);--pico-font-size:100%;--pico-spacing:.75rem;--pico-block-spacing-vertical:.8rem;--pico-form-element-spacing-vertical:.38rem;--pico-form-element-spacing-horizontal:.6rem}.admin-app{width:min(1280px,calc(100vw - 48px));margin:auto;padding:18px 0}#root .admin-app>header,#root article>header{display:flex;align-items:center;justify-content:space-between;gap:1rem}#root h1{font-size:1.6rem;margin:0}#root h2{font-size:1.15rem}#root nav{align-items:center;gap:1rem}#root nav ul:last-child{justify-content:flex-end;flex-wrap:wrap}#root :is(input,select,textarea,button){font-size:.875rem}#root :is(button,[type=submit],[type=button],[type=reset]){width:auto;padding:.42rem .68rem;line-height:1.2}#root :is(#importForm,#problemFilterForm,#jobFilterForm,#auditFilterForm){display:grid;grid-template-columns:repeat(4,minmax(0,1fr)) auto;gap:.6rem;align-items:end}#root #importForm{grid-template-columns:minmax(220px,1fr) 220px auto}#root article{overflow:auto}#root table{white-space:nowrap}#root .audit-query{display:block;max-width:72ch;white-space:normal;overflow-wrap:anywhere;line-height:1.45}#root td button{margin:.1rem .2rem .1rem 0}#root mark{background:#fee2e2;color:#b42318;padding:.08rem .25rem}#root mark.status{background:#eef2ff;color:#3730a3}#root mark.succeeded,#root mark.active,#root mark.built{background:#dcfce7;color:#166534}#root mark.failed,#root mark.blocked{background:#fee2e2;color:#b42318}#root mark.running,#root mark.building{background:#dbeafe;color:#1d4ed8}#root mark.queued,#root mark.draft,#root mark.retired{background:#f1f5f9;color:#475569}#root textarea{width:100%}@media(max-width:720px){.admin-app{width:calc(100vw - 24px);padding:12px 0}#root nav,#root .admin-app>header{display:block}#root :is(#importForm,#problemFilterForm,#jobFilterForm,#auditFilterForm){display:block}#root :is(button,[type=submit],[type=button],[type=reset]){width:100%}}";
  document.head.append(style);
  const { startAdmin } = await import("./admin");
  startAdmin(root);
}

function startSearchApp(): void {
const state = createInitialState();
let abortController: AbortController | null = null;

const actions: Actions = {
  setQuery(value) {
    state.queryText = value;
    const button = appRoot.querySelector<HTMLButtonElement>(".search-button");
    if (button) button.disabled = state.isRunning || !state.queryText.trim();
  },
  submit() {
    void runSearch();
  },
  toggleSettings() {
    state.settingsOpen = !state.settingsOpen;
    render();
  },
  setUseRewrite(value) {
    state.useRewrite = value;
    render();
  },
  setUseRerank(value) {
    state.useRerank = value;
    if (!value && state.sortMode === "rerank") state.sortMode = "combined";
    render();
  },
  setSortMode(value: SortMode) {
    state.sortMode = value;
    render();
  },
  setResultView(value: ResultView) {
    state.resultView = value;
    saveResultView(value);
    render();
  },
  toggleRewrite() {
    state.rewriteOpen = !state.rewriteOpen;
    state.settingsOpen = false;
    render();
  },
  setEditRewrite(view, value) {
    state.editRewrite = {
      ...state.editRewrite,
      [view]: value
    };
  },
  resubmitRewrite() {
    if (!state.editRewrite.statement.trim()) return;
    void runSearch(state.editRewrite);
  },
};

function render(): void {
  renderApp(appRoot, state, actions);
}

async function runSearch(overrides?: RewritePayload): Promise<void> {
  const queryText = state.queryText.trim();
  if (!queryText) return;

  abortController?.abort();
  abortController = new AbortController();

  state.isRunning = true;
  state.hasSearched = true;
  state.error = "";
  state.candidates = [];
  state.cost = null;
  state.stages = initialStages();
  state.settingsOpen = false;
  if (!overrides) {
    state.rewrite = null;
    state.rewriteOpen = false;
    state.editRewrite = {
      clean: "",
      statement: "",
      abstract: "",
      abstract_zh: "",
      raw: ""
    };
  }
  render();

  try {
    await streamSearch(
      {
        query_text: queryText,
        use_rewrite: state.useRewrite,
        use_rerank: state.useRerank,
        beta: state.config.beta,
        edited_clean: overrides?.clean,
        edited_statement: overrides?.statement,
        edited_abstract: overrides?.abstract,
        edited_abstract_zh: overrides?.abstract_zh
      },
      handleEvent,
      abortController.signal
    );
  } catch (error) {
    if ((error as Error).name !== "AbortError") {
      state.error = (error as Error).message;
      markActiveStageError();
    }
  } finally {
    state.isRunning = false;
    render();
  }
}

function handleEvent(event: StreamEvent): void {
  if (event.type === "stage") {
    const { type: _type, ...stage } = event;
    state.stages[stage.name] = stage;
    render();
    return;
  }

  if (event.type === "rewrite") {
    state.rewrite = {
      clean: event.clean,
      statement: event.statement,
      abstract: event.abstract,
      abstract_zh: event.abstract_zh,
      raw: event.raw,
      edited: event.edited
    };
    state.editRewrite = {
      clean: event.clean,
      statement: event.statement,
      abstract: event.abstract,
      abstract_zh: event.abstract_zh,
      raw: event.raw,
      edited: event.edited
    };
    render();
    return;
  }

  if (event.type === "candidates") {
    state.candidates = event.candidates;
    state.cost = event.cost || null;
    if (state.sortMode === "rerank" && !state.candidates.some((candidate) => typeof candidate.rerank_score === "number")) {
      state.sortMode = "combined";
    }
    render();
    return;
  }

  if (event.type === "error") {
    state.error = event.message;
    markActiveStageError();
    render();
  }
}

function markActiveStageError(): void {
  const active = stageOrder.find((name) => state.stages[name].state === "active");
  if (!active) return;
  state.stages[active] = { ...state.stages[active], state: "error" };
}

render();

fetchConfig()
  .then((config) => {
    state.config = config;
    state.useRerank = state.config.default_rerank;
    render();
  })
  .catch(() => {
    render();
  });

fetchHealth()
  .then((health) => {
    state.activeProblemCount = health.problem_count;
    state.sourceCounts = health.source_counts || [];
    render();
  })
  .catch(() => {
    render();
  });
}
