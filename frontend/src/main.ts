import "misans-vf/lib/MiSans.min.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "temml/dist/Temml-Local.css";
import { fetchConfig, streamSearch, type StreamEvent } from "./api";
import { startAdmin } from "./admin";
import { renderApp, type Actions } from "./render";
import { createInitialState, initialStages, stageOrder, type SortMode } from "./state";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root");
const appRoot = root;

if (window.location.pathname.startsWith("/admin")) {
  startAdmin(appRoot);
} else {
  startSearchApp();
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
  setAlpha(value) {
    state.alpha = value;
    render();
  },
  setSortMode(value: SortMode) {
    state.sortMode = value;
    render();
  },
  toggleRewrite() {
    state.rewriteOpen = !state.rewriteOpen;
    state.settingsOpen = false;
    render();
  },
  setEditStatement(value) {
    state.editStatement = value;
  },
  setEditAbstract(value) {
    state.editAbstract = value;
  },
  resubmitRewrite() {
    if (!state.editStatement.trim()) return;
    void runSearch({
      statement: state.editStatement,
      abstract: state.editAbstract
    });
  },
  toggleResult(id) {
    if (!id) return;
    if (state.expanded.has(id)) state.expanded.delete(id);
    else state.expanded.add(id);
    render();
  }
};

function render(): void {
  renderApp(appRoot, state, actions);
}

async function runSearch(overrides?: { statement?: string; abstract?: string }): Promise<void> {
  const queryText = state.queryText.trim();
  if (!queryText) return;

  abortController?.abort();
  abortController = new AbortController();

  state.isRunning = true;
  state.hasSearched = true;
  state.error = "";
  state.candidates = [];
  state.stages = initialStages();
  state.expanded = new Set<string>();
  state.settingsOpen = false;
  if (!overrides) {
    state.rewrite = null;
    state.rewriteOpen = false;
    state.editStatement = "";
    state.editAbstract = "";
  }
  render();

  try {
    await streamSearch(
      {
        query_text: queryText,
        use_rewrite: state.useRewrite,
        use_rerank: state.useRerank,
        alpha: state.alpha,
        beta: state.config.default_beta,
        edited_statement: overrides?.statement,
        edited_abstract: overrides?.abstract
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
      statement: event.statement,
      abstract: event.abstract,
      raw: event.raw,
      edited: event.edited
    };
    state.editStatement = event.statement;
    state.editAbstract = event.abstract;
    render();
    return;
  }

  if (event.type === "candidates") {
    state.candidates = event.candidates;
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
    state.useRerank = config.default_rerank;
    state.alpha = config.default_alpha;
    render();
  })
  .catch(() => {
    render();
  });
}
