import { escapeHtml, renderMathText } from "./math";
import { icon, renderIcons } from "./icons";
import { installMathCopy } from "./copy";
import {
  AppState,
  Candidate,
  SortMode,
  candidateScore,
  hasRerankScores,
  sortedCandidates,
  sortLabels,
  stageLabels,
  stageOrder
} from "./state";

export type Actions = {
  setQuery(value: string): void;
  submit(): void;
  toggleSettings(): void;
  setUseRewrite(value: boolean): void;
  setUseRerank(value: boolean): void;
  setAlpha(value: number): void;
  setSortMode(value: SortMode): void;
  toggleRewrite(): void;
  setEditStatement(value: string): void;
  setEditAbstract(value: string): void;
  resubmitRewrite(): void;
  toggleResult(id: string): void;
};

function formatElapsed(value?: number): string {
  if (typeof value !== "number") return "";
  if (value < 0.1) return `${Math.round(value * 1000)}ms`;
  return `${value.toFixed(1)}s`;
}

function sourceLabel(candidate: Candidate): string {
  if (!candidate.url) return candidate.problem_id;

  try {
    const url = new URL(candidate.url);
    if (url.hostname.includes("codeforces.com")) {
      const parts = url.pathname.split("/").filter(Boolean);
      const problemsetIndex = parts.indexOf("problemset");
      if (problemsetIndex >= 0 && parts[problemsetIndex + 2] && parts[problemsetIndex + 3]) {
        return `CodeForces / ${parts[problemsetIndex + 2]}${parts[problemsetIndex + 3]}`;
      }
      const contestIndex = parts.indexOf("contest");
      if (contestIndex >= 0 && parts[contestIndex + 1] && parts[contestIndex + 3]) {
        return `CodeForces / ${parts[contestIndex + 1]}${parts[contestIndex + 3]}`;
      }
      return `CodeForces / ${candidate.problem_id}`;
    }
    if (url.hostname.includes("atcoder.jp")) {
      const parts = url.pathname.split("/").filter(Boolean);
      const contest = parts[1]?.toUpperCase();
      const task = parts[3]?.replace(`${parts[1]}_`, "").toUpperCase();
      if (contest && task) return `AtCoder / ${contest} ${task}`;
      return `AtCoder / ${candidate.problem_id}`;
    }
    if (url.hostname.includes("luogu.com.cn")) {
      return `Luogu / ${url.pathname.split("/").filter(Boolean).pop() || candidate.problem_id}`;
    }
  } catch {
    return candidate.problem_id;
  }

  return candidate.problem_id;
}

function formatScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "off";
}

function scorePercent(value: number): string {
  if (!Number.isFinite(value)) return "0%";
  const clamped = Math.max(0, Math.min(1, value));
  return `${(clamped * 100).toFixed(1)}%`;
}

function renderScoreMeter(label: string, value: number | null, title: string): string {
  const numericValue = typeof value === "number" && Number.isFinite(value) ? value : null;
  const displayValue = numericValue === null ? "off" : numericValue.toFixed(3);
  const width = numericValue === null ? "0%" : scorePercent(numericValue);
  const aria = `${title} ${displayValue}`;
  return `
    <div class="score-meter ${numericValue === null ? "off" : ""}" title="${escapeHtml(aria)}" aria-label="${escapeHtml(aria)}">
      <span class="score-meter-label">${label}</span>
      <span class="score-meter-track" aria-hidden="true">
        <span class="score-meter-fill" style="width: ${width}"></span>
      </span>
    </div>`;
}

function renderScoreBlock(candidate: Candidate, scoreText: string): string {
  return `
    <div class="score-block">
      <strong>${scoreText}</strong>
      <div class="score-bars">
        ${renderScoreMeter("rr", candidate.rerank_score, "Rerank score")}
        ${renderScoreMeter("emb", candidate.embedding_score, "Embedding score")}
      </div>
    </div>`;
}

function showStages(state: AppState): boolean {
  return (
    state.isRunning ||
    state.hasSearched ||
    Boolean(state.rewrite) ||
    Boolean(state.error) ||
    stageOrder.some((name) => state.stages[name].state !== "idle")
  );
}

function renderSettings(state: AppState): string {
  if (!state.settingsOpen) return "";
  return `
    <div class="settings-panel" role="dialog" aria-label="Settings">
      <label class="switch-row">
        <span>Rewrite</span>
        <input id="useRewrite" type="checkbox" ${state.useRewrite ? "checked" : ""}>
      </label>
      <label class="switch-row">
        <span>Rerank</span>
        <input id="useRerank" type="checkbox" ${state.useRerank ? "checked" : ""}>
      </label>
      <label class="range-row">
        <span>Alpha <b>${state.alpha.toFixed(2)}</b></span>
        <input id="alphaInput" type="range" min="0" max="1" step="0.05" value="${state.alpha}">
      </label>
    </div>`;
}

function renderStatus(state: AppState): string {
  if (!showStages(state)) return "";
  return `
    <section class="status-shell" aria-label="Search progress">
      <div class="status-row">
        ${stageOrder
          .map((name) => {
            const stage = state.stages[name];
            const isRewrite = name === "rewrite";
            const hasRewrite = isRewrite && Boolean(state.rewrite);
            const detail = stage.state === "skip" ? stage.detail || "off" : formatElapsed(stage.elapsed);
            return `
              <div class="stage stage-${stage.state}">
                <span class="stage-dot" aria-hidden="true"></span>
                <span class="stage-label">${stageLabels[name]}</span>
                ${
                  hasRewrite
                    ? `<button id="rewriteToggle" class="stage-edit" type="button" aria-label="Edit rewrite" title="Edit rewrite">${icon("pencil")}</button>`
                    : ""
                }
                ${detail ? `<span class="stage-time">${escapeHtml(detail)}</span>` : ""}
              </div>`;
          })
          .join("")}
      </div>
      ${state.rewriteOpen && state.rewrite ? renderRewritePopover(state) : ""}
    </section>`;
}

function renderRewritePopover(state: AppState): string {
  return `
    <div class="rewrite-popover" role="dialog" aria-label="Rewrite edit">
      <button id="rewriteClose" class="rewrite-close" type="button" aria-label="Close rewrite editor" title="Close">
        ${icon("x")}
      </button>
      <label>
        <span>Statement</span>
        <textarea id="editStatement" rows="3">${escapeHtml(state.editStatement)}</textarea>
      </label>
      <label>
        <span>Abstract</span>
        <textarea id="editAbstract" rows="3">${escapeHtml(state.editAbstract)}</textarea>
      </label>
      <button id="resubmitRewrite" class="ghost-action" type="button" ${state.isRunning || !state.editStatement.trim() ? "disabled" : ""}>
        Resubmit
      </button>
    </div>`;
}

function renderToolbar(state: AppState): string {
  if (!state.candidates.length) return "";
  const total = state.candidates.length;
  const shown = Math.min(state.config.top_display, total);
  const rerankReady = hasRerankScores(state.candidates);
  return `
    <section class="result-toolbar" aria-label="Result controls">
      <div class="result-count"><b>${shown}</b> / ${total} results</div>
      <div class="sort-segment" role="radiogroup" aria-label="Sort">
        <span>Sort</span>
        ${(["combined", "embedding", "rerank"] as SortMode[])
          .map((mode) => {
            const disabled = mode === "rerank" && !rerankReady;
            return `<button class="${state.sortMode === mode ? "on" : ""}" data-sort="${mode}" role="radio" aria-checked="${state.sortMode === mode ? "true" : "false"}" type="button" ${disabled ? "disabled" : ""}>${sortLabels[mode]}</button>`;
          })
          .join("")}
      </div>
    </section>`;
}

function renderError(state: AppState): string {
  if (!state.error) return "";
  return `<div class="error-panel" role="alert">${escapeHtml(state.error)}</div>`;
}

function renderEmpty(state: AppState): string {
  if (state.hasSearched || state.isRunning) return "";
  return `<div class="empty-note">Paste a statement and search.</div>`;
}

function renderResults(state: AppState): string {
  if (state.isRunning && !state.candidates.length) return renderSkeleton();
  if (!state.candidates.length) return "";
  const candidates = sortedCandidates(state).slice(0, state.config.top_display);
  return `
    <section class="results" aria-label="Search results">
      ${candidates.map((candidate) => renderResultRow(candidate, state)).join("")}
    </section>`;
}

function renderSkeleton(): string {
  return `
    <section class="results skeletons" aria-label="Loading results">
      ${Array.from({ length: 5 })
        .map(
          () => `
        <article class="result-row skeleton-row">
          <div class="skeleton-main">
            <div class="skel skel-title"></div>
            <div class="skel skel-line"></div>
            <div class="skel skel-line short"></div>
          </div>
          <div class="skeleton-score">
            <div class="skel skel-score"></div>
            <div class="skel skel-mini"></div>
          </div>
        </article>`
        )
        .join("")}
    </section>`;
}

function renderResultRow(candidate: Candidate, state: AppState): string {
  const id = escapeHtml(candidate.problem_id);
  const expanded = state.expanded.has(candidate.problem_id);
  const title = escapeHtml(candidate.title || candidate.problem_id);
  const source = escapeHtml(sourceLabel(candidate));
  const score = candidateScore(candidate, state.sortMode, state.config.default_beta);
  const scoreText = formatScore(score);
  const statement = renderMathText(candidate.statement || candidate.original_text);
  const abstract = candidate.abstract ? renderMathText(candidate.abstract) : "";
  const titleLink = candidate.url
    ? `<a class="result-title" href="${escapeHtml(candidate.url)}" target="_blank" rel="noreferrer">${title}</a>`
    : `<span class="result-title">${title}</span>`;
  const sourceLink = candidate.url
    ? `<a class="result-source" href="${escapeHtml(candidate.url)}" target="_blank" rel="noreferrer">${icon("external-link", "source-icon")}<span>${source}</span></a>`
    : `<span class="result-source">${source}</span>`;
  const expandLabel = expanded ? "Collapse result" : "Expand result";
  const scoreBlock = renderScoreBlock(candidate, scoreText);

  return `
    <article class="result-row ${expanded ? "open" : ""}" data-id="${id}">
      <div class="result-main">
        <div class="result-content">
          <div class="title-line">
            <div class="title-links">
              ${titleLink}
              ${sourceLink}
            </div>
            <div class="score-inline">${scoreBlock}</div>
          </div>
          <div class="statement ${expanded ? "" : "clamped"}">${statement}</div>
          ${
            expanded && abstract
              ? `<div class="abstract"><span class="abstract-label">Abstract</span><div class="abstract-body">${abstract}</div></div>`
              : ""
          }
        </div>
        ${scoreBlock}
      </div>
      <button class="result-expand-strip" type="button" data-result-toggle="${id}" aria-expanded="${expanded ? "true" : "false"}" aria-label="${expandLabel}" title="${expandLabel}">
        ${icon(expanded ? "chevron-up" : "chevron-down")}
      </button>
    </article>`;
}

export function renderApp(root: HTMLElement, state: AppState, actions: Actions): void {
  const resultsMode = state.hasSearched || state.candidates.length > 0 || state.isRunning;
  root.innerHTML = `
    <div class="app">
      <header class="topbar">
        <div class="topbar-inner">
          <div class="brand">
            <span class="brand-mark" aria-hidden="true"></span>
            <span class="brand-name">Yuantiji</span>
          </div>
          <button id="settingsToggle" class="icon-button" type="button" aria-label="Settings">${icon("settings")}</button>
          ${renderSettings(state)}
        </div>
      </header>
      <main class="wrap ${resultsMode ? "with-results" : ""}">
        <form id="searchForm" class="querybox" autocomplete="off">
          <textarea id="queryInput" rows="1" spellcheck="false" placeholder="Paste problem statement here...">${escapeHtml(state.queryText)}</textarea>
          <button class="search-button" type="submit" ${state.isRunning || !state.queryText.trim() ? "disabled" : ""}>
            ${icon("search")}
            <span>${state.isRunning ? "Searching..." : "Search"}</span>
          </button>
        </form>
        ${renderStatus(state)}
        ${renderError(state)}
        ${renderToolbar(state)}
        ${renderResults(state)}
        ${renderEmpty(state)}
      </main>
    </div>`;

  renderIcons(root);
  installMathCopy(root);
  bind(root, actions);
  installResizeAutosize(root);
  requestAnimationFrame(() => autosizeTextareas(root));
}

function bind(root: HTMLElement, actions: Actions): void {
  const queryInput = root.querySelector<HTMLTextAreaElement>("#queryInput");
  queryInput?.addEventListener("input", () => {
    actions.setQuery(queryInput.value);
    autosize(queryInput);
  });
  queryInput?.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      actions.submit();
    }
  });

  root.querySelector<HTMLFormElement>("#searchForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    actions.submit();
  });

  root.querySelector<HTMLButtonElement>("#settingsToggle")?.addEventListener("click", actions.toggleSettings);
  root.querySelector<HTMLInputElement>("#useRewrite")?.addEventListener("change", (event) => {
    actions.setUseRewrite((event.currentTarget as HTMLInputElement).checked);
  });
  root.querySelector<HTMLInputElement>("#useRerank")?.addEventListener("change", (event) => {
    actions.setUseRerank((event.currentTarget as HTMLInputElement).checked);
  });
  root.querySelector<HTMLInputElement>("#alphaInput")?.addEventListener("input", (event) => {
    actions.setAlpha(Number((event.currentTarget as HTMLInputElement).value));
  });

  root.querySelector<HTMLButtonElement>("#rewriteToggle")?.addEventListener("click", (event) => {
    event.stopPropagation();
    actions.toggleRewrite();
  });
  root.querySelector<HTMLButtonElement>("#rewriteClose")?.addEventListener("click", actions.toggleRewrite);
  root.querySelector<HTMLTextAreaElement>("#editStatement")?.addEventListener("input", (event) => {
    actions.setEditStatement((event.currentTarget as HTMLTextAreaElement).value);
    autosize(event.currentTarget as HTMLTextAreaElement);
  });
  root.querySelector<HTMLTextAreaElement>("#editAbstract")?.addEventListener("input", (event) => {
    actions.setEditAbstract((event.currentTarget as HTMLTextAreaElement).value);
    autosize(event.currentTarget as HTMLTextAreaElement);
  });
  root.querySelector<HTMLButtonElement>("#resubmitRewrite")?.addEventListener("click", actions.resubmitRewrite);

  root.querySelectorAll<HTMLButtonElement>("[data-sort]").forEach((button) => {
    button.addEventListener("click", () => actions.setSortMode(button.dataset.sort as SortMode));
  });

  root.querySelectorAll<HTMLButtonElement>("[data-result-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      actions.toggleResult(button.dataset.resultToggle || "");
    });
  });
}

function autosizeTextareas(root: HTMLElement): void {
  root.querySelectorAll<HTMLTextAreaElement>("textarea").forEach(autosize);
}

let autosizeRoot: HTMLElement | null = null;
let autosizeResizeRaf = 0;
let autosizeResizeInstalled = false;

function installResizeAutosize(root: HTMLElement): void {
  autosizeRoot = root;
  if (autosizeResizeInstalled) return;
  autosizeResizeInstalled = true;

  const scheduleAutosize = () => {
    if (autosizeResizeRaf) cancelAnimationFrame(autosizeResizeRaf);
    autosizeResizeRaf = requestAnimationFrame(() => {
      autosizeResizeRaf = 0;
      if (autosizeRoot) autosizeTextareas(autosizeRoot);
    });
  };

  window.addEventListener("resize", scheduleAutosize);
  window.visualViewport?.addEventListener("resize", scheduleAutosize);
}

function autosize(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 280)}px`;
}
