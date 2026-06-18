import { createIcons, ExternalLink, Pencil, Search, Settings, X } from "lucide";
import { escapeHtml, renderMathText } from "./math";
import {
  AppState,
  Candidate,
  ResultView,
  SortMode,
  candidateScore,
  hasRerankScores,
  resultViewLabels,
  sortedCandidates,
  sortLabels,
  stageLabels,
  stageOrder
} from "./state";
const icons = {
  ExternalLink,
  Pencil,
  Search,
  Settings,
  X
};

type IconName = "external-link" | "pencil" | "search" | "settings" | "x";

function icon(name: IconName, className = ""): string {
  const classAttr = className ? ` class="${className}"` : "";
  return `<i data-lucide="${name}"${classAttr} aria-hidden="true"></i>`;
}

function renderIcons(root: HTMLElement): void {
  createIcons({ root, icons });
}

let activeCopyRoot: HTMLElement | null = null;
let mathCopyInstalled = false;

function installMathCopy(root: HTMLElement): void {
  activeCopyRoot = root;
  if (mathCopyInstalled) return;
  mathCopyInstalled = true;
  document.addEventListener("copy", handleCopy);
}

function handleCopy(event: ClipboardEvent): void {
  if (!activeCopyRoot || !event.clipboardData) return;
  const selection = document.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  const scopes = selectedElements(activeCopyRoot, range, ".statement");
  if (!scopes.length || !selectedElements(activeCopyRoot, range, ".math-fragment").length) return;
  const text = normalizeCopiedText(scopes.map((scope) => textFromNode(scope, range)).join("\n"));
  if (!text) return;
  event.preventDefault();
  event.clipboardData.setData("text/plain", text);
}

function selectedElements(root: HTMLElement, range: Range, selector: string): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(selector)).filter((element) => intersects(range, element));
}

function intersects(range: Range, node: Node): boolean {
  try {
    return range.intersectsNode(node);
  } catch {
    return false;
  }
}

function textFromNode(node: Node, range: Range): string {
  if (!intersects(range, node)) return "";
  if (node.nodeType === Node.TEXT_NODE) return textFromTextNode(node as Text, range);
  if (node.nodeType !== Node.ELEMENT_NODE) return "";
  const element = node as HTMLElement;
  if (element.classList.contains("math-fragment")) return textFromMath(element);
  if (element.tagName === "BR") return "\n";
  return Array.from(element.childNodes).map((child) => textFromNode(child, range)).join("");
}

function textFromTextNode(node: Text, range: Range): string {
  let start = 0;
  let end = node.data.length;
  if (range.startContainer === node) start = range.startOffset;
  if (range.endContainer === node) end = range.endOffset;
  return node.data.slice(start, end);
}

function textFromMath(element: HTMLElement): string {
  const tex = element.dataset.tex?.trim() || element.textContent?.trim() || "";
  if (!tex) return "";
  if (element.dataset.display === "true") return `\n$$${tex}$$\n`;
  return tex;
}

function normalizeCopiedText(value: string): string {
  return value
    .replace(/\u00a0/g, " ")
    .replace(/[ \t\f\v]+\n/g, "\n")
    .replace(/\n[ \t\f\v]+/g, "\n")
    .replace(/[ \t\f\v]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export type Actions = {
  setQuery(value: string): void;
  submit(): void;
  toggleSettings(): void;
  setUseRewrite(value: boolean): void;
  setUseRerank(value: boolean): void;
  setSortMode(value: SortMode): void;
  setResultView(value: ResultView): void;
  toggleRewrite(): void;
  setEditRewrite(view: ResultView, value: string): void;
  resubmitRewrite(): void;
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

function formatCost(state: AppState): string {
  const microusd = state.cost?.microusd;
  if (typeof microusd !== "number" || !Number.isFinite(microusd)) return "";
  const usd = microusd / 1_000_000;
  if (state.resultView === "abstract_zh") {
    return `￥${(usd * 7).toFixed(6)}`;
  }
  return `$${usd.toFixed(6)}`;
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
  const editViews: ResultView[] = ["clean", "statement", "abstract", "abstract_zh"];
  return `
    <div class="rewrite-popover" role="dialog" aria-label="Rewrite edit">
      <button id="rewriteClose" class="rewrite-close" type="button" aria-label="Close rewrite editor" title="Close">
        ${icon("x")}
      </button>
      ${editViews
        .map(
          (view) => `
            <label>
              <span>${escapeHtml(resultViewLabels[view])}</span>
              <textarea data-edit-rewrite="${view}" rows="3">${escapeHtml(state.editRewrite[view])}</textarea>
            </label>`
        )
        .join("")}
      <button id="resubmitRewrite" class="ghost-action" type="button" ${state.isRunning || !state.editRewrite.statement.trim() ? "disabled" : ""}>
        Resubmit
      </button>
    </div>`;
}

function renderToolbar(state: AppState): string {
  if (!state.candidates.length) return "";
  const total = state.candidates.length;
  const shown = Math.min(state.config.top_display, total);
  const rerankReady = hasRerankScores(state.candidates);
  const cost = formatCost(state);
  return `
    <section class="result-toolbar" aria-label="Result controls">
      <div class="result-count">
        <span><b>${shown}</b> / ${total} results</span>
        ${cost ? `<span class="result-cost" title="Estimated request cost">${escapeHtml(cost)}</span>` : ""}
      </div>
      <div class="toolbar-controls">
        <label class="toolbar-select">
          <span>Sort</span>
          <select id="sortSelect" aria-label="Sort">
            ${(["combined", "embedding", "rerank"] as SortMode[])
              .map((mode) => {
                const disabled = mode === "rerank" && !rerankReady;
                return `<option value="${mode}" ${state.sortMode === mode ? "selected" : ""} ${disabled ? "disabled" : ""}>${sortLabels[mode]}</option>`;
              })
              .join("")}
          </select>
        </label>
        <label class="toolbar-select">
          <span>View</span>
          <select id="resultViewSelect" aria-label="Result view">
            ${(["clean", "statement", "abstract", "abstract_zh"] as ResultView[])
              .map(
                (view) =>
                  `<option value="${view}" ${state.resultView === view ? "selected" : ""}>${resultViewLabels[view]}</option>`
              )
              .join("")}
          </select>
        </label>
      </div>
    </section>`;
}

function renderError(state: AppState): string {
  if (!state.error) return "";
  return `<div class="error-panel" role="alert">${escapeHtml(state.error)}</div>`;
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

function renderFooter(state: AppState): string {
  if (typeof state.activeProblemCount !== "number") return "";
  const hasSourceCounts = state.sourceCounts.length > 0;
  return `
    <footer class="app-footer">
      <div>
        Active index: <span>${state.activeProblemCount.toLocaleString()}</span> problems${
          hasSourceCounts ? ` <a id="sourceCountsToggle" href="#sourceCountsPopover">(view)</a>` : ""
        }
      </div>
      ${hasSourceCounts ? `
        <div id="sourceCountsPopover" class="settings-panel" popover>
          ${state.sourceCounts
            .map(
              (item) => `
                <div class="switch-row">
                  <span>${escapeHtml(item.source)}</span>
                  <span>${item.count.toLocaleString()}</span>
                </div>`
            )
            .join("")}
        </div>` : ""}
      <div>Your input will be retained for audition.</div>
    </footer>`;
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
  const title = escapeHtml(candidate.title || candidate.problem_id);
  const source = escapeHtml(sourceLabel(candidate));
  const score = candidateScore(candidate, state.sortMode, state.config.beta);
  const scoreText = formatScore(score);
  const viewText = candidate[state.resultView] || candidate.statement || candidate.clean;
  const statement = renderMathText(viewText);
  const titleLink = candidate.url
    ? `<a class="result-title" href="${escapeHtml(candidate.url)}" target="_blank" rel="noreferrer">${title}</a>`
    : `<span class="result-title">${title}</span>`;
  const sourceLink = candidate.url
    ? `<a class="result-source" href="${escapeHtml(candidate.url)}" target="_blank" rel="noreferrer">${icon("external-link", "source-icon")}<span>${source}</span></a>`
    : `<span class="result-source">${source}</span>`;
  const scoreBlock = renderScoreBlock(candidate, scoreText);

  return `
    <article class="result-row" data-id="${id}">
      <div class="result-main">
        <div class="result-content">
          <div class="title-line">
            <div class="title-links">
              ${titleLink}
              ${sourceLink}
            </div>
            <div class="score-inline">${scoreBlock}</div>
          </div>
          <div class="statement clamped">${statement}</div>
        </div>
        ${scoreBlock}
      </div>
    </article>`;
}

export function renderApp(root: HTMLElement, state: AppState, actions: Actions): void {
  const resultsMode = state.hasSearched || state.candidates.length > 0 || state.isRunning;
  root.innerHTML = `
    <div class="app">
      <header class="topbar">
        <div class="topbar-inner">
          <div class="brand">
            <img class="brand-mark" src="/irminsul.png" alt="" aria-hidden="true">
            <span class="brand-name">Irminsul</span>
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
        ${renderFooter(state)}
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
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      actions.submit();
    }
  });

  root.querySelector<HTMLFormElement>("#searchForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    actions.submit();
  });

  root.querySelector<HTMLButtonElement>("#settingsToggle")?.addEventListener("click", actions.toggleSettings);
  root.querySelector<HTMLAnchorElement>("#sourceCountsToggle")?.addEventListener("click", (event) => {
    event.preventDefault();
    const popover = root.querySelector<HTMLElement>("#sourceCountsPopover");
    popover?.togglePopover();
  });
  root.querySelector<HTMLInputElement>("#useRewrite")?.addEventListener("change", (event) => {
    actions.setUseRewrite((event.currentTarget as HTMLInputElement).checked);
  });
  root.querySelector<HTMLInputElement>("#useRerank")?.addEventListener("change", (event) => {
    actions.setUseRerank((event.currentTarget as HTMLInputElement).checked);
  });

  root.querySelector<HTMLButtonElement>("#rewriteToggle")?.addEventListener("click", (event) => {
    event.stopPropagation();
    actions.toggleRewrite();
  });
  root.querySelector<HTMLButtonElement>("#rewriteClose")?.addEventListener("click", actions.toggleRewrite);
  root.querySelectorAll<HTMLTextAreaElement>("[data-edit-rewrite]").forEach((textarea) => {
    textarea.addEventListener("input", (event) => {
      const target = event.currentTarget as HTMLTextAreaElement;
      actions.setEditRewrite(target.dataset.editRewrite as ResultView, target.value);
      autosize(target);
    });
  });
  root.querySelector<HTMLButtonElement>("#resubmitRewrite")?.addEventListener("click", actions.resubmitRewrite);

  root.querySelector<HTMLSelectElement>("#sortSelect")?.addEventListener("change", (event) => {
    actions.setSortMode((event.currentTarget as HTMLSelectElement).value as SortMode);
  });

  root.querySelector<HTMLSelectElement>("#resultViewSelect")?.addEventListener("change", (event) => {
    actions.setResultView((event.currentTarget as HTMLSelectElement).value as ResultView);
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
