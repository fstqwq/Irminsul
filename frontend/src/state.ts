export type StageName = "rewrite" | "embed" | "search" | "rerank";
export type StageState = "idle" | "active" | "done" | "skip" | "error";
export type SortMode = "combined" | "embedding" | "rerank";
export type ResultView = "clean" | "statement" | "abstract" | "abstract_zh";

export type Stage = {
  name: StageName;
  state: StageState;
  elapsed?: number;
  detail?: string;
};

export type Config = {
  top_retrieval: number;
  top_display: number;
  default_beta: number;
  default_rerank: boolean;
};

export type Health = {
  ok: boolean;
  loaded_index_key: string | null;
  problem_count: number;
  embedding_shape: [number, number] | null;
  views: string[];
  switching: boolean;
  source_counts: SourceCount[];
};

export type SourceCount = {
  source: string;
  count: number;
};

export type Cost = {
  microusd: number;
};

export type RewritePayload = {
  clean: string;
  statement: string;
  abstract: string;
  abstract_zh: string;
  raw: string;
  edited?: boolean;
};

export type Candidate = {
  problem_id: string;
  title: string;
  url: string;
  clean: string;
  statement: string;
  abstract: string;
  abstract_zh: string;
  embedding_score: number;
  rerank_score: number | null;
  final_score: number | null;
};

export type AppState = {
  config: Config;
  activeProblemCount: number | null;
  sourceCounts: SourceCount[];
  queryText: string;
  useRewrite: boolean;
  useRerank: boolean;
  sortMode: SortMode;
  resultView: ResultView;
  settingsOpen: boolean;
  rewriteOpen: boolean;
  rewrite: RewritePayload | null;
  editRewrite: RewritePayload;
  stages: Record<StageName, Stage>;
  candidates: Candidate[];
  cost: Cost | null;
  error: string;
  isRunning: boolean;
  hasSearched: boolean;
};

export const stageOrder: StageName[] = ["rewrite", "embed", "search", "rerank"];

export const stageLabels: Record<StageName, string> = {
  rewrite: "Rewrite",
  embed: "Embed",
  search: "Search",
  rerank: "Rerank"
};

export const sortLabels: Record<SortMode, string> = {
  combined: "Combined",
  embedding: "Embedding",
  rerank: "Rerank"
};

export const resultViewLabels: Record<ResultView, string> = {
  clean: "Filtered",
  statement: "Statement",
  abstract: "Abstract",
  abstract_zh: "中文"
};

export const defaultConfig: Config = {
  top_retrieval: 200,
  top_display: 20,
  default_beta: 0.75,
  default_rerank: true
};

export const sampleQuery = "Given a bipartite graph, find the maximum matching.";

const resultViewStorageKey = "yuantiji.resultView";

export function loadResultView(): ResultView {
  let value: string | null = null;
  try {
    value = window.localStorage?.getItem(resultViewStorageKey) || null;
  } catch {
    value = null;
  }
  if (value === "clean" || value === "statement" || value === "abstract" || value === "abstract_zh") {
    return value;
  }
  return "statement";
}

export function saveResultView(value: ResultView): void {
  try {
    window.localStorage?.setItem(resultViewStorageKey, value);
  } catch {
    return;
  }
}

export function initialStages(): Record<StageName, Stage> {
  return {
    rewrite: { name: "rewrite", state: "idle" },
    embed: { name: "embed", state: "idle" },
    search: { name: "search", state: "idle" },
    rerank: { name: "rerank", state: "idle" }
  };
}

export function createInitialState(): AppState {
  return {
    config: defaultConfig,
    activeProblemCount: null,
    sourceCounts: [],
    queryText: sampleQuery,
    useRewrite: true,
    useRerank: defaultConfig.default_rerank,
    sortMode: "combined",
    resultView: loadResultView(),
    settingsOpen: false,
    rewriteOpen: false,
    rewrite: null,
    editRewrite: {
      clean: "",
      statement: "",
      abstract: "",
      abstract_zh: "",
      raw: ""
    },
    stages: initialStages(),
    candidates: [],
    cost: null,
    error: "",
    isRunning: false,
    hasSearched: false
  };
}

export function hasRerankScores(candidates: Candidate[]): boolean {
  return candidates.some((candidate) => typeof candidate.rerank_score === "number");
}

export function combinedScore(candidate: Candidate, beta: number): number {
  if (typeof candidate.rerank_score !== "number") return candidate.embedding_score;
  return beta * candidate.rerank_score + (1 - beta) * candidate.embedding_score;
}

export function candidateScore(candidate: Candidate, mode: SortMode, beta: number): number {
  if (mode === "embedding") return candidate.embedding_score;
  if (mode === "rerank") return typeof candidate.rerank_score === "number" ? candidate.rerank_score : -Infinity;
  return combinedScore(candidate, beta);
}

export function sortedCandidates(state: AppState): Candidate[] {
  const mode = state.sortMode === "rerank" && !hasRerankScores(state.candidates) ? "combined" : state.sortMode;
  return [...state.candidates]
    .map((candidate) => ({
      ...candidate,
      final_score: candidateScore(candidate, mode, state.config.default_beta)
    }))
    .sort((a, b) => {
      const diff =
        candidateScore(b, mode, state.config.default_beta) -
        candidateScore(a, mode, state.config.default_beta);
      if (Number.isFinite(diff) && diff !== 0) return diff;
      return (a.title || a.problem_id).localeCompare(b.title || b.problem_id);
    });
}
