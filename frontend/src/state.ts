export type StageName = "rewrite" | "embed" | "search" | "rerank";
export type StageState = "idle" | "active" | "done" | "skip" | "error";
export type SortMode = "combined" | "embedding" | "rerank";

export type Stage = {
  name: StageName;
  state: StageState;
  elapsed?: number;
  detail?: string;
};

export type Config = {
  top_retrieval: number;
  top_display: number;
  default_alpha: number;
  default_beta: number;
  default_rerank: boolean;
};

export type RewritePayload = {
  statement: string;
  abstract: string;
  raw: string;
  edited?: boolean;
};

export type Candidate = {
  problem_id: string;
  title: string;
  url: string;
  original_text: string;
  statement: string;
  abstract: string;
  embedding_score: number;
  rerank_score: number | null;
  final_score: number | null;
};

export type AppState = {
  config: Config;
  queryText: string;
  useRewrite: boolean;
  useRerank: boolean;
  alpha: number;
  sortMode: SortMode;
  settingsOpen: boolean;
  rewriteOpen: boolean;
  rewrite: RewritePayload | null;
  editStatement: string;
  editAbstract: string;
  stages: Record<StageName, Stage>;
  candidates: Candidate[];
  error: string;
  isRunning: boolean;
  hasSearched: boolean;
  expanded: Set<string>;
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

export const defaultConfig: Config = {
  top_retrieval: 200,
  top_display: 20,
  default_alpha: 0.5,
  default_beta: 0.75,
  default_rerank: true
};

export const sampleQuery = "Given a bipartite graph, find the maximum matching.";

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
    queryText: sampleQuery,
    useRewrite: true,
    useRerank: defaultConfig.default_rerank,
    alpha: defaultConfig.default_alpha,
    sortMode: "combined",
    settingsOpen: false,
    rewriteOpen: false,
    rewrite: null,
    editStatement: "",
    editAbstract: "",
    stages: initialStages(),
    candidates: [],
    error: "",
    isRunning: false,
    hasSearched: false,
    expanded: new Set<string>()
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
