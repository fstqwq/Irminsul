import type { Candidate, Config, Cost, Health, RewritePayload, Stage } from "./state";

export type Timings = Partial<Record<Stage["name"], number | string>>;

export type StreamEvent =
  | ({ type: "stage" } & Stage)
  | ({ type: "rewrite" } & RewritePayload)
  | { type: "candidates"; candidates: Candidate[]; timings: Timings; cost?: Cost }
  | { type: "error"; message: string }
  | { type: "done" };

export type SearchRequest = {
  query_text: string;
  use_rewrite: boolean;
  use_rerank: boolean;
  beta: number;
  edited_clean?: string;
  edited_statement?: string;
  edited_abstract?: string;
  edited_abstract_zh?: string;
};

export async function fetchConfig(): Promise<Config> {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error(`Config failed (${response.status})`);
  return response.json() as Promise<Config>;
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(`Health failed (${response.status})`);
  return response.json() as Promise<Health>;
}

export async function streamSearch(
  request: SearchRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`Search failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line) as StreamEvent);
    }

    if (done) break;
  }

  if (buffer.trim()) {
    onEvent(JSON.parse(buffer) as StreamEvent);
  }
}
