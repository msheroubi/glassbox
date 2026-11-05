import type { AnalysisResult, AskResponse, HistoryPoint, InterventionType, ReplayResult, RuntimeStatus, SavedRun } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      // non-JSON error body: keep the status text
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  status: () => request<RuntimeStatus>("GET", "/api/status"),
  analyze: () => request<AnalysisResult>("POST", "/api/analyze"),
  analysis: (id: string) => request<AnalysisResult>("GET", `/api/analysis/${id}`),
  regenerateCounterfactual: (id: string) => request<{ ok: boolean }>("POST", `/api/analysis/${id}/counterfactual`),
  applyFix: (analysisId: string, override?: InterventionType) =>
    request<{ ok: boolean; intervention_type: InterventionType; message: string }>("POST", "/api/apply_fix", {
      analysis_id: analysisId,
      intervention_type: override ?? null,
    }),
  clearInterventions: () => request<RuntimeStatus>("POST", "/api/clear_interventions"),
  reset: () => request<RuntimeStatus>("POST", "/api/reset"),
  estop: () => request<RuntimeStatus>("POST", "/api/estop"),
  resume: () => request<RuntimeStatus>("POST", "/api/resume"),
  ask: (question: string) => request<AskResponse>("POST", "/api/ask", { question }),
  buffer: () => request<{ frames: HistoryPoint[] }>("GET", "/api/buffer"),
  hold: () => request<RuntimeStatus>("POST", "/api/hold"),
  settings: (patch: { self_correct?: boolean }) => request<RuntimeStatus>("POST", "/api/settings", patch),
  replays: () => request<{ runs: SavedRun[]; latest_id: string | null }>("GET", "/api/replays"),
  replay: (id: string) => request<ReplayResult>("GET", `/api/replay/${id}`),
  openReplay: (id: string) => request<ReplayResult>("POST", `/api/replay/${id}/open`),
  applyReplayFix: (id: string) => request<{ ok: boolean; message: string }>("POST", `/api/replay/${id}/apply`),
};

export const replayFrameUrl = (id: string, index: number, view: "policy" | "raw") =>
  `/api/replay/${id}/frame/${index}.jpg?view=${view}`;

export const bufferFrameUrl = (index: number, overlay = true) => `/api/buffer/${index}.jpg?overlay=${overlay}`;

export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}
