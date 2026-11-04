import { useCallback, useEffect, useRef, useState } from "react";
import { api, wsUrl } from "./api";
import type { AnalysisResult, AnomalyEvent, FrameMeta, HistoryPoint, LogEntry, ReplayResult, RuntimeStatus } from "./types";

export interface LiveFrame {
  meta: FrameMeta;
  bitmap: ImageBitmap;
}

const HISTORY_LEN = 400;
const UI_REFRESH_MS = 400; // React re-renders (chart, HUD) at ~2.5 Hz; the canvas still draws every frame
const STATUS_POLL_MS = 2000;
const LOG_LEN = 150;

let logSeq = 0;

/**
 * One websocket to /ws/telemetry. Frames are decoded into an ImageBitmap held in a ref and drawn imperatively
 * by subscribers (no React render per frame); telemetry history and the latest frame meta are flushed into
 * React state at ~7 Hz.
 */
export function useTelemetry() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [frameMeta, setFrameMeta] = useState<FrameMeta | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyEvent[]>([]);
  const [analyses, setAnalyses] = useState<Record<string, AnalysisResult>>({});
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [replays, setReplays] = useState<Record<string, ReplayResult>>({});
  const [replayId, setReplayId] = useState<string | null>(null);
  const [replayStarted, setReplayStarted] = useState(0); // bumps on every new failure replay

  const frameRef = useRef<LiveFrame | null>(null);
  const listeners = useRef(new Set<() => void>());
  const wsRef = useRef<WebSocket | null>(null);
  const viewRef = useRef<"policy" | "raw">("policy");
  const historyRef = useRef<HistoryPoint[]>([]);
  const pendingMeta = useRef<FrameMeta | null>(null);
  const latestMeta = useRef<FrameMeta | null>(null);
  const dirty = useRef(false);
  const lastState = useRef<string | null>(null);

  const pushLog = useCallback((kind: LogEntry["kind"], text: string) => {
    setLog((l) => [{ id: ++logSeq, t: Date.now() / 1000, kind, text }, ...l].slice(0, LOG_LEN));
  }, []);

  const upsertAnalysis = useCallback((r: AnalysisResult, makeCurrent: boolean) => {
    setAnalyses((a) => ({ ...a, [r.analysis_id]: r }));
    if (makeCurrent) setCurrentId(r.analysis_id);
  }, []);

  const replaysRef = useRef<Record<string, ReplayResult>>({});
  const upsertReplay = useCallback((r: ReplayResult, makeCurrent: boolean) => {
    replaysRef.current = { ...replaysRef.current, [r.replay_id]: r };
    setReplays(replaysRef.current);
    if (makeCurrent) setReplayId(r.replay_id);
    else setReplayId((cur) => cur ?? r.replay_id);
  }, []);

  const subscribeFrame = useCallback((fn: () => void) => {
    listeners.current.add(fn);
    return () => {
      listeners.current.delete(fn);
    };
  }, []);

  const setView = useCallback((view: "policy" | "raw") => {
    viewRef.current = view;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ view }));
  }, []);

  const handleEvent = useCallback(
    (msg: Record<string, any>) => {
      switch (msg.type) {
        case "status":
          setStatus(msg.status as RuntimeStatus);
          setAnalyzing(Boolean(msg.status.analysis_running));
          break;
        case "state":
          setStatus((s) => (s ? { ...s, state: msg.state, status: msg.status } : s));
          if (msg.state !== lastState.current) {
            lastState.current = msg.state;
            pushLog("info", `${msg.state}: ${msg.status}`);
          }
          break;
        case "anomaly": {
          const ev = msg.event as AnomalyEvent;
          setAnomalies((a) => [...a, ev].slice(-20));
          pushLog("anomaly", `Anomaly: ${ev.reasons.join("; ")}`);
          break;
        }
        case "analysis_started":
          setAnalyzing(true);
          pushLog("analysis", `Failure analysis started (${msg.trigger})`);
          break;
        case "analysis": {
          const r = msg.result as AnalysisResult;
          upsertAnalysis(r, true);
          setStatus((s) => (s ? { ...s, latest_analysis_id: r.analysis_id, analysis_running: false } : s));
          setAnalyzing(false);
          pushLog(
            "analysis",
            `${r.diagnosis.failure_type} (${Math.round(r.diagnosis.confidence * 100)}%, ${r.diagnosis_source}): ${r.diagnosis.headline_summary}`,
          );
          break;
        }
        case "analysis_update": {
          const r = msg.result as AnalysisResult;
          upsertAnalysis(r, false);
          break;
        }
        case "fix_applied":
          pushLog("fix", `Fix applied (${msg.intervention_type}): ${msg.message}`);
          break;
        case "replay_started": {
          const r = msg.replay as ReplayResult;
          upsertReplay(r, true);
          setReplayStarted((n) => n + 1);
          pushLog("analysis", `Recorded ${r.n_frames} frames (${(r.n_frames / r.fps).toFixed(1)} s); offline replay analysis started`);
          break;
        }
        case "replay_update": {
          const r = msg.replay as ReplayResult;
          const prev = replaysRef.current[r.replay_id];
          upsertReplay(r, false);
          if (prev && !prev.breakdown && r.breakdown) {
            pushLog("analysis", `Muse Wonder (${r.breakdown_source}): ${r.breakdown.failure_stage}: ${r.breakdown.headline}`);
          }
          if (prev && prev.self_correction !== r.self_correction && r.self_correction !== "off") {
            pushLog(r.self_correction === "applied" ? "fix" : "info", `Self-correction ${r.self_correction}: ${r.self_correction_note}`);
          }
          break;
        }
        case "replay_progress": {
          const r = replaysRef.current[msg.replay_id];
          if (r) upsertReplay({ ...r, status: msg.status, progress: msg.progress, progress_text: msg.text }, false);
          break;
        }
        case "staging":
          setStatus((s) => (s ? { ...s, staging: msg.staging } : s));
          if (msg.staging) pushLog("info", `Demo staging: ${msg.staging.label} moved onto the goal in the policy's view`);
          break;
        case "settings":
          setStatus((s) => (s ? { ...s, self_correct: Boolean(msg.self_correct) } : s));
          pushLog("info", `Self-correction ${msg.self_correct ? "enabled" : "disabled"}`);
          break;
        case "reset":
          historyRef.current = [];
          setHistory([]);
          setAnomalies([]);
          setCurrentId(null);
          setStatus((s) => (s ? { ...s, latest_analysis_id: null } : s));
          pushLog("info", `Scenario reset${msg.scenario ? `: ${msg.scenario}` : ""}`);
          break;
        default:
          break;
      }
    },
    [pushLog, upsertAnalysis, upsertReplay],
  );

  // Websocket with automatic reconnect.
  useEffect(() => {
    let closed = false;
    let retry: number | undefined;

    const connect = () => {
      const ws = new WebSocket(wsUrl("/ws/telemetry"));
      ws.binaryType = "blob";
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        ws.send(JSON.stringify({ view: viewRef.current }));
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 1000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          if (msg.type === "frame") {
            const meta = msg as FrameMeta;
            pendingMeta.current = meta;
            latestMeta.current = meta;
            const tel = meta.telemetry;
            const h = historyRef.current;
            if (h.length && meta.index < h[h.length - 1].index) h.length = 0; // runtime restarted
            h.push({
              index: meta.index,
              t: meta.t,
              joint_error: tel.joint_error,
              chunk_variance: tel.action_chunk_variance,
              max_load: Math.max(...tel.torque_load.slice(0, 5).map(Math.abs)),
            });
            if (h.length > HISTORY_LEN) h.splice(0, h.length - HISTORY_LEN);
            dirty.current = true;
          } else {
            handleEvent(msg);
          }
          return;
        }
        const meta = pendingMeta.current;
        pendingMeta.current = null;
        if (!meta || document.hidden) return; // background tab: skip JPEG decode entirely
        createImageBitmap(ev.data as Blob)
          .then((bitmap) => {
            const prev = frameRef.current;
            if (prev && prev.meta.index > meta.index) {
              bitmap.close(); // decoded out of order
              return;
            }
            prev?.bitmap.close();
            frameRef.current = { meta, bitmap };
            listeners.current.forEach((fn) => fn());
          })
          .catch(() => undefined);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [handleEvent]);

  // Throttled flush of per-frame data into React state.
  useEffect(() => {
    const id = window.setInterval(() => {
      if (!dirty.current || document.hidden) return;
      dirty.current = false;
      const meta = latestMeta.current;
      setHistory(historyRef.current.slice());
      if (meta) {
        setFrameMeta(meta);
        setStatus((s) =>
          s
            ? {
                ...s,
                state: meta.state,
                status: meta.status,
                phase: meta.phase,
                fps: meta.fps,
                scenario: meta.scenario,
                grasp_offset_mm: meta.grasp_offset_mm,
                suppression: meta.suppression.objects,
                recovery_active: meta.recovery.active,
              }
            : s,
        );
      }
    }, UI_REFRESH_MS);
    return () => window.clearInterval(id);
  }, []);

  // Slow status poll catches fields the stream does not carry (analysis_running, latest id, ...).
  useEffect(() => {
    const poll = () =>
      api
        .status()
        .then((s) => {
          setStatus(s);
          if (!s.analysis_running) setAnalyzing((a) => (a ? false : a));
        })
        .catch(() => undefined);
    poll();
    const id = window.setInterval(poll, STATUS_POLL_MS);
    return () => window.clearInterval(id);
  }, []);

  return {
    connected,
    status,
    setStatus,
    frameMeta,
    history,
    anomalies,
    analysis: currentId ? analyses[currentId] ?? null : null,
    upsertAnalysis,
    analyzing,
    setAnalyzing,
    log,
    pushLog,
    frameRef,
    subscribeFrame,
    setView,
    replays,
    replay: replayId ? replays[replayId] ?? null : null,
    upsertReplay,
    replayStarted,
  };
}

export type Telemetry = ReturnType<typeof useTelemetry>;
