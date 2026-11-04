import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import AttentionTimeline from "./components/AttentionTimeline";
import MuseXAIInspector, { type ChatMessage } from "./components/MuseXAIInspector";
import OverheadCanvas from "./components/OverheadCanvas";
import ReplayBreakdown from "./components/ReplayBreakdown";
import ReplayPlayer from "./components/ReplayPlayer";
import TelemetryScrubber from "./components/TelemetryScrubber";
import { MAP_ORDER, frameAtTime } from "./replayMath";
import type { HistoryPoint, InterventionType, SavedRun } from "./types";
import { useTelemetry } from "./useTelemetry";
import { useVoice, type VoiceReply } from "./useVoice";

let chatSeq = 0;

export default function App() {
  const tel = useTelemetry();
  const { status, analysis, pushLog, setStatus, upsertAnalysis, setAnalyzing, replay, upsertReplay } = tel;
  const [tab, setTab] = useState<"live" | "replay">(() => (window.location.hash === "#replay" ? "replay" : "live"));
  const [replayPos, setReplayPos] = useState(0);
  const [mapKind, setMapKind] = useState<string | null>(null);
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);
  const [applyingReplay, setApplyingReplay] = useState(false);
  const [scrub, setScrub] = useState<{ frames: HistoryPoint[]; pos: number } | null>(null);
  const [applying, setApplying] = useState(false);
  const [asking, setAsking] = useState(false);
  const [chat, setChat] = useState<ChatMessage[]>([]);

  const say = useCallback((role: ChatMessage["role"], text: string, source?: string) => {
    setChat((c) => [...c, { id: ++chatSeq, role, text, source }].slice(-40));
  }, []);

  const fail = useCallback(
    (what: string, e: unknown) => {
      const msg = e instanceof ApiError ? `${what}: ${e.message} (HTTP ${e.status})` : `${what}: ${(e as Error).message}`;
      pushLog("error", msg);
    },
    [pushLog],
  );

  const voice = useVoice(
    useCallback(
      (r: VoiceReply) => {
        switch (r.type) {
          case "transcript":
            if (r.final && r.text.trim()) say("user", `🎙 ${r.text.trim()}`, "Muse Voice");
            break;
          case "command":
            say("system", `${r.command.replace("_", " ")}: ${r.message}`);
            pushLog("voice", `Voice command "${r.text}" → ${r.command}`);
            break;
          case "answer":
            say("assistant", r.answer, r.source);
            break;
          case "error":
            say("system", `⚠ ${r.message}`);
            pushLog("error", r.message);
            break;
        }
      },
      [say, pushLog],
    ),
  );

  const enterScrub = useCallback(
    async (t?: number) => {
      try {
        const { frames } = await api.buffer();
        if (!frames.length) return;
        let pos = frames.length - 1;
        if (t !== undefined) {
          frames.forEach((f, i) => {
            if (Math.abs(f.t - t) < Math.abs(frames[pos].t - t)) pos = i;
          });
        }
        setScrub({ frames, pos });
      } catch (e) {
        fail("Rewind failed", e);
      }
    },
    [fail],
  );

  const onAnalyze = useCallback(async () => {
    setAnalyzing(true);
    try {
      const r = await api.analyze();
      upsertAnalysis(r, true);
    } catch (e) {
      fail("Analysis failed", e);
    } finally {
      setAnalyzing(false);
    }
  }, [fail, setAnalyzing, upsertAnalysis]);

  const onApplyFix = useCallback(
    async (override?: InterventionType) => {
      if (!analysis) return;
      setApplying(true);
      setScrub(null);
      try {
        await api.applyFix(analysis.analysis_id, override);
        setStatus(await api.status());
      } catch (e) {
        fail("Apply fix failed", e);
      } finally {
        setApplying(false);
      }
    },
    [analysis, fail, setStatus],
  );

  const statusAction = useCallback(
    (what: string, fn: () => Promise<typeof status>) => async () => {
      try {
        const s = await fn();
        if (s) setStatus(s);
      } catch (e) {
        fail(what, e);
      }
    },
    [fail, setStatus],
  );

  const onReset = useCallback(
    (scenario?: string) => {
      setScrub(null);
      void statusAction("Reset failed", () => api.reset(scenario))();
    },
    [statusAction],
  );

  const onAsk = useCallback(
    async (q: string) => {
      say("user", q);
      setAsking(true);
      try {
        const r = await api.ask(q);
        say("assistant", r.answer, r.source);
      } catch (e) {
        fail("Ask failed", e);
        say("system", "⚠ Could not reach the backend.");
      } finally {
        setAsking(false);
      }
    },
    [fail, say],
  );

  const onRegen = useCallback(async () => {
    if (!analysis) return;
    try {
      await api.regenerateCounterfactual(analysis.analysis_id);
      pushLog("info", "Re-rendering the Muse Image counterfactual…");
    } catch (e) {
      fail("Counterfactual failed", e);
    }
  }, [analysis, fail, pushLog]);

  // A new failure replay: switch to it and play it from the start.
  useEffect(() => {
    if (tel.replayStarted > 0) {
      setTab("replay");
      setReplayPos(0);
    }
  }, [tel.replayStarted]);

  // Default to the most informative map that has data (SmolVLA when it is the lens, else ACT, else SAM 3).
  useEffect(() => {
    if (!replay) return;
    const have = (k: string | null) => replay.maps.some((m) => m.kind === k && m.keyframes.length > 0);
    if (!have(mapKind)) {
      const best = MAP_ORDER.find(have) ?? replay.maps.find((m) => m.keyframes.length)?.kind ?? null;
      if (best !== mapKind) setMapKind(best);
    }
  }, [replay, mapKind]);

  const refreshRuns = useCallback(() => {
    api
      .replays()
      .then((r) => setSavedRuns(r.runs))
      .catch(() => undefined);
  }, []);
  useEffect(() => {
    if (tab === "replay") refreshRuns();
  }, [tab, replay?.replay_id, refreshRuns]);

  const openRun = useCallback(
    async (id: string) => {
      try {
        upsertReplay(await api.openReplay(id), true);
        setReplayPos(0);
      } catch (e) {
        fail("Opening the run failed", e);
      }
    },
    [fail, upsertReplay],
  );

  const onApplyReplay = useCallback(async () => {
    if (!replay) return;
    setApplyingReplay(true);
    try {
      await api.applyReplayFix(replay.replay_id);
      setStatus(await api.status());
    } catch (e) {
      fail("Self-correction failed", e);
    } finally {
      setApplyingReplay(false);
    }
  }, [replay, fail, setStatus]);

  const seekTime = useCallback(
    (t: number) => {
      if (!replay) return;
      setTab("replay");
      setReplayPos(frameAtTime(replay, t));
    },
    [replay],
  );

  const onSelfCorrect = useCallback(
    async (on: boolean) => {
      try {
        setStatus(await api.settings({ self_correct: on }));
      } catch (e) {
        fail("Settings failed", e);
      }
    },
    [fail, setStatus],
  );

  const replayBusy = replay && replay.status !== "done" && replay.status !== "failed";
  const models = status?.models;
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◧</span>
          <div>
            <h1>GlassBox-SO101</h1>
            <span className="muted small">explainable failure analysis &amp; recovery for a LeRobot SO-101 policy</span>
          </div>
        </div>
        <span className="spacer" />
        {models && (
          <span className="muted small models">
            {models.sam} · {models.spark} · {models.wonder} · {models.image} · {models.voice}
          </span>
        )}
        <span className={`conn ${tel.connected ? "up" : "down"}`}>{tel.connected ? "● connected" : "○ offline"}</span>
      </header>
      <main className="split">
        <div className="left">
          <nav className="tabs">
            <button className={tab === "live" ? "on" : ""} onClick={() => setTab("live")}>
              <span className={`dot ${status?.state === "RUNNING" ? "run" : ""}`} /> Live
            </button>
            <button className={tab === "replay" ? "on" : ""} onClick={() => setTab("replay")} disabled={!status?.replay?.enabled}>
              ⏪ Failure replay
              {replayBusy && <span className="spinner tiny" />}
              {replay?.breakdown && (
                <span className={`stage-pill ${replay.breakdown.failure_stage.toLowerCase()}`}>
                  {replay.breakdown.failure_stage}
                </span>
              )}
            </button>
            <span className="spacer" />
            <span className="muted small tab-hint">
              {tab === "live"
                ? "ACT runs live on the arm; when it fails, the run is saved and replayed offline"
                : "Space play/pause · ←/→ step · Shift+←/→ keyframes · click the timeline to seek"}
            </span>
          </nav>
          {tab === "replay" ? (
            <>
              <ReplayPlayer
                replay={replay}
                pos={replayPos}
                onPos={setReplayPos}
                mapKind={mapKind}
                onMapKind={setMapKind}
                autoplayKey={tel.replayStarted}
                savedRuns={savedRuns}
                onOpenRun={openRun}
                onRefreshRuns={refreshRuns}
              />
              <AttentionTimeline
                replay={replay}
                mapKind={mapKind}
                pos={replayPos}
                onSeek={setReplayPos}
                culprit={replay?.breakdown?.culprit_label ?? null}
              />
            </>
          ) : (
            <>
              <OverheadCanvas
                frameRef={tel.frameRef}
                subscribeFrame={tel.subscribeFrame}
                setView={tel.setView}
                status={status}
                analysis={analysis}
                scrubIndex={scrub ? (scrub.frames[scrub.pos]?.index ?? null) : null}
                connected={tel.connected}
              />
              <TelemetryScrubber
                history={tel.history}
                anomalies={tel.anomalies}
                status={status}
                scrub={scrub}
                onEnterScrub={enterScrub}
                onScrubPos={(pos) => setScrub((s) => (s ? { ...s, pos } : s))}
                onLive={() => setScrub(null)}
              />
            </>
          )}
        </div>
        <aside className="right">
          <MuseXAIInspector
            connected={tel.connected}
            status={status}
            analysis={analysis}
            analyzing={tel.analyzing}
            applying={applying}
            log={tel.log}
            chat={chat}
            asking={asking}
            voicePhase={voice.phase}
            voicePartial={voice.partial}
            onAnalyze={onAnalyze}
            onApplyFix={onApplyFix}
            onReset={onReset}
            onEstop={statusAction("E-stop failed", api.estop)}
            onResume={statusAction("Resume failed", api.resume)}
            onHold={statusAction("Hold failed", api.hold)}
            onSelfCorrect={onSelfCorrect}
            forensicsFirst={tab === "replay"}
            forensics={
              replay && (
                <ReplayBreakdown
                  replay={replay}
                  armState={status?.state}
                  applying={applyingReplay}
                  onApply={onApplyReplay}
                  onSeekTime={seekTime}
                  onHold={statusAction("Hold failed", api.hold)}
                />
              )
            }
            onClearInterventions={statusAction("Clear failed", api.clearInterventions)}
            onRegenerateCounterfactual={onRegen}
            onAsk={onAsk}
            onVoicePress={voice.press}
            onVoiceRelease={voice.release}
          />
        </aside>
      </main>
    </div>
  );
}
