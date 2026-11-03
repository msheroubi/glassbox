import { useCallback, useState } from "react";
import { api, ApiError } from "./api";
import MuseXAIInspector, { type ChatMessage } from "./components/MuseXAIInspector";
import OverheadCanvas from "./components/OverheadCanvas";
import TelemetryScrubber from "./components/TelemetryScrubber";
import type { HistoryPoint, InterventionType } from "./types";
import { useTelemetry } from "./useTelemetry";
import { useVoice, type VoiceReply } from "./useVoice";

let chatSeq = 0;

export default function App() {
  const tel = useTelemetry();
  const { status, analysis, pushLog, setStatus, upsertAnalysis, setAnalyzing } = tel;
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
          <OverheadCanvas
            frameRef={tel.frameRef}
            subscribeFrame={tel.subscribeFrame}
            setView={tel.setView}
            status={status}
            analysis={analysis}
            scrubIndex={scrub ? scrub.frames[scrub.pos]?.index ?? null : null}
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
