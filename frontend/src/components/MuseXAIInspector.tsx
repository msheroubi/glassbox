import { useEffect, useRef, useState } from "react";
import type { VoicePhase } from "../useVoice";
import { INTERVENTIONS } from "../types";
import type { AnalysisResult, InterventionType, LogEntry, ResultSource, RuntimeStatus } from "../types";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant" | "system";
  text: string;
  source?: ResultSource | string;
}

interface Props {
  connected: boolean;
  status: RuntimeStatus | null;
  analysis: AnalysisResult | null;
  analyzing: boolean;
  applying: boolean;
  log: LogEntry[];
  chat: ChatMessage[];
  asking: boolean;
  voicePhase: VoicePhase;
  voicePartial: string;
  onAnalyze: () => void;
  onApplyFix: (override?: InterventionType) => void;
  onReset: (scenario?: string) => void;
  onEstop: () => void;
  onResume: () => void;
  onClearInterventions: () => void;
  onRegenerateCounterfactual: () => void;
  onAsk: (q: string) => void;
  onVoicePress: () => void;
  onVoiceRelease: () => void;
}

const FAILURE_LABEL: Record<string, string> = {
  ATTENTION_HIJACK: "Attention hijack",
  SERVO_STALL_COLLISION: "Servo stall / collision",
  OCCLUSION_OOD: "Occlusion / out-of-distribution",
  SPECULAR_GLARE: "Specular glare",
  NONE: "No failure",
};

const FIX_LABEL: Record<InterventionType, string> = {
  INPAINT_MASK_SUPPRESSION: "Real-time visual mask suppression",
  WAYPOINT_BYPASS: "Waypoint bypass (Muse Wonder)",
  RETRY_GRASP_OFFSET: "Retry grasp with offset",
  NONE: "Operator clears workspace, resume",
};

const NON_DISTRACTOR = new Set(["target", "gripper", "goal"]);

function SourceChip({ label, source }: { label: string; source: string | null | undefined }) {
  const cls = source === "meta-api" ? "ok" : source === "cache" ? "cache" : source ? "warn" : "off";
  return (
    <span className={`chip ${cls}`} title={source ?? "not run"}>
      {label}: {source ?? "-"}
    </span>
  );
}

function fmtParams(p: Record<string, number | string>): string {
  const entries = Object.entries(p);
  return entries.length ? entries.map(([k, v]) => `${k}=${v}`).join(", ") : "none";
}

function fmtTime(t: number): string {
  return new Date(t * 1000).toLocaleTimeString([], { hour12: false });
}

export default function MuseXAIInspector(props: Props) {
  const { status, analysis, analyzing, applying } = props;
  const [scenario, setScenario] = useState<string>("");
  const [override, setOverride] = useState<"" | InterventionType>("");
  const [question, setQuestion] = useState("");
  const chatEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (status?.scenario && !scenario) setScenario(status.scenario);
  }, [status?.scenario, scenario]);
  useEffect(() => setOverride(""), [analysis?.analysis_id]);
  useEffect(() => chatEnd.current?.scrollIntoView({ block: "nearest" }), [props.chat.length]);

  const diag = analysis?.diagnosis;
  const thr = status?.thresholds.attention_hijack ?? 0.45;
  const fix: InterventionType | undefined = override || diag?.intervention_type;
  const state = status?.state ?? "OFFLINE";
  const scenarios = status?.scenarios ?? {};

  const submitQuestion = () => {
    const q = question.trim();
    if (!q) return;
    props.onAsk(q);
    setQuestion("");
  };

  return (
    <div className="inspector">
      {/* ------------------------------------------------------------ run state + controls */}
      <section className="card">
        <div className="row">
          <span className={`state-badge s-${state.toLowerCase()}`}>{state}</span>
          <span className="status-text" title={status?.status}>{status?.status ?? (props.connected ? "" : "backend offline")}</span>
        </div>
        <div className="row chips">
          <span className="chip">
            {status?.camera_only ? "webcam only (no arm)" : status?.mock_hardware ? "mock hardware" : "SO-101 hardware"}
          </span>
          <span className="chip">policy: {status?.policy ?? "-"}</span>
          <span className="chip">phase: {status?.phase ?? "-"}</span>
          <span className={`chip ${status?.meta_api_enabled ? "ok" : "warn"}`}>
            Meta API {status?.meta_api_enabled ? "connected" : "off (local fallbacks)"}
          </span>
          {status?.recovery_active && <span className="chip cache">recovery executing</span>}
          {status && (status.grasp_offset_mm[0] !== 0 || status.grasp_offset_mm[1] !== 0) && (
            <span className="chip cache">
              grasp offset ({status.grasp_offset_mm[0]}, {status.grasp_offset_mm[1]}) mm
            </span>
          )}
        </div>
        <div className="row controls">
          {status?.mock_hardware && (
            <>
              <select value={scenario} onChange={(e) => setScenario(e.target.value)} title={scenarios[scenario]}>
                {Object.keys(scenarios).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
              <button className="btn" onClick={() => props.onReset(scenario || undefined)}>
                ↺ Reset scenario
              </button>
            </>
          )}
          {!status?.mock_hardware && (
            <button className="btn" onClick={() => props.onReset()}>
              ↺ Reset
            </button>
          )}
          <span className="spacer" />
          {state === "ESTOP" || state === "HOLD" ? (
            <button className="btn" onClick={props.onResume}>
              ▶ Resume policy
            </button>
          ) : null}
          <button className="btn danger" onClick={props.onEstop}>
            ■ E-STOP
          </button>
        </div>
        {status?.mock_hardware && scenarios[scenario] && <p className="muted small">{scenarios[scenario]}</p>}
        {!!status?.suppression.length && (
          <div className="row suppress-row">
            <span>
              Suppressing in policy input:{" "}
              {status.suppression.map((s) => (
                <b key={s.mask_id}>
                  #{s.mask_id} {s.label}{" "}
                </b>
              ))}
            </span>
            <span className="spacer" />
            <button className="btn small" onClick={props.onClearInterventions}>
              Clear interventions
            </button>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------ analysis trigger + diagnosis */}
      <section className="card">
        <button className="btn primary big" onClick={props.onAnalyze} disabled={analyzing || !props.connected}>
          {analyzing ? <span className="spinner" /> : "🔍"} {analyzing ? "Analysing with SAM 3.1 + Muse Spark…" : "Trigger Failure Analysis"}
        </button>

        {!analysis && !analyzing && status?.camera_only && (
          <p className="muted small">
            Camera-only mode: no arm is connected, so the joint/servo triggers are off. The heatmap is bottom-up visual
            saliency (no policy). Trigger analysis to have SAM 3.1 segment the named objects and Muse Spark explain the scene,
            or ask a question below.
          </p>
        )}
        {!analysis && !analyzing && !status?.camera_only && (
          <p className="muted small">
            Anomalies (joint error &gt; {status?.thresholds.joint_error_deg ?? "-"}°, chunk variance &gt;{" "}
            {status?.thresholds.chunk_variance ?? "-"}, or A(distractor) &gt; {thr}) pause the policy and trigger analysis
            automatically. You can also trigger it manually, or say "analyze".
          </p>
        )}

        {analysis && diag && (
          <div className="diagnosis">
            <div className="row">
              <span className={`failure ${diag.failure_detected ? "bad" : "good"}`}>{FAILURE_LABEL[diag.failure_type]}</span>
              <span className="confidence">
                <span style={{ width: `${Math.round(diag.confidence * 100)}%` }} />
                <em>{Math.round(diag.confidence * 100)}% confidence</em>
              </span>
            </div>
            <h2 className="headline">{diag.headline_summary}</h2>
            <p className="root-cause">{diag.technical_root_cause}</p>
            <div className="row chips">
              <SourceChip label="SAM 3.1" source={analysis.sam_source} />
              <SourceChip label="Muse Spark" source={analysis.diagnosis_source} />
              {analysis.plan_status !== "not_needed" && <SourceChip label="Muse Wonder" source={analysis.plan_source} />}
              <SourceChip label="Muse Image" source={analysis.counterfactual_source} />
            </div>
            <div className="meta-line muted small">
              {analysis.trigger} trigger · {fmtTime(analysis.created_at)} · culprit mask{" "}
              {diag.culprit_mask_id === null ? "none" : `#${diag.culprit_mask_id}`} ·{" "}
              {Object.entries(analysis.latency_ms)
                .map(([k, v]) => `${k.replace(/_ms$/, "")} ${Math.round(v)} ms`)
                .join(" · ")}
            </div>
            {analysis.anomaly && (
              <ul className="reasons">
                {analysis.anomaly.reasons.map((r) => (
                  <li key={r}>⚠ {r}</li>
                ))}
              </ul>
            )}
            <details>
              <summary>Evidence sent to Muse Spark</summary>
              <pre className="code">{JSON.stringify(analysis.evidence, null, 1)}</pre>
            </details>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------ fix */}
      {analysis && diag && (
        <section className="card fix-card">
          <div className="row">
            <div>
              <div className="muted small">Recommended intervention</div>
              <div className="fix-name">{FIX_LABEL[diag.intervention_type]}</div>
              <div className="mono small">{fmtParams(diag.intervention_parameters)}</div>
            </div>
            <span className="spacer" />
            <select value={override} onChange={(e) => setOverride(e.target.value as "" | InterventionType)} title="Override the intervention">
              <option value="">as diagnosed</option>
              {INTERVENTIONS.map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </div>
          <button
            className="btn fix big"
            onClick={() => props.onApplyFix(override || undefined)}
            disabled={applying || analysis.analysis_id !== status?.latest_analysis_id}
            title={analysis.analysis_id !== status?.latest_analysis_id ? "This analysis is stale; run a new one" : ""}
          >
            {applying ? <span className="spinner" /> : "⚡"} 1-Click Apply Muse Fix{fix ? `: ${FIX_LABEL[fix]}` : ""}
          </button>
          {fix === "WAYPOINT_BYPASS" && analysis.plan_status === "pending" && (
            <p className="muted small">
              Muse Wonder is still refining the plan; applying now executes the validated local plan shown below.
            </p>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------ attribution */}
      {analysis && (
        <section className="card">
          <h3>Attention-to-mask attribution A(i)</h3>
          <div className="bars">
            {[...analysis.attributions]
              .sort((a, b) => b.attention_share - a.attention_share)
              .map((a) => {
                const hijack = !NON_DISTRACTOR.has(a.label) && a.attention_share > thr;
                const culprit = a.mask_id === diag?.culprit_mask_id;
                return (
                  <div key={a.mask_id} className={`bar-row ${culprit ? "culprit" : ""}`}>
                    <span className="bar-label">
                      #{a.mask_id} {a.label}
                    </span>
                    <span className="bar-track">
                      <span
                        className={`bar-fill ${hijack ? "hijack" : a.label === "target" ? "target" : ""}`}
                        style={{ width: `${Math.min(100, a.attention_share * 100)}%` }}
                      />
                      <span className="bar-thr" style={{ left: `${thr * 100}%` }} title={`hijack threshold ${thr}`} />
                    </span>
                    <span className="bar-val mono">{a.attention_share.toFixed(2)}</span>
                  </div>
                );
              })}
          </div>
          <p className="muted small">
            A(i) = saliency inside SAM mask i / total saliency. Dashed line: hijack threshold {thr}.
          </p>
        </section>
      )}

      {/* ------------------------------------------------------------ keyframe + counterfactual */}
      {analysis && (
        <section className="card">
          <h3>Keyframe and Muse Image counterfactual</h3>
          <div className="images">
            <figure>
              <img src={analysis.keyframe_url} alt="keyframe: raw | SAM masks + saliency" />
              <figcaption>Anomaly keyframe: raw | SAM 3.1 masks + saliency</figcaption>
            </figure>
            <figure>
              {analysis.counterfactual_url ? (
                <img src={analysis.counterfactual_url} alt="counterfactual" />
              ) : (
                <div className="img-placeholder">
                  {diag?.culprit_mask_id !== null ? (
                    <>
                      <span className="spinner" /> rendering counterfactual…
                    </>
                  ) : (
                    "no culprit object to remove"
                  )}
                </div>
              )}
              <figcaption>
                "What if the culprit were not there?"{" "}
                <button className="link" onClick={props.onRegenerateCounterfactual}>
                  re-render
                </button>
              </figcaption>
            </figure>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------ wonder plan */}
      {analysis && analysis.plan_status !== "not_needed" && (
        <section className="card">
          <h3>
            Recovery plan{" "}
            <span className={`chip ${analysis.plan_status === "ready" ? "ok" : analysis.plan_status === "failed" ? "warn" : "cache"}`}>
              {analysis.plan_status === "pending" ? "Muse Wonder refining…" : analysis.plan_status}
            </span>
            {analysis.plan_source && <SourceChip label="source" source={analysis.plan_source} />}
          </h3>
          {analysis.plan ? (
            <>
              <p>{analysis.plan.summary}</p>
              {analysis.plan.waypoints.length > 0 && (
                <table className="wp">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>pan</th>
                      <th>lift</th>
                      <th>elbow</th>
                      <th>wrist</th>
                      <th>roll</th>
                      <th>grip</th>
                      <th>s</th>
                      <th>note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.plan.waypoints.map((w, i) => (
                      <tr key={i}>
                        <td>WP{i + 1}</td>
                        {w.qpos.map((q, j) => (
                          <td key={j} className="mono">
                            {q.toFixed(1)}
                          </td>
                        ))}
                        <td className="mono">{w.duration_s.toFixed(1)}</td>
                        <td className="note">{w.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="muted small">
                suppress masks: {analysis.plan.suppress_mask_ids.length ? analysis.plan.suppress_mask_ids.join(", ") : "none"} · grasp offset (
                {analysis.plan.grasp_offset_mm.join(", ")}) mm
              </div>
              {analysis.plan.safety_notes && <p className="safety">🛡 {analysis.plan.safety_notes}</p>}
              <details>
                <summary>Muse Wonder routine (audit only, never executed)</summary>
                <pre className="code">{analysis.plan.python_routine}</pre>
              </details>
            </>
          ) : (
            <p className="muted">{analysis.plan_status === "failed" ? "Planning failed." : "Planning…"}</p>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------ assistant */}
      <section className="card">
        <h3>Ask GlassBox (Muse Spark + Muse Voice)</h3>
        <div className="chat">
          {props.chat.length === 0 && (
            <p className="muted small">
              Ask "why did it stop?" or hold the mic and say "analyze", "apply the fix", "reset" or "emergency stop".
            </p>
          )}
          {props.chat.map((m) => (
            <div key={m.id} className={`msg ${m.role}`}>
              <span>{m.text}</span>
              {m.source && <em className="muted small"> · {m.source}</em>}
            </div>
          ))}
          {props.voicePhase !== "idle" && props.voicePartial && <div className="msg user partial">{props.voicePartial}…</div>}
          <div ref={chatEnd} />
        </div>
        <div className="row ask">
          <input
            value={question}
            placeholder="Why is the robot hesitating?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submitQuestion()}
            disabled={props.asking}
          />
          <button className="btn" onClick={submitQuestion} disabled={props.asking || !question.trim()}>
            {props.asking ? <span className="spinner" /> : "Ask"}
          </button>
          <button
            className={`btn mic ${props.voicePhase}`}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              props.onVoicePress();
            }}
            onPointerUp={props.onVoiceRelease}
            onPointerCancel={props.onVoiceRelease}
            title="Hold to talk (Muse Voice Transcribe)"
          >
            {props.voicePhase === "listening"
              ? "🎙 Listening…"
              : props.voicePhase === "connecting"
                ? "🎙 Connecting…"
                : props.voicePhase === "processing"
                  ? "🎙 Thinking…"
                  : "🎙 Hold to talk"}
          </button>
        </div>
      </section>

      {/* ------------------------------------------------------------ log */}
      <section className="card">
        <h3>Event log</h3>
        <div className="log">
          {props.log.map((e) => (
            <div key={e.id} className={`log-row ${e.kind}`}>
              <span className="mono muted">{fmtTime(e.t)}</span> {e.text}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
