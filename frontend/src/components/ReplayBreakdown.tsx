import CounterfactualCard from "./CounterfactualCard";
import type { FailureStage, InterventionType, ReplayResult, ResultSource } from "../types";

interface Props {
  replay: ReplayResult;
  armState: string | undefined;
  applying: boolean;
  onApply: () => void;
  onSeekTime: (t: number) => void;
  onHold: () => void;
}

const STAGES: { key: FailureStage; title: string; question: string }[] = [
  { key: "PERCEPTION", title: "Perception", question: "Did it see the scene right?" },
  { key: "DECISION", title: "Decision", question: "Did it choose the right action?" },
  { key: "ACTUATION", title: "Actuation", question: "Could the arm execute it?" },
];

const FIX_LABEL: Record<InterventionType, string> = {
  INPAINT_MASK_SUPPRESSION: "Erase the culprit from the policy's camera input",
  WAYPOINT_BYPASS: "Detour around the obstacle",
  RETRY_GRASP_OFFSET: "Retry the grasp with an offset",
  NONE: "No automatic fix",
};

const STATUS_TEXT: Record<string, string> = {
  saving: "saving the run",
  segmenting: "SAM 3.1 segmenting keyframes",
  attention: "computing attention",
  breakdown: "Muse Wonder is analysing",
  refining: "refining attention keyframes",
  done: "complete",
  failed: "failed",
};

function sourceClass(source: ResultSource | null | undefined): string {
  return source === "meta-api" ? "ok" : source === "cache" ? "cache" : source ? "warn" : "";
}

function fmtParams(p: Record<string, number | string>): string {
  const e = Object.entries(p);
  return e.length ? e.map(([k, v]) => `${k}=${v}`).join(", ") : "";
}

/** Muse Wonder's forensic breakdown of the replayed failure, with the self-correction outcome. */
export default function ReplayBreakdown({ replay, armState, applying, onApply, onSeekTime, onHold }: Props) {
  const b = replay.breakdown;
  const busy = replay.status !== "done" && replay.status !== "failed";
  const failing = b?.failure_stage ?? null;
  const failIdx = STAGES.findIndex((s) => s.key === failing);

  return (
    <section className="card forensic">
      <h3>
        Forensic breakdown
        <span className="spacer" />
        {b && <span className={`chip ${sourceClass(replay.breakdown_source)}`}>Muse Wonder: {replay.breakdown_source}</span>}
        <span className={`chip ${busy ? "cache" : replay.status === "failed" ? "warn" : "ok"}`}>
          {busy && <span className="spinner tiny" />} {STATUS_TEXT[replay.status] ?? replay.status}
        </span>
      </h3>

      {busy && (
        <div className="progress-line">
          <div className="analysis-progress inline">
            <span style={{ width: `${Math.round(replay.progress * 100)}%` }} />
            <em>{replay.progress_text || STATUS_TEXT[replay.status]}</em>
          </div>
          {replay.status === "refining" && armState === "RUNNING" && (
            <p className="muted small">
              Attention refinement pauses while the arm is running so the policy keeps the CPU.{" "}
              <button className="link" onClick={onHold}>
                Hold the arm
              </button>{" "}
              to let it finish.
            </p>
          )}
        </div>
      )}

      <div className="stages">
        {STAGES.map((s, i) => {
          const state = !b ? "pending" : failIdx < 0 ? "ok" : i < failIdx ? "ok" : i === failIdx ? "bad" : "na";
          return (
            <div key={s.key} className={`stage ${state}`} title={s.question}>
              <span className="stage-icon">{state === "ok" ? "✓" : state === "bad" ? "✕" : state === "na" ? "–" : "…"}</span>
              <span className="stage-title">{s.title}</span>
              <span className="stage-q">
                {state === "bad" ? "failure starts here" : state === "na" ? "not reached" : s.question}
              </span>
            </div>
          );
        })}
      </div>

      {!b && (
        <p className="muted small">
          GlassBox is replaying the {replay.n_frames} recorded frames ({(replay.n_frames / replay.fps).toFixed(1)} s) offline: SAM
          3.1 segments the scene, ACT's attention (and the observe-only lens's maps) is attributed to every object, then Muse Wonder decides whether the
          failure started in perception, decision or actuation.
        </p>
      )}

      {b && (
        <>
          <h2 className="headline">{b.headline}</h2>
          <p className="lead">{b.operator_explanation}</p>
          <div className="row">
            <span className="confidence">
              <span style={{ width: `${Math.round(b.confidence * 100)}%` }} />
              <em>{Math.round(b.confidence * 100)}% confidence</em>
            </span>
            {b.culprit_label && <span className="chip culprit-chip">culprit: {b.culprit_label}</span>}
          </div>

          <div className="facts">
            <div>
              <div className="fact-k">What the policy perceived</div>
              <p>{b.perception_analysis}</p>
            </div>
            <div>
              <div className="fact-k">Root cause</div>
              <p>{b.root_cause}</p>
            </div>
          </div>

          {b.timeline.length > 0 && (
            <ol className="forensic-timeline">
              {b.timeline.map((e, i) => (
                <li key={i} onClick={() => onSeekTime(e.t)} title="Jump the replay here">
                  <span className="mono t">
                    {e.t > 0 ? "+" : ""}
                    {e.t.toFixed(1)} s
                  </span>
                  <span>
                    <b>{e.event}</b>
                    <span className="muted"> · {e.evidence}</span>
                  </span>
                </li>
              ))}
            </ol>
          )}

          {!replay.counterfactual && (
            <div className="counterfactual">
              <span className="fact-k">Counterfactual</span> {b.counterfactual}
            </div>
          )}

          <SelfCorrection replay={replay} applying={applying} onApply={onApply} />
        </>
      )}

      {replay.counterfactual && (
        <CounterfactualCard
          cf={replay.counterfactual}
          onShowFrame={
            replay.counterfactual.frame !== null
              ? () => onSeekTime(replay.frames[replay.counterfactual!.frame!]?.t ?? 0)
              : undefined
          }
        />
      )}

      <div className="meta-line muted small">
        run {replay.replay_id} · {replay.trigger} trigger · {replay.n_frames} frames @ {replay.fps} fps
        {Object.entries(replay.latency_ms)
          .map(([k, v]) => ` · ${k.replace(/_ms$/, "")} ${(v / 1000).toFixed(1)} s`)
          .join("")}
      </div>
    </section>
  );
}

function SelfCorrection({ replay, applying, onApply }: { replay: ReplayResult; applying: boolean; onApply: () => void }) {
  const b = replay.breakdown!;
  const fix = `${FIX_LABEL[b.intervention_type]}${fmtParams(b.intervention_parameters) ? ` (${fmtParams(b.intervention_parameters)})` : ""}`;
  const sc = replay.self_correction;

  if (sc === "applied") {
    return (
      <div className="selfcorrect applied">
        <div className="sc-title">✓ Self-corrected, no human needed</div>
        <div>{fix}</div>
        {replay.self_correction_note && <div className="muted small">{replay.self_correction_note}</div>}
      </div>
    );
  }
  if (b.needs_human || !b.self_correctable) {
    return (
      <div className="selfcorrect human">
        <div className="sc-title">✋ Needs a human</div>
        <div>{b.human_action || "GlassBox cannot fix this by itself."}</div>
        {replay.self_correction_note && <div className="muted small">{replay.self_correction_note}</div>}
      </div>
    );
  }
  const canApply = sc === "pending" || (sc === "off" && !!replay.analysis_id);
  return (
    <div className="selfcorrect fixable">
      <div className="sc-title">⚡ GlassBox can fix this itself</div>
      <div>{fix}</div>
      {replay.self_correction_note && <div className="muted small">{replay.self_correction_note}</div>}
      {canApply && (
        <button className="btn fix big" onClick={onApply} disabled={applying}>
          {applying ? <span className="spinner" /> : "⚡"} Apply self-correction and resume
        </button>
      )}
    </div>
  );
}
