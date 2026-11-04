import { useRef, useState, type PointerEvent } from "react";
import type { Counterfactual, CounterfactualMap } from "../types";

interface Props {
  cf: Counterfactual;
  onRerender?: () => void;
  onShowFrame?: () => void;
}

// Mirrors MOVED_LITTLE / PATH_LITTLE_PX in backend/xai_engine/counterfactual.py.
const MOVED_LITTLE = 0.15;
const PATH_LITTLE_PX = 15;

const OUTCOMES: Record<string, { cls: string; text: string }> = {
  confirmed: { cls: "ok", text: "cause confirmed" },
  implicated: { cls: "ok", text: "attention pulled by suspect" },
  partial: { cls: "cache", text: "partial effect" },
  ruled_out: { cls: "warn", text: "suspect ruled out" },
};

function outcome(cf: Counterfactual): { cls: string; text: string } | null {
  const m = cf.maps[0];
  if (cf.status !== "ready" || !m) return null;
  if (cf.outcome && OUTCOMES[cf.outcome]) return OUTCOMES[cf.outcome];
  // Counterfactuals recorded before the backend assessed them (no near-object flow).
  const path = cf.path_shift_px ?? 0;
  if (m.moved < MOVED_LITTLE && path < PATH_LITTLE_PX) return { cls: "warn", text: "suspect ruled out" };
  if (m.moved >= 0.25 || path >= PATH_LITTLE_PX) return { cls: "ok", text: "cause confirmed" };
  return { cls: "cache", text: "partial effect" };
}

function viewUrl(cf: Counterfactual, view: string): string {
  const [path, query] = (cf.image_url ?? "").split("?");
  return `${path}?view=${view}${query ? `&${query}` : ""}`;
}

function pct(v: number | undefined): string {
  return v === undefined ? "–" : `${Math.round(v * 100)}%`;
}

/** Attention near the object (its neighbourhood) before → after; the in-mask share for older counterfactuals. */
function ShareDelta({ label, m, name, lessIsBetter = false }: { label: string; m: CounterfactualMap; name: string; lessIsBetter?: boolean }) {
  const near = m.near_before?.[label] !== undefined && m.near_after?.[label] !== undefined;
  const a = near ? m.near_before![label] : m.shares_before[label];
  const b = near ? m.near_after![label] : m.shares_after[label];
  if (a === undefined || b === undefined || Math.max(a, b) < 0.01) return null;
  const better = lessIsBetter ? b < a - 0.01 : b > a + 0.01;
  const worse = lessIsBetter ? b > a + 0.01 : b < a - 0.01;
  return (
    <span className="cf-delta" title={near ? `share of the map near the ${name}` : `share of the map inside the ${name}'s SAM mask`}>
      {near ? "near " : ""}
      {name} {pct(a)} → <b className={better ? "good" : worse ? "bad" : ""}>{pct(b)}</b>
    </span>
  );
}

/** Where the attention that moved came from and went: e.g. "29% left the red book (4.3× its area)". */
function Flow({ label, m, name }: { label: string; m: CounterfactualMap; name: string }) {
  const area = m.area?.[label];
  const left = m.left_from?.[label];
  const arrived = m.arrived_at?.target;
  const tArea = m.area?.target;
  const lift = (v: number, a: number | undefined) => (a ? ` (${(v / a).toFixed(1)}× its area)` : "");
  const parts: string[] = [];
  if (left !== undefined && left >= 0.1) parts.push(`${pct(left)} left the ${name}${lift(left, area)}`);
  if (arrived !== undefined && arrived >= 0.1) parts.push(`${pct(arrived)} went near the target${lift(arrived, tArea)}`);
  if (m.peak_before && m.peak_before !== m.peak_after) {
    const where = (l: string | null | undefined) => (l ? (l === label ? `the ${name}` : `the ${l}`) : "open table");
    parts.push(`hottest spot: ${where(m.peak_before)} → ${where(m.peak_after)}`);
  }
  if (!parts.length) return null;
  return <span className="cf-flow muted small" title="of the attention that moved when the culprit was removed">{parts.join(" · ")}</span>;
}

/** Drag to wipe between the scene as the policy saw it (left) and without the culprit (right). */
function Wipe({ before, after, polygons }: { before: string; after: string; polygons: number[][][] }) {
  const [pos, setPos] = useState(50);
  const [size, setSize] = useState<[number, number] | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const move = (e: PointerEvent) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !dragging.current) return;
    setPos(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };

  return (
    <div
      className="cf-wipe"
      ref={box}
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        move(e);
      }}
      onPointerMove={move}
      onPointerUp={() => (dragging.current = false)}
    >
      <img src={after} alt="without the culprit" draggable={false} />
      <img
        src={before}
        alt="as the policy saw it"
        draggable={false}
        className="cf-top"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
      />
      {size && polygons.length > 0 && (
        <svg className="cf-region" viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="none">
          {polygons.map((p, i) => (
            <polygon key={i} points={p.map((q) => q.join(",")).join(" ")} />
          ))}
        </svg>
      )}
      <div className="cf-divider" style={{ left: `${pos}%` }}>
        <span>⇔</span>
      </div>
      <span className="cf-tag left">as the policy saw it</span>
      <span className="cf-tag right">counterfactual</span>
    </div>
  );
}

/** Muse Image removes the culprit inside its SAM region; the policies are asked where they'd look without it. */
export default function CounterfactualCard({ cf, onRerender, onShowFrame }: Props) {
  const [mode, setMode] = useState<"scene" | "attention">("scene");
  const verdict = outcome(cf);
  const hasAttention = cf.maps.length > 0 && cf.image_url !== null;
  const showAttention = mode === "attention" && hasAttention;
  const name = cf.culprit_label ?? "culprit";

  return (
    <div className="cf-card">
      <div className="cf-head">
        <b>Counterfactual test</b>
        <span className="muted">without {cf.removed}</span>
        <span className="spacer" />
        {cf.source && (
          <span className={`chip ${cf.source === "meta-api" ? "ok" : cf.source === "cache" ? "cache" : "warn"}`}>
            Muse Image: {cf.source}
          </span>
        )}
        {verdict && <span className={`chip ${verdict.cls}`}>{verdict.text}</span>}
      </div>

      {cf.image_url ? (
        <>
          <div className="cf-modes">
            <button className={mode === "scene" ? "on" : ""} onClick={() => setMode("scene")}>
              Scene
            </button>
            <button
              className={showAttention ? "on" : ""}
              disabled={!hasAttention}
              onClick={() => setMode("attention")}
              title={hasAttention ? "" : "waiting for the policy's attention on the edited frame"}
            >
              {cf.maps[0]?.title.split(" · ")[0] ?? "Policy"} attention
            </button>
            <span className="muted small">
              {showAttention ? "heatmap + predicted reach (red: with, green: without)" : "dashed: SAM region Muse Image may edit"}
            </span>
          </div>
          {showAttention ? (
            <Wipe before={viewUrl(cf, "attention_before")} after={viewUrl(cf, "attention_after")} polygons={[]} />
          ) : (
            <Wipe before={viewUrl(cf, "original")} after={cf.image_url} polygons={cf.region_polygons} />
          )}
        </>
      ) : (
        <div className="img-placeholder">
          {cf.status === "failed" ? (
            cf.verdict
          ) : (
            <>
              <span className="spinner" /> Muse Image is removing {cf.removed}…
            </>
          )}
        </div>
      )}

      {cf.maps.length > 0 && (
        <div className="cf-maps">
          {cf.maps.map((m) => (
            <div className="cf-map" key={m.kind}>
              <span className="cf-title">{m.title}</span>
              <span className="cf-moved" title="fraction of the attention that moved (total variation)">
                <span style={{ width: `${Math.round(m.moved * 100)}%` }} />
                <em>{pct(m.moved)} moved</em>
              </span>
              <ShareDelta label={cf.culprit_label ?? ""} m={m} name={name} lessIsBetter />
              <ShareDelta label="target" m={m} name="target" />
              <Flow label={cf.culprit_label ?? ""} m={m} name={name} />
            </div>
          ))}
        </div>
      )}

      {cf.status === "attending" && (
        <div className="muted small">
          <span className="spinner tiny" /> asking the policy where it would look without {cf.removed}…
        </div>
      )}
      {cf.status === "ready" && <p className="cf-verdict">{cf.verdict}</p>}

      <div className="cf-foot muted small">
        Same joint state, only the SAM region edited: any change in attention or reach is caused by {cf.removed}.
        {onShowFrame && (
          <>
            {" "}
            <button className="link" onClick={onShowFrame}>
              show the frame
            </button>
          </>
        )}
        {onRerender && (
          <>
            {" "}
            <button className="link" onClick={onRerender}>
              re-render
            </button>
          </>
        )}
      </div>
    </div>
  );
}
