import { useEffect, useMemo, useRef, useState } from "react";
import type { ReplayResult } from "../types";
import { NON_DISTRACTOR, labelColor } from "../viz";
import { type ShareMeasure, groundedLabel, hasNear, sharesAt, sortedKeyframes } from "../replayMath";

interface Props {
  replay: ReplayResult | null;
  mapKind: string | null;
  pos: number;
  onSeek: (pos: number) => void;
  culprit: string | null;
}

const M = { l: 42, r: 14, t: 30, b: 22 };
const EVENT_COLOR: Record<string, string> = {
  attention_shift: "#ff4d4d",
  failure: "#ff4d4d",
  anomaly: "#ffb020",
  staging: "#ff3df2",
  phase: "#7d8da1",
};
const LABEL_ORDER = ["target", "goal", "gripper"];

function niceStep(max: number): number {
  for (const s of [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5]) if (max / s <= 4) return s;
  return 0.5;
}

/** Share of attention on each SAM label over the replay, auto-scaled, with events and a draggable playhead. */
export default function AttentionTimeline({ replay, mapKind, pos, onSeek, culprit }: Props) {
  const wrap = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 600, h: 180 });
  const dragging = useRef(false);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const map = replay?.maps.find((m) => m.kind === mapKind);
  const kfs = useMemo(() => sortedKeyframes(map), [map]);
  const frames = replay?.frames ?? [];
  const n = frames.length;
  // Near each object by default: diffuse maps put ~1% strictly inside a small mask even when they hug it.
  const [choice, setChoice] = useState<ShareMeasure>("near");
  const measure: ShareMeasure = choice === "near" && hasNear(kfs) ? "near" : "shares";
  const val = (k: (typeof kfs)[number]) => (measure === "near" ? k.near ?? {} : k.shares);

  const labels = useMemo(() => {
    const set = new Set<string>();
    kfs.forEach((k) => Object.keys(measure === "near" ? k.near ?? {} : k.shares).forEach((l) => set.add(l)));
    return [...set].sort((a, b) => {
      const ia = LABEL_ORDER.indexOf(a);
      const ib = LABEL_ORDER.indexOf(b);
      return (ia < 0 ? 9 : ia) - (ib < 0 ? 9 : ib) || a.localeCompare(b);
    });
  }, [kfs, measure]);

  const { w, h } = size;
  const pw = Math.max(10, w - M.l - M.r);
  const ph = Math.max(10, h - M.t - M.b);
  const t0 = n ? frames[0].t : -1;
  const t1 = n ? frames[n - 1].t : 0;
  const span = Math.max(1e-3, t1 - t0);
  const x = (t: number) => M.l + ((t - t0) / span) * pw;
  const tAt = (p: number) => {
    if (!n) return 0;
    const i = Math.max(0, Math.min(n - 1, Math.floor(p)));
    const j = Math.min(n - 1, i + 1);
    return frames[i].t + (frames[j].t - frames[i].t) * (p - i);
  };

  const peak = Math.max(0.05, ...kfs.flatMap((k) => Object.values(val(k))));
  const step = niceStep(peak * 1.1);
  const ymax = Math.ceil((peak * 1.1) / step) * step;
  const y = (v: number) => M.t + ph - (v / ymax) * ph;
  const ticks = Array.from({ length: Math.round(ymax / step) + 1 }, (_, i) => i * step);

  const maxErr = Math.max(1e-6, ...frames.map((f) => f.joint_error));
  const errPath = n
    ? `M ${x(frames[0].t)} ${M.t + ph} ` +
      frames.map((f) => `L ${x(f.t)} ${M.t + ph - (f.joint_error / maxErr) * ph * 0.35}`).join(" ") +
      ` L ${x(t1)} ${M.t + ph} Z`
    : "";

  // Phase bands along the top.
  const phases: { from: number; to: number; name: string }[] = [];
  frames.forEach((f, i) => {
    const last = phases[phases.length - 1];
    if (last && last.name === f.phase) last.to = i;
    else phases.push({ from: i, to: i, name: f.phase });
  });

  const events = (replay?.events ?? []).filter((e) => e.kind !== "phase" && (e.kind !== "attention_shift" || e.map === mapKind));
  const now = sharesAt(kfs, pos, measure);
  const tNow = tAt(pos);

  const seekFromEvent = (clientX: number, el: SVGSVGElement) => {
    if (!n) return;
    const r = el.getBoundingClientRect();
    const t = t0 + ((clientX - r.left - M.l) / pw) * span;
    let best = 0;
    frames.forEach((f, i) => {
      if (Math.abs(f.t - t) < Math.abs(frames[best].t - t)) best = i;
    });
    // Interpolate between frames for a smooth drag.
    const nb = frames[best].t < t ? Math.min(n - 1, best + 1) : Math.max(0, best - 1);
    const dt = frames[nb].t - frames[best].t;
    const frac = dt ? Math.max(0, Math.min(1, (t - frames[best].t) / dt)) : 0;
    onSeek(best + (nb - best) * frac);
  };

  return (
    <div className="scrubber timeline">
      <div className="scrubber-head">
        <h3>{groundedLabel(mapKind) ? "Where SAM 3 finds the phrase, over time" : "Attention share over time"}</h3>
        <span className="muted">
          {!map
            ? "no attention map yet"
            : groundedLabel(mapKind)
              ? `${map.title}: share of SAM 3's map for the phrase inside each label's SAM 3.1 masks (perception, not attention)`
              : measure === "near"
              ? `${map.title}: share of the map near each object (within ~9% of the frame width of its SAM mask)`
              : `${map.title}: A(label) = share of the map inside that label's SAM masks`}
        </span>
        <span className="spacer" />
        {hasNear(kfs) && (
          <span className="cf-modes">
            <button className={measure === "near" ? "on" : ""} onClick={() => setChoice("near")}>
              near
            </button>
            <button className={measure === "shares" ? "on" : ""} onClick={() => setChoice("shares")}>
              in mask
            </button>
          </span>
        )}
        <span className="muted small">y auto-scaled · max {Math.round(ymax * 100)}%</span>
      </div>
      <div className="legend-row">
        {labels.map((l) => {
          const hijack = l === culprit;
          return (
            <span key={l} className={`legend-chip ${hijack ? "culprit" : ""}`}>
              <i style={{ background: labelColor(l, hijack) }} />
              {l} <b className="mono">{((now[l] ?? 0) * 100).toFixed(0)}%</b>
            </span>
          );
        })}
        <span className="legend-chip muted">
          <i className="err" />
          joint error
        </span>
        <span className="spacer" />
        <span className="mono small muted">t = {tNow.toFixed(2)} s</span>
      </div>
      <div className="chart" ref={wrap}>
        <svg
          width={w}
          height={h}
          onPointerDown={(e) => {
            dragging.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            seekFromEvent(e.clientX, e.currentTarget);
          }}
          onPointerMove={(e) => dragging.current && seekFromEvent(e.clientX, e.currentTarget)}
          onPointerUp={() => (dragging.current = false)}
        >
          {/* phase bands */}
          {phases.map((p, i) => {
            const xa = x(frames[p.from].t);
            const xb = x(frames[Math.min(n - 1, p.to + 1)].t);
            return (
              <g key={i}>
                <rect x={xa} y={4} width={Math.max(1, xb - xa)} height={12} className={`phase-band p${i % 2}`} />
                {xb - xa > 50 && (
                  <text x={xa + 4} y={13} className="phase-text">
                    {p.name}
                  </text>
                )}
              </g>
            );
          })}
          {/* grid */}
          {ticks.map((v) => (
            <g key={v}>
              <line x1={M.l} x2={M.l + pw} y1={y(v)} y2={y(v)} className="grid" />
              <text x={M.l - 6} y={y(v) + 4} className="axis" textAnchor="end">
                {Math.round(v * 100)}%
              </text>
            </g>
          ))}
          {n > 0 &&
            Array.from({ length: Math.floor(span) + 1 }, (_, i) => Math.ceil(t0) + i)
              .filter((t) => t <= t1 && (Math.floor(span) < 14 || t % 2 === 0))
              .map((t) => (
                <text key={t} x={x(t)} y={h - 6} className="axis" textAnchor="middle">
                  {t === 0 ? "0 s" : `${t}`}
                </text>
              ))}
          <path d={errPath} className="err-area" />
          {/* SAM keyframes */}
          {replay?.segmentations.map((s) => (
            <path key={`sam-${s.frame}`} d={`M ${x(s.t)} ${M.t + ph + 1} l -4 7 h 8 z`} className="sam-tick">
              <title>SAM 3.1 keyframe at {s.t.toFixed(2)} s</title>
            </path>
          ))}
          {/* events */}
          {events.map((e, i) => (
            <g
              key={`${e.kind}-${i}`}
              className="event"
              onPointerDown={(ev) => ev.stopPropagation()}
              onClick={() => onSeek(e.frame)}
            >
              <line
                x1={x(e.t)}
                x2={x(e.t)}
                y1={20}
                y2={M.t + ph}
                stroke={EVENT_COLOR[e.kind]}
                strokeDasharray={e.kind === "failure" ? undefined : "4 3"}
              />
              <circle cx={x(e.t)} cy={22} r={4} fill={EVENT_COLOR[e.kind]} />
              <title>
                {e.label} ({e.t.toFixed(2)} s): {e.detail}
              </title>
            </g>
          ))}
          {/* share lines */}
          {labels.map((l) => {
            const pts = kfs.map((k) => [x(k.t), y(val(k)[l] ?? 0)] as const);
            const hijack = l === culprit;
            const distractor = !NON_DISTRACTOR.has(l);
            return (
              <g key={l} className={`series ${l === "gripper" ? "dim" : ""}`}>
                {pts.length > 1 && (
                  <polyline
                    points={pts.map((p) => p.join(",")).join(" ")}
                    stroke={labelColor(l, hijack)}
                    strokeWidth={hijack || l === "target" ? 2.5 : 1.6}
                    strokeDasharray={distractor && !hijack ? "5 3" : undefined}
                    fill="none"
                  />
                )}
                {pts.map(([px, py], i) => (
                  <circle key={i} cx={px} cy={py} r={hijack ? 3.5 : 2.5} fill={labelColor(l, hijack)} />
                ))}
              </g>
            );
          })}
          {/* playhead */}
          {n > 0 && (
            <g className="playhead">
              <line x1={x(tNow)} x2={x(tNow)} y1={18} y2={M.t + ph} />
              <path d={`M ${x(tNow) - 5} 18 h 10 l -5 6 z`} />
            </g>
          )}
          {!kfs.length && n > 0 && (
            <text x={M.l + pw / 2} y={M.t + ph / 2} className="axis" textAnchor="middle">
              computing attention…
            </text>
          )}
        </svg>
      </div>
    </div>
  );
}
