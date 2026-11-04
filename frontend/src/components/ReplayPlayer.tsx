import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { replayFrameUrl } from "../api";
import { MAP_ORDER, bracket, groundedLabel, hasNear, mapLabel, mapPhrase, nearestSegmentation, sharesAt, sortedKeyframes } from "../replayMath";
import type { ReplayKeyframe, ReplayResult, SavedRun } from "../types";
import { HIJACK_COLOR, JET_LUT, NON_DISTRACTOR, decodeB64, drawTag, labelColor, tracePolygons } from "../viz";

type ReplayView = "policy" | "raw";
const SPEEDS = [0.25, 0.5, 1];

interface Props {
  replay: ReplayResult | null;
  pos: number; // fractional frame index
  onPos: (pos: number) => void;
  mapKind: string | null;
  onMapKind: (kind: string) => void;
  autoplayKey: number; // bumps when a new failure replay arrives
  savedRuns: SavedRun[];
  onOpenRun: (id: string) => void;
  onRefreshRuns: () => void;
}

export default function ReplayPlayer({
  replay,
  pos,
  onPos,
  mapKind,
  onMapKind,
  autoplayKey,
  savedRuns,
  onOpenRun,
  onRefreshRuns,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatRef = useRef<HTMLCanvasElement | null>(null);
  const images = useRef(new Map<string, HTMLImageElement>());
  const decoded = useRef(new Map<string, Uint8Array>());
  const raf = useRef<number | null>(null);
  const [view, setView] = useState<ReplayView>("policy");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [showHeat, setShowHeat] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [opacity, setOpacity] = useState(0.85);
  const [loaded, setLoaded] = useState(0);

  const rid = replay?.replay_id ?? null;
  const n = replay?.n_frames ?? 0;
  const map = replay?.maps.find((m) => m.kind === mapKind);
  const kfs = useMemo(() => sortedKeyframes(map), [map]);
  const frame = Math.max(0, Math.min(n - 1, Math.round(pos)));
  const info = replay?.frames[frame];

  // ------------------------------------------------------------------ frame preloading
  const imgKey = (i: number, v: ReplayView) => `${rid}:${v}:${i}`;
  useEffect(() => {
    setLoaded(0);
    if (!rid) return;
    let count = 0;
    const cache = images.current;
    if (cache.size > 1200) cache.clear();
    for (let i = 0; i < n; i++) {
      const key = imgKey(i, view);
      const have = cache.get(key);
      if (have?.complete && have.naturalWidth) {
        count++;
        continue;
      }
      if (have) continue;
      const img = new Image();
      img.decoding = "async";
      img.onload = () => setLoaded((c) => c + 1);
      img.onerror = () => cache.delete(key); // retried on the next pass
      img.src = replayFrameUrl(rid, i, view);
      cache.set(key, img);
    }
    setLoaded((c) => c + count);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rid, n, view]);

  // ------------------------------------------------------------------ playback
  // The canvas draws from posRef every animation frame (smooth heatmap interpolation); React state, and with
  // it the timeline and the rest of the UI, follows at ~12 Hz.
  const posRef = useRef(pos);
  const emitted = useRef(pos);
  if (pos !== emitted.current) {
    posRef.current = pos; // an external seek
    emitted.current = pos;
  }
  const emit = useCallback(
    (p: number) => {
      emitted.current = p;
      onPos(p);
    },
    [onPos],
  );
  useEffect(() => {
    if (!playing || !n) return;
    let last = performance.now();
    let lastEmit = 0;
    let id = 0;
    const fps = replay?.fps ?? 15;
    const step = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const next = Math.min(n - 1, posRef.current + dt * fps * speed);
      posRef.current = next;
      drawRef.current();
      if (next >= n - 1) {
        emit(next);
        setPlaying(false);
        return;
      }
      if (now - lastEmit > 80) {
        lastEmit = now;
        emit(next);
      }
      id = requestAnimationFrame(step);
    };
    id = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(id);
      emit(posRef.current);
    };
  }, [playing, speed, n, emit, replay?.fps]);

  useEffect(() => {
    if (autoplayKey > 0) {
      onPos(0);
      setPlaying(true);
    }
  }, [autoplayKey, onPos]);

  const togglePlay = () => {
    if (!playing && pos >= n - 1.01) onPos(0);
    setPlaying((p) => !p);
  };
  const jumpKeyframe = (dir: 1 | -1) => {
    setPlaying(false);
    const stops = [...new Set([...kfs.map((k) => k.frame), replay?.failure_frame ?? n - 1, 0])].sort((a, b) => a - b);
    const target = dir > 0 ? stops.find((f) => f > pos + 0.01) : [...stops].reverse().find((f) => f < pos - 0.01);
    if (target !== undefined) onPos(target);
  };

  // ------------------------------------------------------------------ drawing
  const grid = useCallback(
    (kf: ReplayKeyframe): Uint8Array => {
      const key = `${rid}:${mapKind}:${kf.frame}:${kf.grid.length}`;
      let g = decoded.current.get(key);
      if (!g) {
        if (decoded.current.size > 400) decoded.current.clear();
        g = decodeB64(kf.grid);
        decoded.current.set(key, g);
      }
      return g;
    },
    [rid, mapKind],
  );

  const draw = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    if (!canvas || !replay) return;
    const ctx = canvas.getContext("2d")!;
    const p = posRef.current;
    const i = Math.max(0, Math.min(replay.n_frames - 1, Math.round(p)));
    // Fall back to the nearest loaded frame so scrubbing never flashes black.
    let img: HTMLImageElement | undefined;
    for (let d = 0; d < replay.n_frames && !img; d++) {
      for (const j of [i - d, i + d]) {
        const c = images.current.get(`${replay.replay_id}:${view}:${j}`);
        if (c?.complete && c.naturalWidth) {
          img = c;
          break;
        }
      }
    }
    if (!img) return;
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    const W = canvas.width;
    const H = canvas.height;
    const scale = Math.max(1, W / 640);
    ctx.drawImage(img, 0, 0);

    const br = map ? bracket(kfs, p) : null;
    if (showHeat && map && br) {
      let heat = heatRef.current;
      if (!heat) heat = heatRef.current = document.createElement("canvas");
      if (heat.width !== map.grid_w || heat.height !== map.grid_h) {
        heat.width = map.grid_w;
        heat.height = map.grid_h;
      }
      const ga = grid(br.a);
      const gb = br.w > 0 ? grid(br.b) : ga;
      const hctx = heat.getContext("2d")!;
      const out = hctx.createImageData(map.grid_w, map.grid_h);
      const len = Math.min(ga.length, gb.length, map.grid_w * map.grid_h);
      for (let k = 0; k < len; k++) {
        const v = Math.round(ga[k] * (1 - br.w) + gb[k] * br.w);
        const o = k * 4;
        out.data[o] = JET_LUT[v * 4];
        out.data[o + 1] = JET_LUT[v * 4 + 1];
        out.data[o + 2] = JET_LUT[v * 4 + 2];
        out.data[o + 3] = JET_LUT[v * 4 + 3] * opacity;
      }
      hctx.putImageData(out, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(heat, 0, 0, W, H);
    }

    const seg = showMasks ? nearestSegmentation(replay.segmentations, i) : null;
    if (seg) {
      // Attention near each object (in-mask for replays recorded before it); mirrors NEAR_DRAW_LIFT in attribution.py.
      const near = hasNear(kfs);
      const shares = sharesAt(kfs, p, near ? "near" : "shares");
      const lifts = near ? sharesAt(kfs, p, "near_lift") : {};
      // A SAM 3 map is perception, not attention: an object other than the phrase's own holding more of the
      // phrase's map than it is being read as that phrase.
      const ref = groundedLabel(mapKind) ?? "target";
      const target = shares[ref] ?? 0;
      const flag = groundedLabel(mapKind) ? ` ◀ READ AS '${mapPhrase(map?.title ?? "") ?? ref}'` : " ◀ HIJACK";
      const far = Math.abs(seg.frame - i) > 3; // masks are from a SAM keyframe; moving parts drift away from it
      for (const o of seg.objects) {
        if (!o.polygons.length || (far && o.label === "gripper")) continue;
        const share = shares[o.label] ?? 0;
        const hijack = !NON_DISTRACTOR.has(o.label) && o.label !== ref && share > target && share > 0.05 && (!near || (lifts[o.label] ?? 0) >= 1.3);
        const color = labelColor(o.label, hijack);
        tracePolygons(ctx, o.polygons);
        ctx.globalAlpha = far ? 0.06 : 0.12;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = far ? 0.55 : 1;
        ctx.lineWidth = 2 * scale;
        ctx.strokeStyle = color;
        ctx.setLineDash(far ? [6 * scale, 4 * scale] : []);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        const tag = map && o.label in shares ? `${o.label} ${(share * 100).toFixed(0)}%${near ? " near" : ""}${hijack ? flag : ""}` : o.label;
        drawTag(ctx, tag, o.bbox[0], o.bbox[1] - 2, hijack ? HIJACK_COLOR : color, scale);
      }
    }

    if (i >= replay.failure_frame) {
      ctx.lineWidth = 6 * scale;
      ctx.strokeStyle = "rgba(255, 77, 77, 0.85)";
      ctx.strokeRect(3 * scale, 3 * scale, W - 6 * scale, H - 6 * scale);
    }
  }, [replay, view, map, kfs, showHeat, showMasks, opacity, grid]);

  const drawRef = useRef(draw);
  drawRef.current = draw;
  // Always run the latest draw: a frame queued before a prop change must not paint with stale props.
  const schedule = useCallback(() => {
    if (raf.current === null) raf.current = requestAnimationFrame(() => drawRef.current());
  }, []);
  useEffect(() => schedule(), [draw, schedule]);
  useEffect(() => schedule(), [pos, schedule, loaded]);
  useEffect(
    () => () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    },
    [],
  );

  // Keyboard: space play/pause, arrows step a frame, shift+arrows jump keyframes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!replay || (e.target as HTMLElement)?.closest("input, textarea, select")) return;
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const dir = e.key === "ArrowRight" ? 1 : -1;
        if (e.shiftKey) jumpKeyframe(dir);
        else {
          setPlaying(false);
          onPos(Math.max(0, Math.min(n - 1, Math.round(pos) + dir)));
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ------------------------------------------------------------------ overlays
  const t = info?.t ?? 0;
  const live = replay?.events.filter(
    (e) => Math.abs(e.t - t) < 0.6 && (e.kind === "attention_shift" ? e.map === mapKind : e.kind === "staging"),
  );
  const br = bracket(kfs, pos);
  const interp = !br
    ? "no attention keyframes yet"
    : br.w === 0 || br.a === br.b
      ? `keyframe at ${br.a.t.toFixed(2)} s`
      : `interpolated ${br.a.t.toFixed(2)} → ${br.b.t.toFixed(2)} s`;
  const analysing = replay && replay.status !== "done" && replay.status !== "failed";
  const preloadPct = n ? Math.min(100, Math.round((loaded / n) * 100)) : 0;
  const maps = replay ? [...replay.maps].sort((a, b) => MAP_ORDER.indexOf(a.kind) - MAP_ORDER.indexOf(b.kind)) : [];

  return (
    <div className="overhead replay">
      <div className="overhead-toolbar">
        <select
          className="run-select"
          value={rid ?? ""}
          onFocus={onRefreshRuns}
          onChange={(e) => e.target.value && onOpenRun(e.target.value)}
          title="Recorded failure runs (saved on disk)"
        >
          {!rid && <option value="">no run yet</option>}
          {rid && !savedRuns.some((r) => r.run_id === rid) && <option value={rid}>{rid}</option>}
          {savedRuns.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} · {r.duration_s}s{r.staged ? " · staged" : ""}
            </option>
          ))}
        </select>
        <div className="seg">
          <button
            className={view === "policy" ? "on" : ""}
            onClick={() => setView("policy")}
            title="What the policy saw (with any staging or suppression)"
          >
            Policy view
          </button>
          <button className={view === "raw" ? "on" : ""} onClick={() => setView("raw")} title="Unmodified camera">
            Raw camera
          </button>
        </div>
        <div className="seg">
          {maps.map((m) => (
            <button
              key={m.kind}
              className={m.kind === mapKind ? "on" : ""}
              onClick={() => onMapKind(m.kind)}
              title={`${m.title}: ${m.description}`}
              disabled={!m.keyframes.length}
            >
              {mapLabel(m.kind, m.title)}
              <span className="kf-count">
                {m.keyframes.length}
                {m.expected_keyframes > m.keyframes.length ? `/${m.expected_keyframes}` : ""}
              </span>
            </button>
          ))}
        </div>
        <label className="toggle">
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Attention
        </label>
        <input
          className="opacity"
          type="range"
          min={0.2}
          max={1}
          step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          title="Heatmap opacity"
          disabled={!showHeat}
        />
        <label className="toggle">
          <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} /> SAM masks
        </label>
        <span className="spacer" />
        <span className="tag scrub">⏪ REPLAY</span>
      </div>
      {analysing && (
        <div className="analysis-progress" title={replay.progress_text}>
          <span style={{ width: `${Math.round(replay.progress * 100)}%` }} />
          <em>
            <span className="spinner tiny" /> {replay.progress_text || replay.status} · {Math.round(replay.progress * 100)}%
          </em>
        </div>
      )}
      <div className="overhead-stage">
        <canvas ref={canvasRef} width={640} height={360} />
        {!replay && (
          <div className="stage-msg empty-replay">
            <b>No failure recorded yet</b>
            <span>
              When the policy fails, the seconds leading up to it are saved and replayed here, frame by frame, with ACT's
              attention, SAM 3.1 masks, an observe-only lens (SAM 3 or SmolVLA) and a Muse Wonder breakdown.
            </span>
          </div>
        )}
        {replay && preloadPct < 100 && <div className="stage-msg">Loading frames {preloadPct}%…</div>}
        {replay?.staging && view === "policy" && (
          <div
            className="stage-corner staging"
            title="The distractor was moved onto the goal in the policy's input for the demo. Switch to Raw camera to see the real scene."
          >
            DEMO STAGING · {replay.staging.label} on the goal in the policy's view
          </div>
        )}
        {!!live?.length && (
          <div className="event-flashes">
            {live.map((e) => (
              <div key={`${e.kind}-${e.frame}`} className={`event-flash ${e.kind}`}>
                {e.kind === "attention_shift" ? "⚠ " : ""}
                {e.label}
              </div>
            ))}
          </div>
        )}
        {replay && frame >= replay.failure_frame && (
          <div className="stage-badge estop">
            {frame === replay.failure_frame ? "FAILURE · anomaly detected here" : "AFTER THE FAILURE · arm on hold"}
          </div>
        )}
        {replay && (
          <div className="replay-hud mono">
            t = {t >= 0 ? "+" : ""}
            {t.toFixed(2)} s · {info?.phase ?? "-"} · {map ? interp : "attention off"}
          </div>
        )}
      </div>
      <div className="transport">
        <button className="btn small" onClick={() => jumpKeyframe(-1)} disabled={!replay} title="Previous keyframe (Shift+←)">
          ⏮
        </button>
        <button className="btn small play" onClick={togglePlay} disabled={!replay} title="Play / pause (Space)">
          {playing ? "❚❚" : "▶"}
        </button>
        <button className="btn small" onClick={() => jumpKeyframe(1)} disabled={!replay} title="Next keyframe (Shift+→)">
          ⏭
        </button>
        <div className="seg">
          {SPEEDS.map((s) => (
            <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>
              {s}×
            </button>
          ))}
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, n - 1)}
          step={0.01}
          value={pos}
          disabled={!replay}
          onChange={(e) => {
            setPlaying(false);
            onPos(Number(e.target.value));
          }}
        />
        <span className="mono small muted">
          {frame + 1}/{n || "-"}
        </span>
      </div>
    </div>
  );
}
