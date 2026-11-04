import { useCallback, useEffect, useRef, useState } from "react";
import { bufferFrameUrl } from "../api";
import type { LiveFrame } from "../useTelemetry";
import type { AnalysisResult, FrameMeta, LiveObject, RuntimeStatus, SaliencyGrid } from "../types";
import {
  DISTRACTOR_COLOR,
  HIJACK_COLOR,
  JET_LUT,
  LABEL_COLORS,
  PATH_COLOR,
  SUPPRESSED_COLOR,
  decodeB64,
  drawTag,
  tracePolygons,
} from "../viz";

interface Props {
  frameRef: React.MutableRefObject<LiveFrame | null>;
  subscribeFrame: (fn: () => void) => () => void;
  setView: (v: "policy" | "raw") => void;
  status: RuntimeStatus | null;
  analysis: AnalysisResult | null;
  scrubIndex: number | null; // ring-buffer frame index when scrubbing, null when live
  connected: boolean;
}

function objectColor(o: { label: string }, share: number, thr: number): string {
  if (LABEL_COLORS[o.label]) return LABEL_COLORS[o.label];
  return share > thr ? HIJACK_COLOR : DISTRACTOR_COLOR;
}

export default function OverheadCanvas({ frameRef, subscribeFrame, setView, status, analysis, scrubIndex, connected }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatRef = useRef<HTMLCanvasElement | null>(null);
  const heatKey = useRef<string | null>(null);
  const scrubImg = useRef<HTMLImageElement | null>(null);
  const raf = useRef<number | null>(null);
  const [view, setViewState] = useState<"policy" | "raw">("policy");
  const [showHeat, setShowHeat] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [hud, setHud] = useState<{ fps: number; method: string; suppressed: boolean } | null>(null);

  // Everything the draw routine reads, kept in a ref so frame callbacks never see stale props.
  const opts = useRef({ showHeat, showMasks, showPath, status, analysis, scrubIndex });
  opts.current = { showHeat, showMasks, showPath, status, analysis, scrubIndex };

  const drawHeat = useCallback((ctx: CanvasRenderingContext2D, grid: SaliencyGrid, W: number, H: number) => {
    let heat = heatRef.current;
    if (!heat) heat = heatRef.current = document.createElement("canvas");
    if (heatKey.current !== grid.data) {
      // The saliency map changes a few times a second at most; only re-colour it when it does.
      heatKey.current = grid.data;
      if (heat.width !== grid.w || heat.height !== grid.h) {
        heat.width = grid.w;
        heat.height = grid.h;
      }
      const hctx = heat.getContext("2d")!;
      const vals = decodeB64(grid.data);
      const img = hctx.createImageData(grid.w, grid.h);
      for (let i = 0; i < vals.length; i++) img.data.set(JET_LUT.subarray(vals[i] * 4, vals[i] * 4 + 4), i * 4);
      hctx.putImageData(img, 0, 0);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat, 0, 0, W, H);
  }, []);

  const drawLiveObjects = useCallback((ctx: CanvasRenderingContext2D, objects: LiveObject[], thr: number, scale: number) => {
    for (const o of objects) {
      if (!o.polygons.length) continue;
      const color = objectColor(o, o.attention_share, thr);
      tracePolygons(ctx, o.polygons);
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = color;
      ctx.setLineDash([]);
      ctx.stroke();
      const flag = o.label !== "gripper" && o.label !== "target" && o.attention_share > thr ? " HIJACK" : "";
      drawTag(ctx, `#${o.mask_id} ${o.label} A=${o.attention_share.toFixed(2)}${flag}`, o.bbox[0], o.bbox[1] - 2, color, scale);
    }
  }, []);

  const drawOverlays = useCallback(
    (ctx: CanvasRenderingContext2D, meta: FrameMeta | null, W: number) => {
      const { showMasks, showPath, status, analysis } = opts.current;
      const scale = Math.max(1, W / 640);
      const thr = status?.thresholds.attention_hijack ?? 0.45;

      if (showMasks && meta) drawLiveObjects(ctx, meta.objects, thr, scale);

      // Suppressed masks: outline the SAM polygons of the analysis that produced them.
      const suppressed = meta?.suppression.objects ?? status?.suppression ?? [];
      if (suppressed.length && analysis) {
        for (const s of suppressed) {
          const obj = analysis.objects.find((o) => o.mask_id === s.mask_id);
          if (!obj) continue;
          tracePolygons(ctx, obj.polygons);
          ctx.lineWidth = 2.5 * scale;
          ctx.strokeStyle = SUPPRESSED_COLOR;
          ctx.setLineDash([8 * scale, 5 * scale]);
          ctx.stroke();
          ctx.setLineDash([]);
          drawTag(ctx, `SUPPRESSED #${s.mask_id} ${s.label}`, obj.bbox[0], obj.bbox[3] + 18 * scale, SUPPRESSED_COLOR, scale);
        }
      }

      const path = analysis?.plan_path_px;
      if (showPath && path && path.length > 1) {
        ctx.lineWidth = 3 * scale;
        ctx.strokeStyle = PATH_COLOR;
        ctx.setLineDash([10 * scale, 6 * scale]);
        ctx.beginPath();
        ctx.moveTo(path[0][0], path[0][1]);
        for (const p of path.slice(1)) ctx.lineTo(p[0], p[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        path.forEach((p, i) => {
          ctx.beginPath();
          ctx.arc(p[0], p[1], (i === 0 ? 4 : 6) * scale, 0, Math.PI * 2);
          ctx.fillStyle = i === 0 ? "#ffffff" : PATH_COLOR;
          ctx.fill();
          if (i > 0) drawTag(ctx, `WP${i}`, p[0] + 8 * scale, p[1] - 4 * scale, PATH_COLOR, scale);
        });
      }
    },
    [drawLiveObjects],
  );

  const draw = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const { scrubIndex, showHeat } = opts.current;

    if (scrubIndex !== null) {
      const img = scrubImg.current;
      if (!img || !img.complete || !img.naturalWidth) return;
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      ctx.drawImage(img, 0, 0);
      drawOverlays(ctx, null, canvas.width);
      return;
    }

    const frame = frameRef.current;
    if (!frame) return;
    const { bitmap, meta } = frame;
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    ctx.drawImage(bitmap, 0, 0);
    if (showHeat && meta.saliency) drawHeat(ctx, meta.saliency, canvas.width, canvas.height);
    drawOverlays(ctx, meta, canvas.width);
  }, [drawHeat, drawOverlays, frameRef]);

  const drawRef = useRef(draw);
  drawRef.current = draw;
  const schedule = useCallback(() => {
    if (raf.current === null) raf.current = requestAnimationFrame(() => drawRef.current());
  }, []);

  useEffect(() => {
    let last = 0;
    const unsub = subscribeFrame(() => {
      if (opts.current.scrubIndex !== null) return;
      schedule();
      const now = performance.now();
      const meta = frameRef.current?.meta;
      if (meta && now - last > 500) {
        last = now;
        setHud({ fps: meta.fps, method: meta.saliency?.method ?? "none", suppressed: meta.suppression.active });
      }
    });
    return () => {
      unsub();
      if (raf.current !== null) cancelAnimationFrame(raf.current);
      raf.current = null;
    };
  }, [subscribeFrame, schedule, frameRef]);

  // Scrub mode: fetch the stored frame (heatmap baked in server-side when enabled).
  useEffect(() => {
    if (scrubIndex === null) {
      scrubImg.current = null;
      schedule();
      return;
    }
    const img = new Image();
    img.onload = () => {
      if (scrubImg.current === img) schedule();
    };
    img.src = bufferFrameUrl(scrubIndex, showHeat);
    scrubImg.current = img;
  }, [scrubIndex, showHeat, schedule]);

  // Redraw on overlay toggles / new analysis even if the stream is paused.
  useEffect(() => schedule(), [showMasks, showPath, showHeat, analysis, status?.suppression, schedule, draw]);

  const changeView = (v: "policy" | "raw") => {
    setViewState(v);
    setView(v);
  };

  const live = scrubIndex === null;
  return (
    <div className="overhead">
      <div className="overhead-toolbar">
        <div className="seg">
          <button className={view === "policy" ? "on" : ""} onClick={() => changeView("policy")} title="What the policy sees (after mask suppression)">
            Policy view
          </button>
          <button className={view === "raw" ? "on" : ""} onClick={() => changeView("raw")} title="Unmodified camera">
            Raw camera
          </button>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Saliency
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} /> Masks + A(i)
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} /> Recovery path
        </label>
        <span className="spacer" />
        {live ? (
          <span className="tag live">● LIVE {hud ? `${hud.fps.toFixed(0)} fps` : ""}</span>
        ) : (
          <span className="tag scrub">⏪ REPLAY frame {scrubIndex}</span>
        )}
      </div>
      <div className="overhead-stage">
        <canvas ref={canvasRef} width={status?.frame_size[0] ?? 640} height={status?.frame_size[1] ?? 360} />
        {!connected && <div className="stage-msg">Connecting to the GlassBox backend…</div>}
        {connected && live && !frameRef.current && <div className="stage-msg">Waiting for camera frames…</div>}
        {live && hud?.suppressed && view === "policy" && (
          <div className="stage-badge suppress">Real-time mask suppression active: the policy cannot see the erased object</div>
        )}
        {status?.staging && view === "policy" && (
          <div className={`stage-corner staging ${status.staging.suppressed ? "off" : ""}`} title="Demo staging: the distractor is moved onto the goal in the policy's input only. The raw camera shows the real scene.">
            DEMO STAGING · {status.staging.label} {status.staging.suppressed ? "erased by GlassBox" : "on the goal (policy view)"}
          </div>
        )}
        {status?.preflight && <div className="stage-badge hold">PRE-FLIGHT: SAM 3.1 is scanning the scene</div>}
        {status?.state === "HOLD" && !status.preflight && (
          <div className="stage-badge hold">{status.awaiting_start ? "READY: press ▶ Start policy" : "HOLD: policy paused"}</div>
        )}
        {status?.state === "ESTOP" && <div className="stage-badge estop">E-STOP</div>}
        {status?.state === "COMPLETE" && <div className="stage-badge done">Task complete</div>}
      </div>
      <div className="overhead-legend">
        <span><i style={{ background: LABEL_COLORS.target }} />target</span>
        <span><i style={{ background: LABEL_COLORS.gripper }} />gripper</span>
        <span><i style={{ background: DISTRACTOR_COLOR }} />distractor</span>
        <span><i style={{ background: HIJACK_COLOR }} />A &gt; {status?.thresholds.attention_hijack ?? 0.45}</span>
        <span><i style={{ background: SUPPRESSED_COLOR }} />suppressed</span>
        <span><i style={{ background: PATH_COLOR }} />Wonder path</span>
        <span className="muted">saliency: {hud?.method ?? status?.saliency_method ?? "-"}</span>
      </div>
    </div>
  );
}
