// Drawing helpers shared by the live canvas and the replay player.

export const LABEL_COLORS: Record<string, string> = {
  target: "#3ddc84",
  gripper: "#4aa3ff",
  hand: "#ffb020",
  obstacle: "#c58cff",
  goal: "#28d0c8",
};
export const HIJACK_COLOR = "#ff4d4d";
export const DISTRACTOR_COLOR = "#ffd166";
export const SUPPRESSED_COLOR = "#ff3df2";
export const PATH_COLOR = "#33e1ff";
export const NON_DISTRACTOR = new Set(["target", "gripper", "goal", "hand", "obstacle"]);

// Jet colormap with saliency-dependent alpha, as RGBA bytes for 256 levels.
export const JET_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    lut[i * 4] = 255 * clamp(1.5 - Math.abs(4 * v - 3));
    lut[i * 4 + 1] = 255 * clamp(1.5 - Math.abs(4 * v - 2));
    lut[i * 4 + 2] = 255 * clamp(1.5 - Math.abs(4 * v - 1));
    lut[i * 4 + 3] = v < 0.06 ? 0 : 255 * Math.min(0.72, 0.15 + v * 0.9);
  }
  return lut;
})();

export function decodeB64(data: string): Uint8Array {
  const bin = atob(data);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Colour for a label in charts and contours; distractors get a stable warm hue. */
export function labelColor(label: string, hijack = false): string {
  if (LABEL_COLORS[label]) return LABEL_COLORS[label];
  return hijack ? HIJACK_COLOR : DISTRACTOR_COLOR;
}

export function tracePolygons(ctx: CanvasRenderingContext2D, polygons: number[][][]) {
  ctx.beginPath();
  for (const poly of polygons) {
    if (poly.length < 2) continue;
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
    ctx.closePath();
  }
}

export function drawTag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, scale: number) {
  ctx.font = `600 ${Math.round(12 * scale)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  const w = ctx.measureText(text).width + 8 * scale;
  const h = 16 * scale;
  const ty = Math.max(0, y - h);
  ctx.fillStyle = "rgba(10, 14, 20, 0.78)";
  ctx.fillRect(x, ty, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 4 * scale, ty + 12 * scale);
}
