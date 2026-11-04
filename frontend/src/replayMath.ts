// Keyframe interpolation and lookups shared by the replay player, the timeline and the breakdown card.
import type { ReplayKeyframe, ReplayMap, ReplayResult, ReplaySegmentation } from "./types";

// The policy that ran comes before SAM 3's perception maps; SmolVLA (when it is the lens) before both.
export const MAP_ORDER = ["vla_language", "vla_action", "act", "sam3_target", "sam3_goal"];

/** The concept a SAM 3 perception map is prompted with ("target", "goal"), or null for attention maps. */
export function groundedLabel(kind: string | null | undefined): string | null {
  return kind?.startsWith("sam3_") ? kind.slice(5) : null;
}

/** The phrase in a SAM 3 map's title ("SAM 3 · 'red cube'" -> "red cube"). */
export function mapPhrase(title: string): string | null {
  return /'(.+)'/.exec(title)?.[1] ?? null;
}

/** Keyframes sorted by frame (they arrive in bisect order while the replay refines). */
export function sortedKeyframes(m: ReplayMap | undefined): ReplayKeyframe[] {
  return m ? [...m.keyframes].sort((a, b) => a.frame - b.frame) : [];
}

/** The two keyframes around ``pos`` and the blend weight of the second. */
export function bracket(kfs: ReplayKeyframe[], pos: number): { a: ReplayKeyframe; b: ReplayKeyframe; w: number } | null {
  if (!kfs.length) return null;
  if (pos <= kfs[0].frame) return { a: kfs[0], b: kfs[0], w: 0 };
  const last = kfs[kfs.length - 1];
  if (pos >= last.frame) return { a: last, b: last, w: 0 };
  let i = 0;
  while (kfs[i + 1].frame < pos) i++;
  const a = kfs[i];
  const b = kfs[i + 1];
  return { a, b, w: (pos - a.frame) / Math.max(1e-6, b.frame - a.frame) };
}

export type ShareMeasure = "shares" | "near" | "near_lift";

/** True when every keyframe has the attention near each object (replays saved before it have only ``shares``). */
export function hasNear(kfs: ReplayKeyframe[]): boolean {
  return kfs.length > 0 && kfs.every((k) => k.near && Object.keys(k.near).length > 0);
}

/** Attention shares at ``pos``, linearly interpolated between keyframes. */
export function sharesAt(kfs: ReplayKeyframe[], pos: number, measure: ShareMeasure = "shares"): Record<string, number> {
  const br = bracket(kfs, pos);
  if (!br) return {};
  const a = br.a[measure] ?? {};
  const b = br.b[measure] ?? {};
  const out: Record<string, number> = {};
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    out[k] = (a[k] ?? 0) * (1 - br.w) + (b[k] ?? 0) * br.w;
  }
  return out;
}

export function frameAtTime(replay: ReplayResult, t: number): number {
  let best = 0;
  replay.frames.forEach((f, i) => {
    if (Math.abs(f.t - t) < Math.abs(replay.frames[best].t - t)) best = i;
  });
  return best;
}

export function nearestSegmentation(segs: ReplaySegmentation[], frame: number): ReplaySegmentation | null {
  let best: ReplaySegmentation | null = null;
  for (const s of segs) if (!best || Math.abs(s.frame - frame) < Math.abs(best.frame - frame)) best = s;
  return best;
}

export function mapLabel(kind: string, title?: string): string {
  if (groundedLabel(kind)) return title ?? `SAM 3 · ${groundedLabel(kind)}`;
  return kind === "vla_language"
    ? "SmolVLA · grounding"
    : kind === "vla_action"
      ? "SmolVLA · action"
      : kind === "act"
        ? "ACT · live"
        : kind;
}
