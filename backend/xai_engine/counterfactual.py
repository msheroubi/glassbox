"""Mask-guided counterfactual frames: the scene the policy would have seen without the culprit.

Muse Image edits take an instruction but no mask, and a free-form "remove the book" edit tends to
redraw the whole frame (and sometimes the object). So the region comes from SAM instead:

1. ``removal_region`` grows SAM's mask over what SAM leaves out but is still part of the object in
   the picture: its dark side faces, edges and contact shadow.
2. ``erase_local`` fills that region with the background plate (or Telea inpainting), so the object
   is already gone before Muse sees the frame; Muse is only asked to make the smudge look like table.
3. ``composite`` pastes Muse's pixels back inside the feathered region only, so everything the policy
   saw outside it (arm, cube, tile, lighting) is the original frame, pixel for pixel.
4. ``culprit_still_visible`` rejects a result where the culprit's colour survives, in which case the
   local erase is used.

Then the policy is asked the counterfactual question: the same joint state, the edited frame; where does
it look (``compare_maps``) and where does its predicted reach go? ``verdict`` puts the answer in words,
including when removing the suspect changes nothing, which rules it out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.schemas import CounterfactualMap
from backend.xai_engine.attribution import (
    NON_DISTRACTOR_LABELS, SegmentedObject, grid_b64, label_shares, neighbourhoods,
)

GROW_WINDOW = 0.35  # look this far around the mask (fraction of the object's size) for parts SAM missed
MAX_GROWTH = 2.5  # never let the grown region exceed this multiple of the dilated mask's area


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def removal_region(
    frame: np.ndarray, mask: np.ndarray, plate: np.ndarray | None = None, keep_out: np.ndarray | None = None
) -> np.ndarray:
    """Bool mask of everything to erase so ``mask``'s object is gone, shadows and missed edges included.

    ``keep_out`` (e.g. the target and the arm) is never included."""
    m = mask.astype(bool)
    if not m.any():
        return m
    h, w = m.shape
    size = float(np.sqrt(m.sum()))
    base = cv2.dilate(m.astype(np.uint8), _disk(max(4, round(0.1 * size)))).astype(bool)

    ys, xs = np.nonzero(m)
    pad = int(round(GROW_WINDOW * size))
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)
    win = (slice(y0, y1), slice(x0, x1))
    lab = _lab(frame[win])

    if plate is not None and plate.shape == frame.shape:
        dist = np.linalg.norm(lab - _lab(plate[win]), axis=2)
        differs = dist > 18
    else:
        # What empty table looks like here: the colour of a ring around the object (robust to vignetting
        # because the ring is local), excluding pixels that belong to other objects.
        ring = cv2.dilate(m.astype(np.uint8), _disk(pad + 4)).astype(bool) & ~cv2.dilate(
            m.astype(np.uint8), _disk(max(6, pad // 2))
        ).astype(bool)
        if keep_out is not None:
            ring &= ~keep_out
        ring_px = _lab(frame)[ring]
        if len(ring_px) < 20:
            return base
        ref = np.median(ring_px, axis=0)
        spread = np.median(np.linalg.norm(ring_px - ref, axis=1))
        dist = np.linalg.norm(lab - ref, axis=2)
        differs = dist > max(22.0, 3.0 * spread)

    cand = np.zeros_like(m)
    cand[win] = differs
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)).astype(bool) | base
    if keep_out is not None:
        cand &= ~keep_out
    # Keep only what touches the object itself.
    n, comp = cv2.connectedComponents(cand.astype(np.uint8))
    touching = np.unique(comp[base & cand])
    grown = np.isin(comp, touching[touching > 0])
    grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, _disk(max(3, round(0.08 * size))))
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(grown)
    cv2.drawContours(filled, contours, -1, 1, -1)
    region = cv2.dilate(filled, _disk(3)).astype(bool) | base
    if keep_out is not None:
        region &= ~(keep_out & ~m)
    if region.sum() > MAX_GROWTH * base.sum():
        return base
    return region


def feather(region: np.ndarray, px: float = 4.0) -> np.ndarray:
    """Float (h, w, 1) alpha that is 1 inside ``region`` and fades out over ~``px`` pixels outside it."""
    grown = cv2.dilate(region.astype(np.uint8), _disk(max(1, round(px))))
    return np.clip(cv2.GaussianBlur(grown.astype(np.float32), (0, 0), px / 2) * 1.0, 0, 1)[..., None]


def erase_local(frame: np.ndarray, region: np.ndarray, plate: np.ndarray | None = None) -> np.ndarray:
    """``frame`` with ``region`` filled by the background plate, or inpainted when there is none."""
    if not region.any():
        return frame.copy()
    if plate is not None and plate.shape == frame.shape:
        fill = plate
    else:
        fill = cv2.inpaint(frame, region.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    return composite(frame, fill, region)


def composite(original: np.ndarray, edited: np.ndarray, region: np.ndarray, px: float = 4.0) -> np.ndarray:
    """``edited`` inside ``region`` (feathered), ``original`` everywhere else."""
    if edited.shape[:2] != original.shape[:2]:
        edited = cv2.resize(edited, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    a = feather(region, px)
    return (original.astype(np.float32) * (1 - a) + edited.astype(np.float32) * a).round().astype(np.uint8)


def culprit_still_visible(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> bool:
    """True if the object's colour survives in ``result``: too many of its pixels still match it."""
    m = mask.astype(bool)
    if m.sum() < 30:
        return False
    a, b = _lab(original)[m], _lab(result)[m]
    ref = np.median(a, axis=0)
    # Pixels that still look like the object (close to its median colour) where they used to.
    was = np.linalg.norm(a - ref, axis=1) < 25
    still = np.linalg.norm(b - ref, axis=1) < 25
    return bool(was.sum() > 0 and (still & was).sum() / was.sum() > 0.25)


def align(edited: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Resize ``edited`` to ``original`` and undo the small shift/scale image editors introduce (ECC affine)."""
    h, w = original.shape[:2]
    out = cv2.resize(edited, (w, h), interpolation=cv2.INTER_AREA) if edited.shape[:2] != (h, w) else edited

    def gray(x):
        return cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32) / 255

    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, warp = cv2.findTransformECC(gray(original), gray(out), warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return out
    if np.abs(warp[:, :2] - np.eye(2)).max() > 0.05 or np.abs(warp[:, 2]).max() > 0.05 * w:
        return out  # not a small correction: don't trust it
    return cv2.warpAffine(out, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def looks_like_table(result: np.ndarray, region: np.ndarray) -> bool:
    """False if the edit put something in ``region`` (e.g. a new object) instead of bare table: the region
    must match the colour of the ring of table right around it."""
    r = region.astype(bool)
    if r.sum() < 30:
        return True
    size = float(np.sqrt(r.sum()))
    ring = cv2.dilate(r.astype(np.uint8), _disk(max(6, round(0.3 * size)))).astype(bool) & ~cv2.dilate(
        r.astype(np.uint8), _disk(3)).astype(bool)
    lab = _lab(result)
    ring_px = lab[ring]
    ref = np.median(ring_px, axis=0)
    spread = np.median(np.linalg.norm(ring_px - ref, axis=1))
    off = np.linalg.norm(lab[r] - ref, axis=1) > max(25.0, 3.0 * spread)
    return bool(off.mean() < 0.15)


def remove_glare(frame: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Inpaint the specular (bright, colourless) pixels inside ``region``."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    spec = region.astype(bool) & (hsv[..., 2] > 225) & (hsv[..., 1] < 70)
    if not spec.any():
        return frame.copy()
    spec = cv2.dilate(spec.astype(np.uint8), _disk(2))
    return cv2.inpaint(frame, spec, 5, cv2.INPAINT_TELEA)



# ====================================================================================== what to edit
@dataclass
class EditPlan:
    region: np.ndarray  # the only pixels the edit may change
    culprit_mask: np.ndarray | None  # must be gone afterwards (None for glare: the target stays)
    pre_erase: bool  # fill the region locally before Muse sees the frame
    removed: str  # "the red book"
    culprit_label: str | None
    bbox: list[int]


def choose_edit(
    failure_type: str | None, objects: list[SegmentedObject], suspect: SegmentedObject | None,
) -> tuple[str, SegmentedObject | None] | None:
    """Which counterfactual to run, ``(failure_type, culprit)``, or ``None``.

    A distractor in the scene is what gets removed, whatever the diagnosis's first guess was: from a single
    frame it often calls the book's sheen "glare on the target", and repainting glare tests nothing about the
    book. ``suspect`` is the diagnosis's culprit; when that isn't a distractor the biggest distractor is used.
    A hand (occlusion) is masked out instead; glare without any distractor de-glares the target."""
    distractors = [o for o in objects if o.label not in NON_DISTRACTOR_LABELS and o.label != "hand" and o.mask.any()]
    if suspect is not None and suspect in distractors:
        culprit = suspect
    else:
        culprit = max(distractors, key=lambda o: int(o.mask.sum()), default=None)
    if failure_type == "OCCLUSION_OOD" and any(o.label == "hand" for o in objects):
        return "OCCLUSION_OOD", suspect
    if culprit is not None:
        return ("SERVO_STALL_COLLISION" if failure_type == "SERVO_STALL_COLLISION" else "ATTENTION_HIJACK"), culprit
    if failure_type == "SPECULAR_GLARE":
        return "SPECULAR_GLARE", None
    return None


def plan_edit(
    failure_type: str, frame: np.ndarray, objects: list[SegmentedObject], culprit: SegmentedObject | None,
    plate: np.ndarray | None = None,
) -> EditPlan | None:
    """What a counterfactual for this failure removes, and where. ``None`` when there is nothing to edit."""
    target = next((o for o in objects if o.label == "target"), None)
    if failure_type in ("ATTENTION_HIJACK", "SERVO_STALL_COLLISION"):
        if culprit is None or culprit.label in NON_DISTRACTOR_LABELS or not culprit.mask.any():
            return None
        others = [o.mask for o in objects if o is not culprit]
        keep = np.logical_or.reduce(others) if others else None
        region = removal_region(frame, culprit.mask, plate, keep)
        return EditPlan(region, culprit.mask, True, f"the {culprit.prompt}", culprit.label, culprit.bbox)
    if failure_type == "OCCLUSION_OOD":
        hand = culprit if culprit is not None and culprit.label == "hand" else next((o for o in objects if o.label == "hand"), None)
        if hand is None or not hand.mask.any():
            return None
        size = float(np.sqrt(hand.mask.sum()))
        region = cv2.dilate(hand.mask.astype(np.uint8), _disk(max(6, round(0.12 * size)))).astype(bool)
        return EditPlan(region, hand.mask, False, "the hand", hand.label, hand.bbox)
    if failure_type == "SPECULAR_GLARE":
        # Only the target: "de-glaring" a distractor just repaints it, which tests nothing.
        if target is None or not target.mask.any():
            return None
        region = cv2.dilate(target.mask.astype(np.uint8), _disk(5)).astype(bool)
        return EditPlan(region, None, False, f"the glare on the {target.prompt}", None, target.bbox)
    return None


def region_polygons(region: np.ndarray) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in contours if cv2.contourArea(c) >= 20]


# ====================================================================================== what changed
def compare_maps(
    kind: str, title: str, before: np.ndarray, after: np.ndarray, objects: list[SegmentedObject],
    grid_wh: tuple[int, int],
) -> CounterfactualMap:
    """Attention before vs after the edit, measured in and around the same (pre-edit) masks.

    Besides the shares inside each mask: the share near each object, and where the attention that moved
    came from (``left_from``) and went to (``arrived_at``). A small total move that all comes from around the
    culprit is the policy's attention letting go of it, which the overall number alone hides."""
    p = before.astype(np.float64) / (before.sum() + 1e-9)
    q = after.astype(np.float64) / (after.sum() + 1e-9)
    if p.shape != q.shape:
        q = cv2.resize(q, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
        q /= q.sum() + 1e-9
    m = CounterfactualMap(
        kind=kind, title=title, grid_w=grid_wh[0], grid_h=grid_wh[1],
        grid_before=grid_b64(before, grid_wh), grid_after=grid_b64(after, grid_wh),
        shares_before=label_shares(before, objects), shares_after=label_shares(after, objects),
        moved=round(float(0.5 * np.abs(p - q).sum()), 3),
    )
    nb = neighbourhoods(objects)
    if nb is None:
        return m
    pw, qw = nb.fit(p), nb.fit(q)
    d = pw - qw
    left, arrived = np.clip(d, 0, None), np.clip(-d, 0, None)
    total = float(left.sum()) + 1e-12
    m.near_before, m.near_after, m.area = nb.sums(pw), nb.sums(qw), nb.area()
    m.left_from = {lab: round(v / total, 3) for lab, v in nb.sums(left).items()}
    m.arrived_at = {lab: round(v / total, 3) for lab, v in nb.sums(arrived).items()}
    for name, field in (("peak_before", pw), ("peak_after", qw)):
        y, x = np.unravel_index(int(field.argmax()), field.shape)
        setattr(m, name, nb.label_at(int(x), int(y)))
    return m


def path_shift(before: list | None, after: list | None) -> float | None:
    if not before or not after:
        return None
    return round(float(np.hypot(before[-1][0] - after[-1][0], before[-1][1] - after[-1][1])), 1)


MOVED_LITTLE = 0.15  # below this fraction of attention moving (and a near-identical path), the edit didn't matter
PATH_LITTLE_PX = 15.0
MOVED_LOT = 0.25
# The change is concentrated around an object when at least CONCENTRATED_MIN of the attention that moved left
# (or reached) its neighbourhood, at CONCENTRATED_LIFT times the rate a change spread evenly over the frame would.
CONCENTRATED_MIN = 0.2
CONCENTRATED_LIFT = 2.5
TASK_LABELS = ("target", "goal")


@dataclass
class Assessment:
    outcome: str  # confirmed | implicated | partial | ruled_out
    pulled: float | None  # fraction of the moved attention that left the culprit's surroundings
    pull_lift: float | None  # ... over the neighbourhood's area fraction (1 = no more than anywhere else)
    peak_on_culprit: bool  # the map's hottest spot was next to the culprit, and isn't once it's gone
    path_moved: bool


def _lift(frac: float | None, area: float | None) -> float | None:
    return None if frac is None or not area else frac / area


def concentrated(m: CounterfactualMap, label: str, side: str = "left") -> tuple[float, float] | None:
    """``(fraction, lift)`` if the move is concentrated around ``label`` (``side``: left from / arrived at)."""
    frac = (m.left_from if side == "left" else m.arrived_at).get(label)
    lift = _lift(frac, m.area.get(label))
    if frac is None or lift is None or frac < CONCENTRATED_MIN or lift < CONCENTRATED_LIFT:
        return None
    return frac, lift


def assess(m: CounterfactualMap | None, culprit_label: str | None, shift_px: float | None) -> Assessment | None:
    """What one model's counterfactual says about the culprit.

    * confirmed: removing it changes a lot of the attention or moves the predicted reach.
    * implicated: little changes overall, but what does change is the attention that sat around the culprit
      (or the hottest spot was beside it): the culprit was drawing the policy's attention.
    * ruled_out: little changes and none of it is concentrated around the culprit.
    """
    if m is None:
        return None
    lab = culprit_label or ""
    pulled = concentrated(m, lab, "left")
    frac = m.left_from.get(lab)
    peak = bool(lab) and m.peak_before == lab and m.peak_after != lab
    path_moved = shift_px is not None and shift_px >= PATH_LITTLE_PX
    if m.moved >= MOVED_LOT or path_moved:
        outcome = "confirmed"
    elif pulled or peak:
        outcome = "implicated"
    elif m.moved < MOVED_LITTLE:
        outcome = "ruled_out"
    else:
        outcome = "partial"
    return Assessment(outcome, frac, _lift(frac, m.area.get(lab)), peak, path_moved)


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _names(culprit_label: str | None, removed: str, target_name: str) -> dict[str, str]:
    return {culprit_label or "": removed, "target": f"the {target_name}", "goal": "the goal tile", "gripper": "the arm"}


def _pct(v: float) -> str:
    return f"{v:.1%}" if v < 0.095 else f"{v:.0%}"


def _near(m: CounterfactualMap, label: str, name: str) -> str | None:
    a, b = m.near_before.get(label), m.near_after.get(label)
    if a is None or b is None:
        return None
    return f"attention near {name} {_pct(a)} → {_pct(b)}"


def _flow(m: CounterfactualMap, names: dict[str, str]) -> list[str]:
    """Where the moved attention concentrated, e.g. ["46% of it from around the red book (6.8x its area)"]."""
    out = []
    culprit = next(iter(names))
    for side, word in (("left", "from around"), ("arrived", "to around")):
        for lab in sorted(m.area, key=lambda k: -(m.left_from if side == "left" else m.arrived_at).get(k, 0.0)):
            hit = concentrated(m, lab, side)
            if hit:
                where = f"to where {names[lab]} was" if side == "arrived" and lab == culprit else (
                    f"{word} {names.get(lab, 'the ' + lab)}")
                out.append(f"{hit[0]:.0%} of it {where} ({hit[1]:.1f}x its area)")
    return out


def verdict(
    removed: str, m: CounterfactualMap | None, culprit_label: str | None, shift_px: float | None,
    reach_to_target_px: tuple[float, float] | None = None, target_name: str = "red cube",
) -> str:
    """A few sentences on what the counterfactual shows about the policy."""
    a = assess(m, culprit_label, shift_px)
    if m is None or a is None:
        return f"Removed {removed}; no attention map was available to compare."
    who = m.title.split(" · ")[0]
    names = _names(culprit_label, removed, target_name)
    reach = f" and its predicted reach ends {shift_px:.0f} px from where it did" if shift_px is not None else ""
    if a.outcome == "ruled_out":
        return (f"Removing {removed} barely changes what {who} does: only {m.moved:.0%} of its attention moves, none of "
                f"it concentrated around {removed}{reach}. {_cap(removed)} is not what the policy was reacting to here.")

    text = f"Without {removed}, {m.moved:.0%} of {who}'s attention moves"
    flow = _flow(m, names)
    if a.pulled is not None and a.pull_lift is not None and not any(f" {removed} (" in f for f in flow):
        flow.insert(0, f"{a.pulled:.0%} of it from around {removed} ({a.pull_lift:.1f}x its area)")
    text += (": " + ", ".join(flow) + ".") if flow else "."
    if a.peak_on_culprit:
        where = f"near {names.get(m.peak_after, 'the ' + m.peak_after)}" if m.peak_after else "away from every object"
        text += f" Its hottest spot was right next to {removed}; without it, the peak moves {where}."
    near = [s for s in (_near(m, culprit_label or "", removed), _near(m, "target", names["target"])) if s]
    if not near:  # no neighbourhoods (no masks): the shares inside the masks
        c0, c1 = m.shares_before.get(culprit_label or ""), m.shares_after.get(culprit_label or "")
        if c0 is not None and c1 is not None and max(c0, c1) >= 0.02:
            near.append(f"its share on where the {culprit_label} was {c0:.0%} → {c1:.0%}")
    if near:
        text += f" {_cap(', '.join(near))}."
    if shift_px is not None:
        text += f" Its predicted reach now ends {shift_px:.0f} px away from where it did"
        if reach_to_target_px is not None:
            d0, d1 = reach_to_target_px
            if d1 < min(d0, 15.0):
                text += f", right on the {target_name} (was {d0:.0f} px off)"
            elif d1 < d0:
                text += f", {d1:.0f} px from the {target_name} (was {d0:.0f} px)"
            else:
                # One re-plan from the failure pose, with the arm already off course: not evidence of innocence.
                text += f" ({d1:.0f} px from the {target_name}, was {d0:.0f} px; one re-plan from the failure pose)"
        text += "."
    if a.outcome == "confirmed":
        text += f" {_cap(removed)} was steering the policy."
    elif a.outcome == "implicated":
        text += f" {_cap(removed)} was drawing the policy's attention."
    return text


def map_sentence(
    m: CounterfactualMap, culprit_label: str | None, removed: str | None = None, target_name: str = "red cube",
) -> str:
    """A short line for a secondary model's map."""
    names = _names(culprit_label, removed or f"the {culprit_label}", target_name)
    text = f"{m.title}: {m.moved:.0%} of its attention moves"
    flow = _flow(m, names)
    if flow:
        return text + ", " + ", ".join(flow) + "."
    c0, c1 = m.shares_before.get(culprit_label or ""), m.shares_after.get(culprit_label or "")
    t0, t1 = m.shares_before.get("target"), m.shares_after.get("target")
    if c0 is not None and c1 is not None and max(c0, c1) >= 0.02:
        text += f", {c0:.0%} → {c1:.0%} on where the {culprit_label} was"
    if t0 is not None and t1 is not None and max(t0, t1) >= 0.02:
        text += f", target {t0:.0%} → {t1:.0%}"
    return text + "."
