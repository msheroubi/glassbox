"""Render the GlassBox-SO101 explainer video from recorded sessions.

    python docs/video/capture.py --base http://localhost:8020 nominal attention_hijack servo_stall specular_glare occlusion
    python docs/video/capture.py --base http://localhost:8021 --webcam
    PYTHONPATH=. python docs/video/make_video.py                      # -> docs/video/glassbox_explainer.mp4
    PYTHONPATH=. python docs/video/make_video.py --preview hijack 12  # one PNG frame, for layout work

Everything on screen comes from the recordings: camera frames, policy saliency, SAM 3.1 masks, the
Muse Spark diagnoses and answers, the Muse Wonder plan, the Muse Image counterfactuals and the
measured latencies. HOLD periods (the arm waiting on the models) are fast-forwarded and labelled.
Captions are the narration; the same text is written to docs/video/narration.md with timestamps.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from backend.robot.calibration import Homography

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / ".cache" / "video_capture"
OUT_MP4 = ROOT / "docs" / "video" / "glassbox_explainer.mp4"
OUT_SCRIPT = ROOT / "docs" / "video" / "narration.md"

W, H, FPS = 1920, 1080, 30
SRC_W, SRC_H = 1280, 720
VIDEO_BOX = (48, 136, 1200, 784)  # 1152 x 648
PANEL_BOX = (1232, 136, 1872, 784)

# ----------------------------------------------------------------------------------------- style
BG = (8, 11, 19)
PANEL = (17, 23, 37)
EDGE = (42, 54, 80)
TEXT = (234, 239, 248)
DIM = (139, 151, 176)
ACCENT = (76, 201, 240)
GOOD = (61, 220, 151)
WARN = (255, 183, 3)
BAD = (239, 71, 111)
MODEL = {
    "SAM 3.1": (46, 196, 182),
    "Muse Spark": (255, 183, 3),
    "Muse Wonder": (167, 108, 255),
    "Muse Image": (255, 93, 143),
    "Muse Voice Transcribe": (72, 149, 239),
}
ROLE = {"target": GOOD, "distractor": BAD, "gripper": (176, 186, 206), "obstacle": WARN, "hand": (255, 140, 90)}
JET = (colormaps["jet"](np.arange(256))[:, :3] * 255).astype(np.float32)

SANS = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"
MONO = "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono[wght].ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEJAVU_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


@lru_cache(None)
def F(size: int, weight: int = 400, mono: bool = False, symbols: bool = False) -> ImageFont.FreeTypeFont:
    if symbols:  # DejaVu covers maths glyphs (Σ, ∈, →) that the UI font may lack
        return ImageFont.truetype(DEJAVU_BOLD if weight >= 600 else DEJAVU, size)
    try:
        f = ImageFont.truetype(MONO if mono else SANS, size)
        axes = f.get_variation_axes()
        vals = []
        for a in axes:
            name = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
            vals.append(weight if name.lower().startswith("weight") else a["default"])
        f.set_variation_by_axes(vals)
        return f
    except Exception:  # noqa: BLE001 - fall back to DejaVu when variable fonts are unsupported
        return ImageFont.truetype(DEJAVU_MONO if mono else DEJAVU_BOLD if weight >= 600 else DEJAVU, size)


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t: float, t0: float, dur: float = 0.5) -> float:
    return ease((t - t0) / dur)


def mix(c1, c2, a: float):
    return tuple(int(round(u * (1 - a) + v * a)) for u, v in zip(c1, c2))


def wrap(text: str, font, width: float) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def text_block(draw, xy, text, font, width, fill=TEXT, reveal: float = 1.0, spacing: float = 1.3, max_lines=99) -> int:
    """Wrapped text with a typewriter reveal (0..1). The layout is fixed so revealing never reflows."""
    lines = wrap(text.replace("→", "->"), font, width)[:max_lines]
    budget = int(round(sum(len(l) for l in lines) * min(1.0, max(0.0, reveal))))
    x, y = xy
    lh = int(font.size * spacing)
    for line in lines:
        if budget <= 0:
            break
        draw.text((x, y), line[:budget], font=font, fill=fill)
        budget -= len(line)
        y += lh
    return len(lines) * lh


def fit_sentences(text: str, font, width: float, max_lines: int) -> str:
    """Longest run of whole sentences that wraps into ``max_lines`` (never cut a model's sentence mid-word)."""
    parts = [p.strip() for p in text.replace("\n", " ").split(". ") if p.strip()]
    out = ""
    for p in parts:
        trial = f"{out} {p}".strip()
        trial = trial if trial.endswith(".") else trial + "."
        if len(wrap(trial, font, width)) > max_lines:
            break
        out = trial
    if out:
        return out
    words, out = text.split(), ""
    for w_ in words:
        if len(wrap(f"{out} {w_} …".strip(), font, width)) > max_lines:
            break
        out = f"{out} {w_}".strip()
    return out.rstrip(",;:") + " …"


def chip(draw, xy, text, color, size=20, solid=False) -> int:
    font = F(size, 600)
    w = int(font.getlength(text)) + 24
    h = size + 14
    x, y = xy
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=color if solid else mix(BG, color, 0.18),
                           outline=color, width=2)
    draw.text((x + 12, y + 6), text, font=font, fill=BG if solid else color)
    return w


def panel(draw, box, title: str | None = None, color=ACCENT) -> tuple[int, int, int, int]:
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline=EDGE, width=2)
    x0, y0, x1, y1 = box
    if not title:
        return (x0 + 22, y0 + 18, x1 - 22, y1 - 18)
    draw.rectangle((x0 + 22, y0 + 24, x0 + 26, y0 + 46), fill=color)
    draw.text((x0 + 36, y0 + 21), title.upper(), font=F(20, 700), fill=DIM)
    return (x0 + 22, y0 + 64, x1 - 22, y1 - 18)


def check(draw, x, y, color=GOOD, s=12):
    draw.line([(x, y + s * 0.55), (x + s * 0.4, y + s), (x + s, y)], fill=color, width=3)


def paste_fit(canvas: Image.Image, im: Image.Image, box, label: str | None = None):
    x0, y0, x1, y1 = box
    r = min((x1 - x0) / im.width, (y1 - y0) / im.height)
    im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    x, y = x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2
    canvas.paste(im, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle((x - 1, y - 1, x + im.width, y + im.height), outline=EDGE, width=2)
    if label:
        f = F(19, 600)
        d.rectangle((x, y + im.height - 34, x + int(f.getlength(label)) + 24, y + im.height), fill=(0, 0, 0))
        d.text((x + 12, y + im.height - 30), label, font=f, fill=TEXT)
    return (x, y, x + im.width, y + im.height)


# -------------------------------------------------------------------------------------- sessions
class Session:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = CAP / name
        if not (self.dir / "frames.jsonl").exists():
            raise SystemExit(f"Missing capture {self.dir}: run docs/video/capture.py first")
        self.frames = [json.loads(l) for l in open(self.dir / "frames.jsonl")]
        self.t = np.array([f["t_rel"] for f in self.frames])
        self.events = [json.loads(l) for l in open(self.dir / "events.jsonl")]
        self.analysis = self._json("analysis.json")
        self.ask = self._json("ask.json")
        self.keyframe = self._img("keyframe.jpg")
        self.counterfactual = self._img("counterfactual.jpg")
        self._rgb: dict[int, np.ndarray] = {}
        self._attr: dict[int, dict[int, float]] = {}

    def _json(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def _img(self, name):
        p = self.dir / name
        return Image.open(p).convert("RGB") if p.exists() else None

    def ev(self, kind: str, nth: int = 0, default: float = 0.0) -> float:
        hits = [e["t_rel"] for e in self.events if e["type"] == kind or (e["type"] == "_mark" and e.get("label") == kind)]
        return hits[nth] if len(hits) > nth else default

    def first_analysis(self) -> dict | None:
        return next((e["result"] for e in self.events if e["type"] == "analysis"), None)

    def idx(self, t: float) -> int:
        return int(min(len(self.frames) - 1, max(0, np.searchsorted(self.t, t, "right") - 1)))

    def rgb(self, i: int) -> np.ndarray:
        if i not in self._rgb:
            if len(self._rgb) > 64:
                self._rgb.pop(next(iter(self._rgb)))
            im = cv2.imread(str(self.dir / "frames" / f"{i:05d}.jpg"))[:, :, ::-1]
            if im.shape[1] != SRC_W:
                im = cv2.resize(im, (SRC_W, SRC_H))
            self._rgb[i] = im.astype(np.float32)
        return self._rgb[i]

    def saliency(self, i: int) -> np.ndarray | None:
        s = self.frames[i].get("saliency")
        if not s or not s.get("data"):
            return None
        return np.frombuffer(base64.b64decode(s["data"]), np.uint8).reshape(s["h"], s["w"]).astype(np.float32) / 255

    def live_attr(self, i: int) -> dict[int, float]:
        """A(i) on the streamed (quarter-resolution) saliency and live masks: an approximation of the backend's."""
        if i not in self._attr:
            sal = self.saliency(i)
            out = {}
            if sal is not None and sal.sum() > 0:
                s = cv2.resize(sal, (320, 180), interpolation=cv2.INTER_LINEAR)
                tot = s.sum() + 1e-9
                for o in self.frames[i].get("objects", []):
                    m = np.zeros((180, 320), np.uint8)
                    for poly in o.get("polygons", []):
                        cv2.fillPoly(m, [(np.array(poly) * 0.25).astype(np.int32)], 1)
                    out[o["mask_id"]] = float((s * m).sum() / tot)
            self._attr[i] = out
        return self._attr[i]

    def live_share(self, i: int, label: str) -> float:
        a = self.live_attr(i)
        return sum(a.get(o["mask_id"], 0.0) for o in self.frames[i].get("objects", []) if o["label"] == label)

    def pipeline(self) -> list[tuple[str, str, float, float]]:
        """(model, task, start, duration) in session seconds, reconstructed from the measured latencies."""
        lat = (self.analysis or {}).get("latency_ms", {})
        ta = self.ev("analysis_started")
        tdiag = self.ev("analysis", default=ta)
        sam = lat.get("sam_ms", 0) / 1000
        steps = [("SAM 3.1", "segment concepts", ta, sam),
                 ("A(i)", "attention per mask", ta + sam, 0.05),
                 ("Muse Spark", "diagnose (JSON schema)", ta + sam + 0.05, max(0.1, tdiag - ta - sam - 0.05))]
        if "wonder_ms" in lat:
            steps.append(("Muse Wonder", "recovery plan", tdiag, lat["wonder_ms"] / 1000))
        if "image_ms" in lat:
            steps.append(("Muse Image", "counterfactual", tdiag, lat["image_ms"] / 1000))
        return steps


@dataclass
class Clip:
    """Piecewise-linear map from scene time to session time: [(scene_t0, scene_t1, src_t0, src_t1), ...]."""

    segs: list[tuple[float, float, float, float]]

    def src(self, t: float) -> float:
        for a, b, s0, s1 in self.segs:
            if t <= b:
                return s0 + (s1 - s0) * min(1.0, max(0.0, (t - a) / max(1e-6, b - a)))
        return self.segs[-1][3]

    def speed(self, t: float) -> float:
        for a, b, s0, s1 in self.segs:
            if t <= b:
                return (s1 - s0) / max(1e-6, b - a)
        return 0.0


# ------------------------------------------------------------------------------------ footage
def footage(sess: Session, t_src: float, *, heat=True, live=True, sam=False, culprit=None, path=None,
            path_color=GOOD, speed=1.0, size=(1152, 648), extra: Callable | None = None, tags=True) -> Image.Image:
    i = sess.idx(t_src)
    j = min(i + 1, len(sess.frames) - 1)
    a = 0.0
    if j != i and sess.frames[i]["view"] == sess.frames[j]["view"] and sess.t[j] - sess.t[i] < 0.35:
        a = float(np.clip((t_src - sess.t[i]) / (sess.t[j] - sess.t[i]), 0, 1))
    img = sess.rgb(i) * (1 - a) + sess.rgb(j) * a
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if heat:
        si, sj = sess.saliency(i), sess.saliency(j)
        if si is not None:
            sal = si if sj is None else si * (1 - a) + sj * a
            up = np.clip(cv2.resize(sal, size, interpolation=cv2.INTER_CUBIC), 0, 1)
            alpha = (up ** 1.1 * 0.62)[..., None]
            img = img * (1 - alpha) + JET[(up * 255).astype(np.uint8)] * alpha
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    k = size[0] / SRC_W
    f = sess.frames[i]

    def outline(o, color, width, tag, fill=40):
        for poly in o.get("polygons", []):
            pts = [(x * k, y * k) for x, y in poly]
            if len(pts) > 2:
                d.polygon(pts, fill=color + (fill,), outline=color + (255,), width=width)
        x0, y0 = o["bbox"][0] * k, o["bbox"][1] * k
        font = F(17, 700)
        tw = font.getlength(tag)
        y = o["bbox"][3] * k + 4 if o["label"] == "gripper" else max(0, y0 - 28)
        d.rounded_rectangle((x0, y, x0 + tw + 16, y + 25), radius=6, fill=(0, 0, 0, 200), outline=color + (255,))
        d.text((x0 + 8, y + 3), tag, font=font, fill=color)

    # Live mask ids are not the analysis ids, so match the suppressed object by label.
    sup = f.get("suppression") or {}
    suppressed = {o["label"] for o in sup.get("objects", [])} if sup.get("active") else set()
    if sam and sess.analysis:
        shares = {x["mask_id"]: x["attention_share"] for x in sess.analysis["attributions"]}
        for o in sess.analysis["objects"]:
            color = BAD if o["mask_id"] == culprit else ROLE.get(o["label"], ACCENT)
            name = o["prompt"].replace(" (sim-ground-truth)", "")
            outline(o, color, 4 if o["mask_id"] == culprit else 2,
                    f"#{o['mask_id']} {name}  A={shares.get(o['mask_id'], 0):.2f}", 60 if o["mask_id"] == culprit else 30)
    elif live:
        attr = sess.live_attr(i)
        for o in f.get("objects", []):
            if o["label"] == "gripper":
                continue
            color = ROLE.get(o["label"], ACCENT)
            name = o["prompt"].replace(" (sim-ground-truth)", "")
            if o["label"] in suppressed:
                outline(o, (150, 150, 150), 2, f"{name}: erased from policy input", 0)
            else:
                outline(o, color, 2, f"{name}  A={attr.get(o['mask_id'], 0):.2f}", 22)
    if path:
        pts = [(x * k, y * k) for x, y in path]
        d.line(pts, fill=path_color + (255,), width=5, joint="curve")
        for n, (x, y) in enumerate(pts):
            r = 7 if n else 9
            d.ellipse((x - r, y - r, x + r, y + r), fill=path_color + (255,) if n else (255, 255, 255, 255),
                      outline=(0, 0, 0, 255), width=2)
    if extra:
        extra(d, k)
    if not tags:
        return im
    view = "POLICY VIEW" if f.get("view") == "policy" else "CAMERA"
    d.rounded_rectangle((14, 14, 40 + F(18, 700).getlength(view), 46), radius=8, fill=(0, 0, 0, 170))
    d.text((27, 19), view, font=F(18, 700), fill=TEXT)
    state = f.get("state", "")
    sc = {"RUNNING": GOOD, "HOLD": WARN, "COMPLETE": ACCENT, "ESTOP": BAD}.get(state, DIM)
    sw = F(20, 800).getlength(state)
    d.rounded_rectangle((size[0] - sw - 50, 14, size[0] - 14, 50), radius=18, fill=sc + (235,))
    d.text((size[0] - sw - 32, 20), state, font=F(20, 800), fill=BG)
    if speed > 1.3:
        txt = f"FAST-FORWARD {speed:.1f}×"
        tw = F(18, 700).getlength(txt)
        d.rounded_rectangle((size[0] - tw - 46, 58, size[0] - 14, 88), radius=8, fill=(0, 0, 0, 180))
        d.text((size[0] - tw - 30, 63), txt, font=F(18, 700), fill=WARN)
    if heat and f.get("saliency"):
        m = f["saliency"].get("method", "")
        txt = f"heatmap: {m.replace('_', ' ')}"
        d.text((16, size[1] - 32), txt, font=F(16, 500), fill=(255, 255, 255, 190))
    return im


# ------------------------------------------------------------------------------------- widgets
def bars(draw, box, rows, thr=None, title_note=""):
    x0, y0, x1, _ = box
    if title_note:
        draw.text((x0, y0), title_note, font=F(18, 500), fill=DIM)
        y0 += 32
    bw = x1 - x0
    for label, v, color in rows:
        draw.text((x0, y0), label, font=F(22, 600), fill=TEXT)
        val = f"{v:.2f}"
        draw.text((x1 - F(22, 700, mono=True).getlength(val), y0), val, font=F(22, 700, mono=True), fill=color)
        y0 += 32
        draw.rounded_rectangle((x0, y0, x1, y0 + 16), radius=8, fill=(30, 38, 58))
        if v > 0.005:
            draw.rounded_rectangle((x0, y0, x0 + max(16, bw * min(1, v)), y0 + 16), radius=8, fill=color)
        if thr is not None:
            tx = x0 + bw * thr
            draw.line([(tx, y0 - 6), (tx, y0 + 22)], fill=TEXT, width=2)
        y0 += 34
    if thr is not None:
        draw.text((x0 + bw * thr - 40, y0 - 8), f"hijack {thr:.2f}", font=F(16, 500), fill=DIM)
    return y0 + 18


def pipeline_rows(draw, box, steps, t_src, only=None) -> int:
    x0, y0, x1, _ = box
    for model, task, start, dur in steps:
        if only and model not in only:
            continue
        color = MODEL.get(model, ACCENT)
        p = (t_src - start) / max(dur, 1e-6)
        done = p >= 1
        draw.ellipse((x0, y0 + 7, x0 + 12, y0 + 19), fill=color if p > 0 else (50, 60, 80))
        draw.text((x0 + 22, y0), model, font=F(21, 700), fill=TEXT if p > 0 else DIM)
        draw.text((x0 + 22 + F(21, 700).getlength(model) + 10, y0 + 2), task, font=F(18, 400), fill=DIM)
        if done:
            label = f"{dur:.2f} s" if dur < 0.1 else f"{dur:.1f} s"
            label = "<5 ms" if dur < 0.1 else label
            tw = F(20, 700, mono=True).getlength(label)
            draw.text((x1 - tw, y0 + 1), label, font=F(20, 700, mono=True), fill=color)
            check(draw, x1 - tw - 24, y0 + 6, GOOD)
        elif p > 0:
            label = f"{t_src - start:.1f} s"
            draw.text((x1 - F(20, 500, mono=True).getlength(label), y0 + 1), label, font=F(20, 500, mono=True), fill=DIM)
        y0 += 30
        draw.rounded_rectangle((x0 + 22, y0, x1, y0 + 6), radius=3, fill=(30, 38, 58))
        if p > 0:
            draw.rounded_rectangle((x0 + 22, y0, x0 + 22 + (x1 - x0 - 22) * min(1, p), y0 + 6), radius=3, fill=color)
        y0 += 20
    return y0


def diag_card(draw, box, diag: dict, source: str, reveal: float, root_cause=True, max_root_lines=7):
    x0, y0, x1, _ = box
    ft = diag["failure_type"]
    color = GOOD if ft == "NONE" else BAD
    draw.text((x0, y0), "MUSE SPARK DIAGNOSIS", font=F(17, 700), fill=MODEL["Muse Spark"])
    src = source
    draw.text((x1 - F(17, 600).getlength(src), y0), src, font=F(17, 600), fill=GOOD if src == "meta-api" else WARN)
    y0 += 30
    draw.text((x0, y0), ft, font=F(34, 800), fill=color)
    conf = f"conf {diag['confidence']:.2f}"
    chip(draw, (x1 - F(18, 600).getlength(conf) - 24, y0 + 4), conf, color, 18)
    y0 += 52
    y0 += text_block(draw, (x0, y0), diag["headline_summary"], F(24, 600), x1 - x0, TEXT, reveal * 2.5, 1.25, 3) + 10
    if root_cause:
        y0 += text_block(draw, (x0, y0), fit_sentences(diag["technical_root_cause"], F(19, 400), x1 - x0, max_root_lines), F(19, 400), x1 - x0, DIM,
                         max(0.0, reveal * 1.4 - 0.3), 1.35, max_root_lines) + 12
    it = diag["intervention_type"]
    params = ", ".join(f"{k}={v}" for k, v in diag["intervention_parameters"].items())
    if reveal > 0.55:
        draw.text((x0, y0), "→", font=F(24, 700, symbols=True), fill=ACCENT)
        w = chip(draw, (x0 + 34, y0 - 2), it, ACCENT if it != "NONE" else WARN, 18)
        draw.text((x0 + 44 + w, y0 + 4), params, font=F(17, 500, mono=True), fill=DIM)
    return y0 + 40


def line_chart(draw, box, xs, series, x_now, x_lo, x_hi, y_max, thr=None, thr_label=""):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=EDGE, width=1)
    for g in range(1, 4):
        gy = y1 - (y1 - y0) * g / 4
        draw.line([(x0, gy), (x1, gy)], fill=(28, 36, 54), width=1)
    sx = lambda x: x0 + (x1 - x0) * (x - x_lo) / (x_hi - x_lo)
    sy = lambda y: y1 - (y1 - y0) * min(1.0, y / y_max)
    if thr is not None:
        draw.line([(x0, sy(thr)), (x1, sy(thr))], fill=WARN, width=2)
        draw.text((x0 + 6, sy(thr) - 22), thr_label, font=F(15, 600), fill=WARN)
    mask = (xs <= x_now) & (xs >= x_lo)
    for ys, color, label in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs[mask], ys[mask])]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
    lx = x1
    for ys, color, label in reversed(series):
        lx -= F(16, 600).getlength(label) + 30
        draw.rectangle((lx, y0 + 10, lx + 12, y0 + 22), fill=color)
        draw.text((lx + 18, y0 + 6), label, font=F(16, 600), fill=TEXT)
    if x_lo <= x_now <= x_hi:
        draw.line([(sx(x_now), y0), (sx(x_now), y1)], fill=(255, 255, 255), width=1)


def code_block(draw, box, lines, size=17, reveal=1.0, colors=None):
    x0, y0, x1, y1 = box
    font = F(size, 500, mono=True)
    n = int(round(len(lines) * min(1, max(0, reveal))))
    for k, line in enumerate(lines[:n]):
        line = line.replace("→", "->")
        c = (colors or {}).get(k, (190, 205, 230))
        if line.lstrip().startswith("#"):
            c = (110, 160, 120)
        if font.getlength(line) > x1 - x0:
            while font.getlength(line + "…") > x1 - x0 and len(line) > 4:
                line = line[:-1]
            line = line.rstrip() + "…"
        draw.text((x0, y0 + k * int(size * 1.45)), line, font=font, fill=c)


# ------------------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    key: str
    chapter: str
    title: str
    duration: float
    draw: Callable[[Image.Image, ImageDraw.ImageDraw, float], None]
    captions: list[tuple[float, float, str]]
    models: list[str] = field(default_factory=list)


def background() -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 - 0.55 * np.hypot((xx - W * 0.3) / W, (yy - H * 0.2) / H)
    base = np.stack([BG[0] + 10 * g, BG[1] + 14 * g, BG[2] + 26 * g], -1)
    base[::48, :, :] += 5
    base[:, ::48, :] += 5
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


BACKGROUND: Image.Image | None = None
S: dict[str, Session] = {}


def load_sessions():
    for n in ["nominal", "attention_hijack", "servo_stall", "specular_glare", "occlusion", "webcam"]:
        S[n] = Session(n)


def models_row(draw, x, y, names, size=20, t=99.0, stagger=0.0):
    for n, name in enumerate(names):
        a = ramp(t, n * stagger, 0.4)
        if a <= 0:
            continue
        x += chip(draw, (x, y), name, mix(BG, MODEL[name], a), size) + 12
    return x


# -- 0 title ---------------------------------------------------------------------------------
def s_title(img, d, t):
    nom = S["nominal"]
    fr = footage(nom, 2.5 + t * 0.4, heat=True, live=False, size=(1920, 1080), tags=False)
    fr = Image.blend(Image.new("RGB", fr.size, BG), fr.filter(ImageFilter.GaussianBlur(6)), 0.16)
    img.paste(fr, (0, 0))
    d = ImageDraw.Draw(img)
    a = ramp(t, 0.3, 0.8)
    d.text((140, 330), "GlassBox", font=F(150, 800), fill=mix(BG, TEXT, a))
    d.text((140 + F(150, 800).getlength("GlassBox") + 20, 380), "SO-101", font=F(80, 300), fill=mix(BG, ACCENT, a))
    b = ramp(t, 1.2, 0.8)
    d.text((146, 530), "Explainable failure recovery for a learned robot policy", font=F(46, 500), fill=mix(BG, TEXT, b))
    d.text((146, 600), "See what the policy sees  ·  name the failure  ·  fix it live", font=F(32, 400), fill=mix(BG, DIM, b))
    models_row(d, 146, 700, list(MODEL), 24, t - 2.0, 0.35)


# -- 1 problem -------------------------------------------------------------------------------
def s_problem(img, d, t):
    nom = S["nominal"]
    src = min(nom.t[-1], t * 0.72)
    img.paste(footage(nom, src, heat=False, live=False), VIDEO_BOX[:2])
    box = panel(d, PANEL_BOX, "the problem", BAD)
    x0, y0, x1, _ = box
    items = [
        (0.6, "Imitation-learned policies (ACT) map camera pixels to joint targets, 30 times a second."),
        (2.8, "They learn from a few dozen demos and generalise poorly to messy, real desks."),
        (6.2, "When they fail there is no error message: just an arm that hovers, jams or misses."),
        (8.6, "The usual fix: record more demos, retrain for hours, and hope."),
    ]
    for t0, txt in items:
        a = ramp(t, t0, 0.5)
        if a <= 0:
            continue
        d.ellipse((x0, y0 + 12, x0 + 10, y0 + 22), fill=mix(PANEL, BAD if t0 > 6 else ACCENT, a))
        y0 += text_block(d, (x0 + 26, y0), txt, F(27, 500), x1 - x0 - 26, mix(PANEL, TEXT, a), 1, 1.3) + 26


# -- 2 architecture --------------------------------------------------------------------------
def s_arch(img, d, t):
    def node(box, title, sub, color, t0):
        a = ramp(t, t0, 0.45)
        if a <= 0:
            return
        d.rounded_rectangle(box, radius=14, fill=mix(BG, mix(PANEL, color, 0.12), a), outline=mix(BG, color, a), width=3)
        d.text((box[0] + 18, box[1] + 14), title, font=F(26, 700), fill=mix(BG, TEXT, a))
        text_block(d, (box[0] + 18, box[1] + 52), sub, F(18, 400), box[2] - box[0] - 36, mix(BG, DIM, a), 1, 1.3, 3)

    def arrow(p0, p1, t0, color=EDGE, dur=0.5):
        a = ramp(t, t0, dur)
        if a <= 0:
            return
        q = (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a)
        d.line([p0, q], fill=color, width=4)
        if a >= 1:
            ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            pts = [p1, (p1[0] - 16 * np.cos(ang - 0.45), p1[1] - 16 * np.sin(ang - 0.45)),
                   (p1[0] - 16 * np.cos(ang + 0.45), p1[1] - 16 * np.sin(ang + 0.45))]
            d.polygon(pts, fill=color)
            # a packet travelling along the arrow
            u = ((t - t0) * 0.6) % 1
            px, py = p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u
            d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)

    # lane labels
    d.text((60, 150), "REAL-TIME LOOP · 30 Hz · never waits on the network", font=F(20, 700), fill=mix(BG, GOOD, ramp(t, 0.2)))
    d.text((60, 440), "ASYNC XAI PIPELINE · runs while the arm HOLDs", font=F(20, 700), fill=mix(BG, WARN, ramp(t, 5.4)))
    d.line([(60, 425), (1860, 425)], fill=mix(BG, EDGE, ramp(t, 5.2)), width=1)
    # row 1
    y = 190
    node((60, y, 400, y + 170), "Overhead camera", "USB webcam, 1280×720", ACCENT, 0.3)
    node((480, y, 900, y + 170), "ACT policy", "Cross-attention hooks give a spatial saliency map every tick", ACCENT, 0.9)
    node((980, y, 1360, y + 170), "SO-101 servos", "Joint targets out; position, velocity and load back", ACCENT, 1.5)
    node((1440, y, 1860, y + 170), "Anomaly triggers", "Joint error · action-chunk variance · A(i) > 0.45 · no progress", BAD, 2.4)
    arrow((400, y + 85), (476, y + 85), 0.8, ACCENT)
    arrow((900, y + 85), (976, y + 85), 1.4, ACCENT)
    arrow((1360, y + 85), (1436, y + 85), 2.2, BAD)
    # row 2
    y = 480
    node((60, y, 400, y + 170), "SAM 3.1", "Segments the scene by plain-English concept", MODEL["SAM 3.1"], 5.6)
    node((480, y, 900, y + 170), "A(i) attribution", "Share of the policy's saliency inside each mask", ACCENT, 6.3)
    node((980, y, 1360, y + 170), "Muse Spark", "Keyframe + evidence JSON in, schema-checked diagnosis out", MODEL["Muse Spark"], 7.0)
    node((1440, y, 1860, y + 170), "One-click fix", "Mask suppression · waypoint bypass · grasp offset · ask a human", GOOD, 11.2)
    arrow((1650, 360), (1650, 476), 5.5, BAD)
    d.text((1666, 395), "HOLD + keyframe", font=F(16, 600), fill=mix(BG, BAD, ramp(t, 5.5)))
    arrow((400, y + 85), (476, y + 85), 6.1, MODEL["SAM 3.1"])
    arrow((900, y + 85), (976, y + 85), 6.8, ACCENT)
    arrow((1360, y + 85), (1436, y + 85), 10.9, MODEL["Muse Spark"])
    # side models
    y = 700
    node((480, y, 900, y + 76), "Muse Image · counterfactual", "", MODEL["Muse Image"], 8.2)
    node((980, y, 1360, y + 76), "Muse Wonder · recovery plan", "", MODEL["Muse Wonder"], 8.8)
    node((1440, y, 1860, y + 76), "Muse Voice Transcribe · Q&A", "", MODEL["Muse Voice Transcribe"], 9.4)
    arrow((1170, 650), (1170, 696), 8.7, MODEL["Muse Wonder"])
    arrow((690, 650), (690, 696), 8.1, MODEL["Muse Image"])
    # the fix goes back to the loop
    a = ramp(t, 12.4, 0.6)
    if a > 0:
        d.line([(1860, 565), (1890, 565), (1890, 275), (1864, 275)], fill=mix(BG, GOOD, a), width=4)
        d.text((1460, 662), "validated, then handed back to the loop", font=F(16, 600), fill=mix(BG, GOOD, a))


# -- 3 the five models -----------------------------------------------------------------------
MODEL_CARDS = [
    ("SAM 3.1", "sam-3.1", "Open-vocab segmentation",
     "Turns \"red ball\", \"cardboard box\", \"human hand\" into pixel masks, so attention can be measured per object.",
     "1.3–3.3 s"),
    ("Muse Spark", "muse-spark-1.3", "Multimodal diagnosis",
     "Reads the heatmap keyframe and the telemetry JSON; returns failure type, culprit mask and a fix, schema-validated.",
     "8.6–11.8 s"),
    ("Muse Wonder", "muse-wonder-1.0", "Deep-reasoning recovery",
     "Given kinematics, 10 s of servo telemetry and the executor's source, checks and explains a waypoint bypass.",
     "22 s · async"),
    ("Muse Image", "muse-image-1.0", "Counterfactual rendering",
     "Edits the keyframe into the scene the policy should have seen: the failure, made visible.",
     "8–14 s · async"),
    ("Muse Voice Transcribe", "muse-voice-transcribe-1.0", "Hands-free operator",
     "Streams transcripts: \"emergency stop\" and \"analyze\" are commands; anything else is a question for Spark.",
     "streaming"),
]


def s_models(img, d, t):
    cw, ch, gap = 342, 520, 22
    x = (W - 5 * cw - 4 * gap) // 2
    for n, (name, mid, role, body, lat) in enumerate(MODEL_CARDS):
        a = ramp(t, 0.4 + n * 0.9, 0.5)
        if a <= 0:
            x += cw + gap
            continue
        c = MODEL[name]
        y = 170 + int((1 - a) * 40)
        d.rounded_rectangle((x, y, x + cw, y + ch), radius=18, fill=mix(BG, PANEL, a), outline=mix(BG, c, a), width=3)
        d.rectangle((x + 3, y + 3, x + cw - 3, y + 12), fill=mix(BG, c, a))
        text_block(d, (x + 24, y + 40), name, F(36, 800), cw - 48, mix(BG, TEXT, a), 1, 1.1, 2)
        yy = y + 40 + (100 if F(36, 800).getlength(name) > cw - 48 else 52)
        d.text((x + 24, yy), mid, font=F(18, 500, mono=True), fill=mix(BG, c, a))
        d.text((x + 24, yy + 44), role, font=F(23, 700), fill=mix(BG, TEXT, a))
        text_block(d, (x + 24, yy + 88), body, F(20, 400), cw - 48, mix(BG, DIM, a), 1, 1.4)
        d.text((x + 24, y + ch - 56), "measured", font=F(16, 500), fill=mix(BG, DIM, a))
        d.text((x + 24, y + ch - 36), lat, font=F(22, 700, mono=True), fill=mix(BG, c, a))
        x += cw + gap


# -- scenario helpers ------------------------------------------------------------------------
def hold_badge(d, sess: Session, t_src: float):
    """Bottom-left 'waiting on the models' clock in real session seconds."""
    ta = sess.ev("anomaly")
    if t_src < ta:
        return
    x, y = VIDEO_BOX[0] + 16, VIDEO_BOX[3] - 76
    txt = f"arm on HOLD for {t_src - ta:4.1f} s (real time)"
    d.rounded_rectangle((x, y, x + F(19, 600).getlength(txt) + 28, y + 36), radius=8, fill=(0, 0, 0))
    d.text((x + 14, y + 6), txt, font=F(19, 600), fill=WARN)


def fix_banner(d, text, color=GOOD, a=1.0):
    x0, y0, x1, _ = VIDEO_BOX
    f = F(24, 700)
    w = f.getlength(text) + 60
    xc = (x0 + x1) / 2
    d.rounded_rectangle((xc - w / 2, y0 + 64, xc + w / 2, y0 + 110), radius=12, fill=mix(BG, color, a))
    d.text((xc - w / 2 + 30, y0 + 73), text, font=f, fill=BG)


# -- 4 attention hijack ----------------------------------------------------------------------
def s_hijack(img, d, t):
    s = S["attention_hijack"]
    an = s.analysis
    diag = an["diagnosis"]
    t_an, t_fix = s.ev("analysis"), s.ev("fix_applied")
    t_ready = s.ev("analysis_complete")
    clip = Clip([(0, 3.2, 0.0, s.ev("anomaly")), (3.2, 15.0, s.ev("anomaly"), t_ready), (15.0, 33.0, t_ready, t_ready),
                 (33.0, 42.0, t_fix, s.t[-1])])
    ts = clip.src(t)
    shares = {a["mask_id"]: a["attention_share"] for a in an["attributions"]}
    by = {a["label"]: a["mask_id"] for a in an["attributions"]}
    if t < 15.0:
        img.paste(footage(s, ts, sam=ts >= t_an, culprit=diag["culprit_mask_id"], speed=clip.speed(t)), VIDEO_BOX[:2])
        hold_badge(d, s, ts)
        box = panel(d, PANEL_BOX, "attention, then diagnosis", MODEL["Muse Spark"])
        if t < 3.2:
            i = s.idx(ts)
            y = bars(d, box, [("red ball  (distractor)", s.live_share(i, "distractor"), BAD),
                              ("red cube  (target)", s.live_share(i, "target"), GOOD)], 0.45, "live A(i), every control tick")
            d.text((box[0], y + 10), "A(i) = Σ saliency ∈ mask i  /  Σ saliency", font=F(25, 600, symbols=True), fill=TEXT)
            text_block(d, (box[0], y + 56), "The policy's own attention, measured inside each object's mask.",
                       F(19, 400), box[2] - box[0], DIM)
        else:
            y = pipeline_rows(d, box, s.pipeline(), ts)
            if ts < t_an:
                bars(d, (box[0], y + 14, box[2], box[3]),
                     [("red ball  (distractor)", shares[by["distractor"]] * ramp(ts, s.ev("anomaly") + 1.3, 1.2), BAD),
                      ("red cube  (target)", shares[by["target"]] * ramp(ts, s.ev("anomaly") + 1.3, 1.2), GOOD)],
                     0.45, "A(i) on SAM 3.1 masks")
            else:
                diag_card(d, (box[0], y + 14, box[2], box[3]), diag, an["diagnosis_source"], (t - clip_t(clip, t_an)) / 5, max_root_lines=5)
    elif t < 24.0:
        a = ramp(t, 15.0, 0.4)
        x0, y0, x1, y1 = VIDEO_BOX
        paste_fit(img, s.keyframe, (x0, y0, x1, y0 + 330), "what Muse Spark saw: camera | SAM 3.1 masks + A(i) + policy saliency")
        if t > 19.4:
            i = s.idx(s.ev("anomaly") + 1)
            raw = Image.fromarray(s.rgb(i).astype(np.uint8))
            paste_fit(img, raw, (x0, y0 + 346, x0 + 568, y1), "camera at the anomaly")
            paste_fit(img, s.counterfactual, (x0 + 584, y0 + 346, x1, y1), "Muse Image: what it should have seen")
        box = panel(d, PANEL_BOX, "evidence sent to muse spark", MODEL["Muse Spark"])
        ev = an["evidence"]
        lines = ["{", f' "anomaly_reasons": ["{ev["anomaly_reasons"][0][:34]}…"],', ' "objects": [']
        for o in ev["objects"]:
            lines.append(f'  {{"mask_id": {o["mask_id"]}, "label": "{o["label"]}",')
            lines.append(f'   "attention_share": {o["attention_share"]:.4f}}},')
        sv = ev["servo"]
        lines += [" ],", ' "servo": {', f'  "joint_error_deg_peak_10s": {sv["joint_error_deg_peak_10s"]},',
                  f'  "action_chunk_variance_peak_10s": {sv["action_chunk_variance_peak_10s"]},', "  …", " },",
                  f' "gripper_xy_mm": {ev["gripper_xy_mm"]},', f' "target_grasp_bias_mm": {ev["target_grasp_bias_mm"]}', "}",
                  "", "Muse Spark reply (json_schema):",
                  f' "failure_type": "{diag["failure_type"]}",', f' "culprit_mask_id": {diag["culprit_mask_id"]},',
                  f' "intervention_type": "{diag["intervention_type"]}",',
                  f' "intervention_parameters": {json.dumps(diag["intervention_parameters"])}']
        code_block(d, box, lines, 17, ramp(t, 15.2, 3.0), {len(lines) - 4 + k: MODEL["Muse Spark"] for k in range(4)})
        _ = a
    elif t < 33.0:
        fr = footage(s, ts, sam=True, culprit=diag["culprit_mask_id"])
        img.paste(Image.blend(Image.new("RGB", fr.size, BG), fr, 0.55), VIDEO_BOX[:2])
        box = panel(d, PANEL_BOX, "ask glassbox", MODEL["Muse Voice Transcribe"])
        x0, y0, x1, _ = box
        d.text((x0, y0), "OPERATOR  ·  voice (Muse Voice Transcribe) or text", font=F(17, 700), fill=MODEL["Muse Voice Transcribe"])
        # waveform while "speaking"
        for k in range(28):
            amp = (np.sin(k * 1.7 + t * 9) * 0.5 + 0.5) * (1 if t < 26.2 else 0.08) * 22 + 3
            cx = x0 + k * 14
            d.rounded_rectangle((cx, y0 + 60 - amp, cx + 7, y0 + 60 + amp), radius=3, fill=MODEL["Muse Voice Transcribe"])
        y = y0 + 98
        y += text_block(d, (x0, y), f"“{s.ask['question']}”", F(26, 600), x1 - x0, TEXT, ramp(t, 24.3, 2.0)) + 26
        if t > 26.6:
            d.text((x0, y), "MUSE SPARK  ·  grounded in the same evidence", font=F(17, 700), fill=MODEL["Muse Spark"])
            text_block(d, (x0, y + 34), s.ask["answer"].replace("\n\n", " "), F(21, 400), x1 - x0, TEXT,
                       (t - 26.8) / 4.0, 1.38, 13)
    else:
        img.paste(footage(s, ts, speed=clip.speed(t)), VIDEO_BOX[:2])
        fix_banner(d, "INPAINT_MASK_SUPPRESSION · mask #1 erased from the policy's input", GOOD, ramp(t, 33.0, 0.4))
        box = panel(d, PANEL_BOX, "after the fix", GOOD)
        i = s.idx(ts)
        y = bars(d, box, [("red ball  (distractor)", s.live_share(i, "distractor"), BAD),
                          ("red cube  (target)", s.live_share(i, "target"), GOOD)], 0.45, "live A(i) on the policy's view")
        y += text_block(d, (box[0], y + 10), "No retraining. The ball is inpainted out of every frame the policy sees; "
                        "SAM 3.1 re-segments it every 3 s so the mask stays locked on.", F(21, 400),
                        box[2] - box[0], DIM) + 20
        state = s.frames[i]["state"]
        chip(d, (box[0], y + 10), "task complete: cube in the bin" if state == "COMPLETE" else "policy running", GOOD if state == "COMPLETE" else ACCENT, 20, state == "COMPLETE")


def clip_t(clip: Clip, src_t: float) -> float:
    """Scene time at which the clip first reaches ``src_t``."""
    for a, b, s0, s1 in clip.segs:
        if s0 <= src_t <= s1 and s1 > s0:
            return a + (b - a) * (src_t - s0) / (s1 - s0)
    return 0.0


# -- 5 servo stall ---------------------------------------------------------------------------
def s_stall(img, d, t):
    s = S["servo_stall"]
    an = s.analysis
    diag, plan = an["diagnosis"], an["plan"]
    t_anom, t_an, t_fix = s.ev("anomaly"), s.ev("analysis"), s.ev("fix_applied")
    t_wonder = s.ev("analysis_update", 1)
    clip = Clip([(0, 5.0, 0.0, t_anom), (5.0, 17.0, t_anom, t_wonder + 0.3), (17.0, 26.0, t_wonder + 0.3, t_wonder + 0.3),
                 (26.0, 35.0, t_fix, s.t[-1])])
    ts = clip.src(t)
    i = s.idx(ts)
    rec = s.frames[i]["recovery"].get("active")
    first = s.first_analysis() or {}
    path, pcolor = None, GOOD
    if t_an <= ts < t_wonder:
        path, pcolor = first.get("plan_path_px"), WARN
    elif ts >= t_wonder and (t < 26 or rec):
        path, pcolor = an["plan_path_px"], MODEL["Muse Wonder"]
    if t < 26.0:
        img.paste(footage(s, ts, sam=ts >= t_an, culprit=diag["culprit_mask_id"], path=path, path_color=pcolor,
                          speed=clip.speed(t)), VIDEO_BOX[:2])
        hold_badge(d, s, ts)
    else:
        img.paste(footage(s, ts, path=an["plan_path_px"] if rec else None, path_color=MODEL["Muse Wonder"],
                          speed=clip.speed(t)), VIDEO_BOX[:2])
        done = not rec and ts > t_fix + 0.5  # the recovery flag lags the fix_applied event by a tick
        msg = "bypass done: control handed back to the policy" if done else \
            f"WAYPOINT_BYPASS · executing Muse Wonder's {len(plan['waypoints'])}-waypoint plan"
        fix_banner(d, msg, GOOD if done else MODEL["Muse Wonder"])
    box = panel(d, PANEL_BOX, "telemetry · diagnosis · plan", MODEL["Muse Wonder"])
    x0, y0, x1, y1 = box
    if t < 5.0 or t >= 26.0:
        xs = s.t
        je = np.array([f["telemetry"]["joint_error"] for f in s.frames])
        ld = np.array([max(abs(v) for v in f["telemetry"]["torque_load"][:5]) for f in s.frames])
        lo = 0.0 if t < 5 else t_fix - 1
        hi = t_anom + 0.5 if t < 5 else s.t[-1]
        line_chart(d, (x0, y0, x1, y0 + 250), xs, [(je, BAD, "joint error °"), (ld / 2, WARN, "peak load % ÷ 2")],
                   ts, lo, hi, 40, 12, "12° trigger")
        sv = an["evidence"]["servo"]
        facts = [("peak joint error", f"{sv['joint_error_deg_peak_10s']:.1f}°"), ("2 s mean error", f"{sv['joint_error_deg_mean_2s']:.1f}°"),
                 ("peak load (elbow)", f"{max(sv['peak_abs_load_pct_10s']):.0f} %"), ("trigger", "no progress for 4 s")]
        y = y0 + 274
        if t >= 26:
            facts = [("bypass", f"{len(plan['waypoints'])} waypoints · 2.4 s"), ("source", an["plan_source"]),
                     ("max joint speed", "≤ 60°/s (executor)"), ("clearance", "≥ 12.5 mm, swept path")]
        for k, (a, b) in enumerate(facts):
            if t < 5 and ts < t_anom - 1.5 and k < 3:
                continue
            d.text((x0, y), a, font=F(21, 500), fill=DIM)
            d.text((x1 - F(22, 700, mono=True).getlength(b), y), b, font=F(22, 700, mono=True), fill=TEXT)
            y += 40
    elif t < 17.0:
        y = pipeline_rows(d, box, s.pipeline(), ts)
        if ts >= t_an:
            y = diag_card(d, (x0, y + 10, x1, y1), diag, an["diagnosis_source"], (t - clip_t(clip, t_an)) / 5, max_root_lines=3)
            if ts < t_wonder:
                chip(d, (x0, y - 2), "local planner: bypass ready + validated", WARN, 18)
            else:
                chip(d, (x0, y - 2), "Muse Wonder plan: validated", MODEL["Muse Wonder"], 18, True)
    else:
        d.text((x0, y0), "MUSE WONDER  ·  reasoning effort: high", font=F(17, 700), fill=MODEL["Muse Wonder"])
        y = y0 + 32
        for k, wp in enumerate(plan["waypoints"]):
            if t < 17.3 + k * 0.6:
                break
            d.text((x0, y), f"{k + 1}", font=F(22, 800), fill=MODEL["Muse Wonder"])
            y += text_block(d, (x0 + 30, y), wp["note"], F(20, 500), x1 - x0 - 30, TEXT, 1, 1.3, 2) + 8
        y += 8
        # The route comments and the execution loop; the joint-space tuples in between are less telling.
        src = [l for l in plan["python_routine"].splitlines() if l.strip()]
        routine = [l for l in src if l.startswith("#") and ("->" in l or "footprint" in l.lower() or "Route" in l)]
        routine += [l for l in src if l.startswith(("for ", "    ", "policy.")) and "([" not in l]
        routine = routine[:10]
        code_block(d, (x0, y, x1, y1), routine, 14, ramp(t, 18.6, 2.0))
        y += len(routine) * int(14 * 1.45) + 16
        if t > 21.5:
            notes = plan["safety_notes"]
            notes = notes[notes.find("The swept path"):] if "The swept path" in notes else notes
            d.text((x0, y), "SAFETY CHECK (Wonder's own words)", font=F(16, 700), fill=DIM)
            text_block(d, (x0, y + 26), fit_sentences(notes, F(17, 400), x1 - x0, 6), F(17, 400), x1 - x0, TEXT,
                       (t - 21.6) / 2.5, 1.35, 6)


# -- 6 specular glare ------------------------------------------------------------------------
HOMOG = Homography.default(SRC_W, SRC_H)


def s_glare(img, d, t):
    s = S["specular_glare"]
    an = s.analysis
    diag = an["diagnosis"]
    t_anom, t_an, t_fix = s.ev("anomaly"), s.ev("analysis"), s.ev("fix_applied")
    clip = Clip([(0, 5.0, 0.0, t_anom), (5.0, 8.0, t_anom, t_an), (8.0, 15.0, t_an, t_fix), (15.0, 22.0, t_fix, s.t[-1])])
    ts = clip.src(t)
    tgt = next(a for a in an["attributions"] if a["label"] == "target")
    bias = an["evidence"]["target_grasp_bias_mm"]
    tx, ty = tgt["workspace_xy_mm"]
    true_px = HOMOG.workspace_to_pixel(tx, ty)
    seen_px = HOMOG.workspace_to_pixel(tx + bias[0], ty + bias[1])

    def bias_arrow(dr, k):
        if ts < t_an or t >= 15.0:
            return
        (ux, uy), (vx, vy) = [(p[0] * k, p[1] * k) for p in (seen_px, true_px)]
        dr.ellipse((ux - 9, uy - 9, ux + 9, uy + 9), outline=BAD + (255,), width=4)
        dr.ellipse((vx - 9, vy - 9, vx + 9, vy + 9), outline=GOOD + (255,), width=4)
        dr.line([(ux, uy), (vx, vy)], fill=(255, 255, 255, 255), width=3)
        for (lx, ly), label, c in [((ux - 250, uy - 12), "policy's grasp point", BAD),
                                   ((vx + 62, vy - 12), "SAM 3.1 centroid", GOOD)]:
            lw = F(19, 700).getlength(label)
            dr.rounded_rectangle((lx - 8, ly - 4, lx + lw + 8, ly + 26), radius=6, fill=(10, 14, 24, 215))
            dr.text((lx, ly), label, font=F(19, 700), fill=c)

    img.paste(footage(s, ts, sam=t_an <= ts and t < 15, culprit=None, speed=clip.speed(t), extra=bias_arrow), VIDEO_BOX[:2])
    if t < 15:
        hold_badge(d, s, ts)
    box = panel(d, PANEL_BOX, "glare shifts the grasp", MODEL["Muse Spark"])
    x0, y0, x1, y1 = box
    if t < 5.0:
        text_block(d, (x0, y0), "A highlight washes out half the cube. The washed-out pixels carry no saliency, so the "
                   "policy's idea of the cube's centre shifts, and every grasp closes on the edge.", F(23, 500), x1 - x0, TEXT,
                   ramp(t, 0.4, 2.5))
    elif t < 15.0:
        y = pipeline_rows(d, box, s.pipeline(), ts, only={"SAM 3.1", "A(i)", "Muse Spark"})
        if ts >= t_an:
            diag_card(d, (x0, y + 10, x1, y1), diag, an["diagnosis_source"], (t - clip_t(clip, t_an)) / 4, max_root_lines=6)
    else:
        fix_banner(d, f"RETRY_GRASP_OFFSET · grasp target shifted ({diag['intervention_parameters'].get('dx_mm', 0):+.1f}, "
                   f"{diag['intervention_parameters'].get('dy_mm', 0):+.1f}) mm", GOOD)
        y = y0
        for a, b in [("perceived grasp bias", f"{np.hypot(*bias):.1f} mm"), ("threshold", "8 mm"),
                     ("correction", f"dy = {diag['intervention_parameters'].get('dy_mm', 0):+.1f} mm"),
                     ("result", "cube picked")]:
            d.text((x0, y), a, font=F(22, 500), fill=DIM)
            d.text((x1 - F(23, 700, mono=True).getlength(b), y), b, font=F(23, 700, mono=True), fill=TEXT)
            y += 46
        text_block(d, (x0, y + 10), "Same policy, same weights: GlassBox only moved the target it reaches for.",
                   F(21, 400), x1 - x0, DIM)


# -- 7 occlusion -----------------------------------------------------------------------------
def s_occlusion(img, d, t):
    s = S["occlusion"]
    an = s.analysis
    diag = an["diagnosis"]
    t_anom, t_an, t_fix = s.ev("anomaly"), s.ev("analysis"), s.ev("fix_applied")
    clip = Clip([(0, 2.0, 0.0, t_anom), (2.0, 9.0, t_anom, t_fix), (9.0, 15.0, t_fix, s.t[-1])])
    ts = clip.src(t)
    img.paste(footage(s, ts, sam=t_an <= ts and t < 9, culprit=diag["culprit_mask_id"], speed=clip.speed(t)), VIDEO_BOX[:2])
    box = panel(d, PANEL_BOX, "when not to act", WARN)
    x0, y0, x1, y1 = box
    if t < 9.0:
        hold_badge(d, s, ts)
        y = pipeline_rows(d, box, s.pipeline(), ts, only={"SAM 3.1", "A(i)", "Muse Spark"})
        if ts >= t_an:
            y = diag_card(d, (x0, y + 10, x1, y1), diag, an["diagnosis_source"], (t - clip_t(clip, t_an)) / 3, root_cause=False)
            text_block(d, (x0, y + 14), "Erasing the hand cannot reveal the cube underneath, so no automatic fix is "
                       "offered: the operator clears the workspace.", F(21, 400), x1 - x0, DIM, ramp(t, 6.5, 1.5))
    else:
        fix_banner(d, "operator cleared the workspace · policy resumes", GOOD)
        text_block(d, box[:2], "The arm stayed on HOLD the whole time: a confident model never overrides an "
                   "out-of-distribution scene it cannot see through.", F(23, 500), x1 - x0, TEXT, ramp(t, 9.2, 1.5))


# -- 8 real desk -----------------------------------------------------------------------------
def s_webcam(img, d, t):
    s = S["webcam"]
    an = s.analysis
    t_start, t_an = s.ev("analysis_started"), s.ev("analysis")
    clip = Clip([(0, 5.0, 2.0, 7.0), (5.0, 12.0, t_start, t_an), (12.0, 18.0, t_an, t_an + 6)])
    ts = clip.src(t)
    img.paste(footage(s, ts, live=False, sam=ts >= t_an, speed=clip.speed(t)), VIDEO_BOX[:2])
    box = panel(d, PANEL_BOX, "my desk · real SO-101 · overhead webcam", MODEL["SAM 3.1"])
    x0, y0, x1, y1 = box
    if t < 5.0:
        text_block(d, (x0, y0), "Camera-only session: the real arm on the real desk. With no policy attached, the "
                   "heatmap is spectral-residual visual saliency, and the UI says so.", F(23, 500), x1 - x0, TEXT,
                   ramp(t, 0.3, 2.0))
    else:
        y = pipeline_rows(d, box, s.pipeline(), ts, only={"SAM 3.1", "A(i)", "Muse Spark"})
        if ts >= t_an:
            y += 8
            for o in an["objects"]:
                txt = f"#{o['mask_id']}  \"{o['prompt']}\"   conf {o['confidence']:.2f}"
                d.text((x0, y), txt, font=F(21, 600), fill=ROLE.get(o["label"], ACCENT))
                y += 34
            y += 14
            d.text((x0, y), "MUSE SPARK READS THE SCENE", font=F(17, 700), fill=MODEL["Muse Spark"])
            text_block(d, (x0, y + 32), s.ask["answer"].replace("\n\n", " "), F(20, 400), x1 - x0, TEXT,
                       (t - 12.2) / 3.5, 1.38, 10)


# -- 9 engineering ---------------------------------------------------------------------------
TILES = [
    ("30 Hz", "Control loop never awaits the network. Analysis runs beside it while the arm HOLDs safely.", GOOD),
    ("4 triggers", "Servo following error, action-chunk variance, attention share A(i), no-progress watchdog.", BAD),
    ("JSON schema", "Spark's output is schema-constrained, then validated: an unknown mask id never reaches the arm.",
     MODEL["Muse Spark"]),
    ("≥ 12.5 mm", "Every bypass is forward-simulated against the SAM obstacle footprint; bad plans fall back to the local planner.",
     MODEL["Muse Wonder"]),
    ("Offline-safe", "Disk cache replays API calls and every model has a deterministic local fallback.", ACCENT),
    ("10 s replay", "Ring buffer of frames, saliency and servo telemetry; epoch guards drop work that straddles a reset.",
     MODEL["Muse Image"]),
]


def s_eng(img, d, t):
    cw, ch, gap = 560, 270, 30
    x0 = (W - 3 * cw - 2 * gap) // 2
    for n, (head, body, c) in enumerate(TILES):
        a = ramp(t, 0.3 + n * 0.55, 0.45)
        if a <= 0:
            continue
        x = x0 + (n % 3) * (cw + gap)
        y = 170 + (n // 3) * (ch + gap)
        d.rounded_rectangle((x, y, x + cw, y + ch), radius=16, fill=mix(BG, PANEL, a), outline=mix(BG, c, a), width=2)
        d.text((x + 28, y + 26), head, font=F(48, 800, symbols="≥" in head), fill=mix(BG, c, a))
        text_block(d, (x + 28, y + 104), body, F(24, 400), cw - 56, mix(BG, TEXT, a), 1, 1.35)


# -- 10 closing ------------------------------------------------------------------------------
def s_close(img, d, t):
    a = ramp(t, 0.3, 0.7)
    d.text((140, 150), "From “it stopped”", font=F(84, 800), fill=mix(BG, TEXT, a))
    b = ramp(t, 1.1, 0.7)
    d.text((140, 250), "to “here's why, and here's the fix”.", font=F(84, 800), fill=mix(BG, ACCENT, b))
    c = ramp(t, 2.4, 0.7)
    for k, line in enumerate(["Any low-cost arm with a webcam: makers, classrooms, research labs.",
                              "Fixes act on the policy's inputs and trajectory, live. No retraining.",
                              "Every step is visible, measured and checkable."]):
        e = ramp(t, 2.4 + k * 0.6, 0.6)
        d.text((146, 410 + k * 58), line, font=F(34, 400), fill=mix(BG, DIM, e))
    _ = c
    f = ramp(t, 5.6, 0.7)
    d.text((146, 620), "GlassBox-SO101", font=F(44, 800), fill=mix(BG, TEXT, f))
    d.text((146, 680), "Built on the Meta Model API", font=F(24, 500), fill=mix(BG, DIM, f))
    models_row(d, 146, 722, list(MODEL), 22, t - 5.8, 0.15)


def build_scenes() -> list[Scene]:
    return [
        Scene("title", "", "", 7.0, s_title, [(0.8, 6.6, "What if a robot could tell you why it failed, and how to fix it?")]),
        Scene("problem", "01", "Learned policies fail silently", 12.0, s_problem, [
            (0.3, 5.8, "This SO-101 arm runs an imitation-learned policy: pick up the red cube, drop it in the bin."),
            (6.0, 11.8, "On a real desk, policies like this fail silently. The usual fix is more demos and a retrain."),
        ]),
        Scene("arch", "02", "How GlassBox works", 16.0, s_arch, [
            (0.3, 5.2, "GlassBox wraps the policy in a 30 Hz safety loop that reads its attention and its servos every tick."),
            (5.4, 10.8, "When a trigger fires, the arm holds and an async pipeline of Meta models works out why."),
            (11.0, 15.8, "Every diagnosis maps to a concrete fix, validated before it is allowed to move the arm."),
        ], list(MODEL)),
        Scene("models", "03", "Five Meta models, five jobs", 14.0, s_models, [
            (0.3, 6.8, "SAM 3.1 names the scene. Muse Spark diagnoses. Muse Wonder plans recoveries."),
            (7.0, 13.8, "Muse Image renders the counterfactual, and Muse Voice Transcribe keeps the operator hands-free."),
        ], list(MODEL)),
        Scene("hijack", "04", "Failure 1 · a look-alike steals the policy's attention", 42.0, s_hijack, [
            (0.2, 3.1, "Failure one: a red ball shares the cube's colour cue."),
            (3.3, 8.6, "Measured inside SAM 3.1 masks, the policy's own attention puts 58% on the ball and 35% on the cube."),
            (8.8, 14.8, "The arm holds. Muse Spark reads the heatmap and the numbers, rules out a stall and glare: attention hijack, mask 1."),
            (15.0, 19.3, "This is exactly what Spark saw, alongside the evidence JSON on the right."),
            (19.5, 23.8, "Muse Image renders the scene the policy should have seen."),
            (24.0, 32.8, "Ask why, by voice or by text. Spark answers from the same evidence, numbers and all."),
            (33.0, 37.4, "One click: the ball is erased from the policy's input on every frame."),
            (37.6, 41.8, "No retraining. Attention returns to the cube and the task completes."),
        ], ["SAM 3.1", "Muse Spark", "Muse Image", "Muse Voice Transcribe"]),
        Scene("stall", "05", "Failure 2 · a box on the reach path", 35.0, s_stall, [
            (0.2, 4.9, "Failure two: a box on the reach path. Joint error and load climb as the gripper jams."),
            (5.1, 10.8, "The no-progress watchdog fires. Spark cites a 15° sustained error, 39% load and the gripper touching mask 2."),
            (11.0, 16.8, "A local planner has a validated bypass ready at once. Muse Wonder gets the kinematics and telemetry to check it."),
            (17.0, 21.4, "Wonder verifies the swept path and joint speeds, explains the route, and writes the routine the executor runs."),
            (21.6, 25.8, "Nothing moves until every waypoint is forward-simulated clear of the obstacle's SAM footprint."),
            (26.0, 30.4, "Apply: the arm backs off, routes around the box and hands control back to the policy."),
            (30.6, 34.8, "Recovered, not restarted."),
        ], ["SAM 3.1", "Muse Spark", "Muse Wonder", "Muse Image"]),
        Scene("glare", "06", "Failure 3 · glare shifts the grasp point", 22.0, s_glare, [
            (0.2, 4.8, "Failure three: glare washes out half the cube, so every grasp lands off-centre."),
            (5.0, 14.8, "Spark compares where the policy looks with SAM's mask centroid: a 16 mm bias. The fix is a grasp offset."),
            (15.0, 21.8, "The same policy retries with the corrected target and makes the pick."),
        ], ["SAM 3.1", "Muse Spark"]),
        Scene("occlusion", "07", "Failure 4 · a hand covers the cube", 15.0, s_occlusion, [
            (0.2, 3.8, "Failure four: a hand covers the cube."),
            (4.0, 8.8, "Erasing the hand can't reveal what's under it, so GlassBox doesn't guess. It asks the operator."),
            (9.0, 14.8, "Knowing when not to act is part of the design. Once the workspace is clear, the policy resumes."),
        ], ["SAM 3.1", "Muse Spark"]),
        Scene("webcam", "08", "The same pipeline on my real desk", 18.0, s_webcam, [
            (0.2, 4.9, "And this is my actual desk, from the overhead webcam."),
            (5.1, 11.8, "Same pipeline: SAM 3.1 finds the real SO-101 and the cube from plain-English prompts."),
            (12.0, 17.8, "Spark reads the live scene: no failure, and the arm is clear of the cube."),
        ], ["SAM 3.1", "Muse Spark"]),
        Scene("eng", "09", "Built to run on a real robot", 16.0, s_eng, [
            (0.3, 7.8, "Under the hood: a loop that never waits on the network, schema-validated outputs and forward-simulated plans."),
            (8.0, 15.8, "Cached replays and local fallbacks for every model, so the demo survives when the Wi-Fi doesn't."),
        ]),
        Scene("close", "", "", 11.0, s_close, [
            (0.3, 5.3, "Any low-cost arm with a webcam can now explain its failures and recover from them."),
            (5.5, 10.8, "GlassBox: see what the policy sees, name the failure, fix it live."),
        ]),
    ]


# ----------------------------------------------------------------------------------- compositor
def caption(img: Image.Image, text: str, alpha: float):
    font = F(38, 500)
    lines = wrap(text, font, 1500)
    if len(lines) > 2:
        font = F(33, 500)
        lines = wrap(text, font, 1600)
    lh = int(font.size * 1.3)
    w = max(font.getlength(l) for l in lines) + 70
    h = lh * len(lines) + 34
    layer = Image.new("RGBA", (int(w), h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((0, 0, w - 1, h - 1), radius=18, fill=(0, 0, 0, 200), outline=(60, 72, 100, 255), width=2)
    for k, l in enumerate(lines):
        ld.text(((w - font.getlength(l)) / 2, 16 + k * lh), l, font=font, fill=(250, 252, 255, 255))
    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
    y = 905 - h // 2
    img.paste(layer, ((W - int(w)) // 2, y), layer)


def header(d, scene: Scene, t: float):
    if not scene.chapter:
        return
    a = ramp(t, 0.0, 0.4)
    d.text((48, 34), scene.chapter, font=F(30, 800), fill=mix(BG, ACCENT, a))
    d.text((104, 34), scene.title, font=F(40, 700), fill=mix(BG, TEXT, a))
    names = scene.models
    if names:
        fsz = 18
        total = sum(F(fsz, 600).getlength(n) + 24 + 10 for n in names)
        x = 1872 - total
        d.text((x - 120, 46), "models used", font=F(17, 500), fill=DIM)
        for n in names:
            x += chip(d, (x, 40), n, MODEL[n], fsz) + 10


def timeline(d, scenes, t_global, total):
    y = 1046
    d.rectangle((48, y, 1872, y + 4), fill=(28, 36, 54))
    d.rectangle((48, y, 48 + 1824 * t_global / total, y + 4), fill=ACCENT)
    acc = 0.0
    for sc in scenes[:-1]:
        acc += sc.duration
        x = 48 + 1824 * acc / total
        d.rectangle((x - 1, y - 4, x + 1, y + 8), fill=EDGE)


def render_frame(scenes, si: int, t: float, t_global: float, total: float) -> Image.Image:
    sc = scenes[si]
    img = BACKGROUND.copy()
    d = ImageDraw.Draw(img)
    sc.draw(img, d, t)
    d = ImageDraw.Draw(img)
    header(d, sc, t)
    for c0, c1, text in sc.captions:
        if c0 <= t <= c1:
            caption(img, text, min(ramp(t, c0, 0.25), 1 - ramp(t, c1 - 0.25, 0.25)))
    timeline(d, scenes, t_global, total)
    fade = min(ramp(t, 0, 0.35), 1 - ramp(t, sc.duration - 0.3, 0.3))
    if si == 0:
        fade = min(fade, ramp(t, 0, 1.0))
    if fade < 1:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, max(0.0, fade))
    return img


def fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def write_script(scenes):
    out = ["# Explainer video: narration script",
           "",
           "Generated by `docs/video/make_video.py` together with the video, so the timings match the",
           "on-screen captions exactly. Read it as a voice-over, or let the captions carry it.",
           ""]
    acc = 0.0
    for sc in scenes:
        head = f"{sc.chapter} · {sc.title}" if sc.chapter else sc.key.capitalize()
        out += [f"## {fmt(acc)} · {head}", "", "| Time | Narration |", "|---|---|"]
        for c0, c1, text in sc.captions:
            out.append(f"| {fmt(acc + c0)} to {fmt(acc + c1)} | {text} |")
        out.append("")
        acc += sc.duration
    out.append(f"Total running time: {fmt(acc)}.")
    OUT_SCRIPT.write_text("\n".join(out) + "\n")


def main():
    global BACKGROUND
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs=2, metavar=("SCENE", "T"), help="write one frame to /tmp/glassbox_preview.png")
    ap.add_argument("--out", default=str(OUT_MP4))
    ap.add_argument("--only", nargs="*", help="render a subset of scenes (for iteration)")
    a = ap.parse_args()
    load_sessions()
    BACKGROUND = background()
    scenes = build_scenes()
    total = sum(s.duration for s in scenes)
    if a.preview:
        key, t = a.preview[0], float(a.preview[1])
        si = next(k for k, s in enumerate(scenes) if s.key == key)
        start = sum(s.duration for s in scenes[:si])
        render_frame(scenes, si, t, start + t, total).save("/tmp/glassbox_preview.png")
        print("/tmp/glassbox_preview.png")
        return
    write_script(scenes)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         a.out],
        stdin=subprocess.PIPE,
    )
    acc = 0.0
    for si, sc in enumerate(scenes):
        if a.only and sc.key not in a.only:
            acc += sc.duration
            continue
        n = int(round(sc.duration * FPS))
        for k in range(n):
            t = k / FPS
            ff.stdin.write(render_frame(scenes, si, t, acc + t, total).tobytes())
        acc += sc.duration
        print(f"  rendered {sc.key:10s} ({sc.duration:.0f} s), {fmt(acc)} / {fmt(total)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {a.out} and {OUT_SCRIPT}")


if __name__ == "__main__":
    main()
