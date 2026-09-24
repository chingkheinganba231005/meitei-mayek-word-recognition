"""Compose an image of a handwritten Meitei Mayek word from character images.

For a word in everyday spelling:

1. Word style: letter height, letter width, spacing, slant, pen width, size of the signs,
   ink and paper, and a style anchor for choosing images (``GlyphStore.pick``).
2. Layout, in units of L (the height of a letter; y up, baseline at 0), from the font
   priors (``assets/glyph_priors.json``, ``scripts/glyph_priors.py``). Letters, lonsum
   letters and digits stand on the baseline one after another, with a gap. A sign is
   placed relative to the pen position after the character before it, as the font does:
   ꯥ, ꯩ and ꯪ above that character, ꯨ below it, ꯦ, ꯣ, ꯧ and ꯤ beside it; apun under the
   whole letter before it. Every size and position is jittered, and the baseline drifts.
3. Each character image (24 x 24, ink stretched to the frame) is resized into its box,
   which gives back the proportions TUMMHCD lost, and its strokes are thickened or
   thinned to the word's pen width, since resizing scales the strokes with the box.
4. The word is slanted and rotated a little, blurred a little, and drawn in ink on paper.

Everything random comes from the numpy Generator passed in, so a seed fixes an image.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .charset import CHEIKHEI, normalise, renderable
from .glyphs import ASSETS


@dataclass
class Config:
    letter_height: float = 32.0         # L in pixels
    letter_height_jitter: float = 0.1   # per word, sd of log
    width: tuple = (0.8, 1.25)          # letter width factor per word
    glyph_width_jitter: float = 0.08    # per character, sd of log
    glyph_height_jitter: float = 0.06
    gap: tuple = (0.02, 0.3)            # mean gap between characters per word, in L
    gap_jitter: float = 0.05            # per gap, in L
    baseline_jitter: float = 0.04       # drift of the baseline, in L
    mark_scale: tuple = (0.85, 1.3)     # size of the signs relative to print, per word
    mark_size_jitter: float = 0.1       # per sign, sd of log
    mark_jitter: float = 0.05           # position of a sign, in L
    slant: float = 0.12                 # sd of the shear (tan of the angle), clipped at 2.5 sd
    rotation: float = 1.5               # sd in degrees, clipped at 2.5 sd
    pen: tuple = (0.07, 0.12)           # pen width per word, in L; None keeps the images' strokes
    style_k: int = 16                   # choose among the k images closest in style; None: any
    blur: tuple = (0.0, 0.8)            # gaussian sigma in px, per word
    noise: float = 3.0                  # sd of pixel noise in grey levels
    paper: tuple = (225.0, 255.0)
    ink: tuple = None                   # ink grey level range; None: from the chosen images
    margin: tuple = (0.1, 0.35)         # around the word, in L


class Sample(NamedTuple):
    image: np.ndarray        # uint8, dark ink on light paper
    text: str                # the word as rendered (everyday spelling)
    boxes: list              # (character, x0, y0, x1, y1): each character's box in the image, clipped to it
    glyphs: list             # index of each character's image in the store
    style: dict              # the word-level parameters


def load_priors(path=None, sizes=None, clip=(0.5, 2.0)):
    """Font priors by character; with `sizes` (``scripts/glyph_sizes.py`` output), the width
    and height measured on TUMMHCD replace the font's where available, clipped to `clip`
    times the font's. Positions stay the font's: an above sign keeps its bottom, a sign
    below keeps its top, a letter or a sign standing on the baseline keeps its bottom."""
    rows = json.loads(Path(path or ASSETS / "glyph_priors.json").read_text(encoding="utf-8"))["classes"]
    prior = {r["char"]: dict(r) for r in rows}
    if sizes:
        measured = json.loads(Path(sizes).read_text(encoding="utf-8"))["classes"]
        for r in measured:
            p = prior[r["char"]]
            h0 = p["top"] - p["bottom"]
            if r.get("w"):
                p["w"] = float(np.clip(r["w"], clip[0] * p["w"], clip[1] * p["w"]))
            if r.get("h"):
                h = float(np.clip(r["h"], clip[0] * h0, clip[1] * h0))
                if p["kind"] == "mark" and p["bottom"] < 0.9 and abs(p["bottom"]) >= 0.05:
                    p["bottom"] = p["top"] - h
                else:
                    p["top"] = p["bottom"] + h
    return prior


def stroke_width(mask):
    """Mean stroke width in pixels: twice the ink area over the number of boundary pixels."""
    n = int(mask.sum())
    if n == 0:
        return 0.0
    pad = np.pad(mask, 1)
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return 2.0 * n / max(n - int(interior.sum()), 1)


def set_pen(alpha, pen):
    """Grow or shrink the strokes of an ink map to about `pen` pixels.

    Returns (ink map, border): the map is re-drawn from the signed distance to the stroke
    edge, moved by half the difference in width, and has grown by `border` pixels on each
    side to make room for thicker strokes.
    """
    mask = alpha > 0.5
    if pen is None or mask.sum() < 3:
        return alpha, 0
    have = stroke_width(mask)
    delta = max((pen - have) / 2, -(have - 1) / 2)  # never thinner than about one pixel
    if abs(delta) < 0.25:
        return alpha, 0
    border = int(math.ceil(max(delta, 0))) + 1
    mask = np.pad(mask, border)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    signed = np.where(mask, inside - 0.5, 0.5 - outside)
    return np.clip(signed + delta + 0.5, 0, 1).astype(np.float32), border


def resize(alpha, w, h):
    return np.clip(np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize((w, h), Image.BILINEAR)), 0, 1)


class WordSynth:
    def __init__(self, store, priors=None, config=None):
        self.store = store
        self.prior = priors or load_priors()
        self.cfg = config or Config()

    # ------------------------------------------------------------------ style and layout

    def word_style(self, rng):
        c = self.cfg

        def uniform(r):
            return float(rng.uniform(*r))

        return {"L": c.letter_height * float(np.exp(rng.normal(0, c.letter_height_jitter))),
                "width": float(np.exp(rng.uniform(np.log(c.width[0]), np.log(c.width[1])))),
                "gap": uniform(c.gap),
                "mark_scale": uniform(c.mark_scale),
                "slant": float(np.clip(rng.normal(0, c.slant), -2.5 * c.slant, 2.5 * c.slant)),
                "rotation": float(np.clip(rng.normal(0, c.rotation), -2.5 * c.rotation, 2.5 * c.rotation)),
                "pen": uniform(c.pen) if c.pen else None,
                "blur": uniform(c.blur),
                "paper": uniform(c.paper),
                "ink": uniform(c.ink) if c.ink else None}

    def layout(self, word, rng, st):
        """-> [(character, x0, x1, bottom, top)] in units of L (y up, baseline at 0)."""
        c = self.cfg

        def jitter(sd):
            return float(np.exp(rng.normal(0, sd)))

        boxes, pen, drift, base = [], 0.0, 0.0, None
        for ch in word:
            p = self.prior[ch]
            if p["kind"] == "base" or base is None:
                drift = 0.6 * drift + float(rng.normal(0, c.baseline_jitter))
                w = p["w"] * st["width"] * jitter(c.glyph_width_jitter)
                h = (p["top"] - p["bottom"]) * jitter(c.glyph_height_jitter)
                gap = st["gap"] + float(rng.normal(0, c.gap_jitter)) if boxes else 0.0
                x0 = pen + p.get("lsb", 0.05) + max(gap, -0.1)
                bottom = p["bottom"] * h / max(p["top"] - p["bottom"], 1e-6) + drift if p["kind"] == "base" else drift
                box = [x0, x0 + w, bottom, bottom + h]
                pen = x0 + w + p.get("rsb", 0.05)
                base = box
            else:
                s = st["mark_scale"] * jitter(c.mark_size_jitter)
                w, h = p["w"] * s, (p["top"] - p["bottom"]) * s
                if p.get("span"):  # apun: under the whole letter before it
                    x0 = base[0] + float(rng.normal(0, c.mark_jitter))
                    x1 = base[1] + float(rng.normal(0, c.mark_jitter))
                    w = max(x1 - x0, 0.3)
                else:
                    x0 = pen + p["off"] + float(rng.normal(0, c.mark_jitter))
                dy = drift + float(rng.normal(0, c.mark_jitter))
                if p["bottom"] >= 0.9 or abs(p["bottom"]) < 0.05:  # above, or standing on the baseline
                    bottom = p["bottom"] + dy
                else:                                              # hanging from its top
                    bottom = p["top"] + dy - h
                box = [x0, x0 + w, bottom, bottom + h]
                if p["adv"] > 0.1:
                    pen = max(pen + p["adv"] * s, x0 + w)
            boxes.append((ch, *box))
        return boxes

    # ------------------------------------------------------------------ drawing

    def render(self, text, rng):
        word = normalise(text)
        if not renderable(word):
            raise ValueError(f"cannot render {text!r}: characters outside the alphabet")
        c, st = self.cfg, self.word_style(rng)
        L = st["L"]
        anchor = self.store.anchor(rng) if c.style_k else None
        glyphs = [self.store.pick(ch, rng, anchor, c.style_k) for ch in word]
        boxes = self.layout(word, rng, st)

        xmin = min(b[1] for b in boxes)
        xmax = max(b[2] for b in boxes)
        ymin = min(b[3] for b in boxes)
        ymax = max(b[4] for b in boxes)
        pad = 1.0 + 2.5 * c.slant + 0.5  # room for slant and rotation; cropped away below
        W = int(math.ceil((xmax - xmin + 2 * pad) * L))
        H = int(math.ceil((ymax - ymin + 2 * pad) * L))
        canvas = np.zeros((H, W), np.float32)
        placed = []
        for (ch, x0, x1, bottom, top), g in zip(boxes, glyphs):
            w = max(int(round((x1 - x0) * L)), 1)
            h = max(int(round((top - bottom) * L)), 1)
            ink = resize(self.store.alpha[g], w, h)
            px, py = int(round((x0 - xmin + pad) * L)), int(round((ymax + pad - top) * L))
            placed.append((ch, px, py, px + w, py + h))
            if st["pen"] is not None and min(w, h) >= 4:
                ink, border = set_pen(ink, st["pen"] * L * float(np.exp(rng.normal(0, 0.05))))
                px, py = px - border, py - border
            assert px >= 0 and py >= 0 and px + ink.shape[1] <= W and py + ink.shape[0] <= H, "canvas too small"
            region = canvas[py:py + ink.shape[0], px:px + ink.shape[1]]
            np.maximum(region, ink, out=region)

        baseline = (ymax + pad) * L
        canvas, placed = self.transform(canvas, placed, st, baseline)
        canvas, placed = self.crop(canvas, placed, rng, L)
        if st["blur"] > 0.05:
            canvas = ndimage.gaussian_filter(canvas, st["blur"])
        ink_level = st["ink"] if st["ink"] is not None else float(np.mean(
            [self.store.raw_style[g][3] for g in glyphs]))
        img = st["paper"] - canvas * (st["paper"] - ink_level)
        if c.noise:
            img = img + rng.normal(0, c.noise, img.shape)
        st["ink"] = ink_level
        return Sample(np.clip(np.round(img), 0, 255).astype(np.uint8), word, placed, glyphs, st)

    @staticmethod
    def transform(canvas, placed, st, baseline):
        """Slant (shear about the baseline: the top leans right when slant > 0), then rotate
        about the centre (clockwise when rotation > 0, image y pointing down); boxes follow."""
        H, W = canvas.shape
        s, a = st["slant"], math.radians(st["rotation"])
        c = np.array([W / 2, H / 2])
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        A = rot @ np.array([[1.0, -s], [0.0, 1.0]])              # image point p -> A p + t
        t = rot @ (np.array([s * baseline, 0.0]) - c) + c
        inv = np.linalg.inv(A)                                    # PIL wants output -> input
        coeffs = (*inv[0], -(inv[0] @ t), *inv[1], -(inv[1] @ t))
        out = Image.fromarray(canvas, "F").transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR)
        moved = []
        for ch, x0, y0, x1, y1 in placed:
            pts = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)]) @ A.T + t
            moved.append((ch, *pts.min(0).tolist(), *pts.max(0).tolist()))
        return np.asarray(out, np.float32), moved

    def crop(self, canvas, placed, rng, L):
        ys, xs = np.nonzero(canvas > 0.02)
        if len(xs) == 0:
            return canvas, placed
        m = [int(round(rng.uniform(*self.cfg.margin) * L)) for _ in range(4)]
        y0, y1 = max(ys.min() - m[0], 0), min(ys.max() + 1 + m[1], canvas.shape[0])
        x0, x1 = max(xs.min() - m[2], 0), min(xs.max() + 1 + m[3], canvas.shape[1])
        out = canvas[y0:y1, x0:x1].copy()
        H, W = out.shape
        boxes = [(ch, min(max(bx0 - x0, 0), W), min(max(by0 - y0, 0), H), min(max(bx1 - x0, 0), W),
                  min(max(by1 - y0, 0), H)) for ch, bx0, by0, bx1, by1 in placed]
        return out, boxes


class Words:
    """Word images on demand. Item i is always the same image: its random numbers come from
    the seed sequence (seed, i). A word is drawn from the lexicon, or with probability
    `numbers` a number in Meitei Mayek digits; with probability `stop` a full stop
    (cheikhei) follows it."""

    def __init__(self, synth, lexicon, seed=0, numbers=0.03, stop=0.02):
        self.synth, self.lexicon, self.seed = synth, lexicon, seed
        self.numbers, self.stop = numbers, stop

    def text(self, rng):
        from .lexicon import number

        text = number(rng) if rng.random() < self.numbers else self.lexicon.sample(rng)
        return text + CHEIKHEI if rng.random() < self.stop else text

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, int(i)])
        return self.synth.render(self.text(rng), rng)
