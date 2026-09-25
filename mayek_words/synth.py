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

from .charset import CHEIKHEI, normalise, renderable, syllables
from .glyphs import ASSETS


@dataclass
class Config:
    letter_height: float = 32.0         # L in pixels
    letter_height_jitter: float = 0.1   # per word, sd of log
    width: tuple = (0.8, 1.25)          # letter width factor per word
    glyph_width_jitter: float = 0.08    # per character, sd of log
    glyph_height_jitter: float = 0.06
    gap: tuple = (-0.10, 0.05)          # per word, added to the font's spacing, in L
    gap_jitter: float = 0.03
    max_overlap: float = 0.05           # neighbours' boxes may overlap this much, in L
    uneven: tuple = (0.0, 0.12)         # most handwriting is evenly spaced; in some words (p_uneven)
    p_uneven: float = 0.2               # syllables stand apart: gaps inside a syllable narrower, between
    #                                     syllables wider, by this, in L (the owner's observation)
    touch: tuple = (0.02, 0.5)          # per word, the chance that a character joins the one before it
    touch_between: float = 0.5          # inside a syllable; between syllables times this. A joined
    #                                     character is slid left until its ink meets the ink before it
    #                                     (handwriting joins about a third of neighbouring letters:
    #                                     results/spacing_web_samples.json)
    # A sign beside its letter (ꯤ, ꯦ, ꯣ, ꯧ), placed on the ink (owner, 25 September 2026): never
    # nearer the next letter than its own, and the next letter never joins it. In evenly spaced
    # words (most) it keeps the layout's gap from its letter; in unevenly spaced words it is
    # almost stuck to its letter and the next syllable keeps further away. Distances are
    # measured from the sign's body (sign_body: the lead-in stroke of ꯤ and ꯧ is left out).
    attach_gap: tuple = (0.02, 0.06)    # uneven words: the sign's body this far from its letter, in L
    sign_touch: float = 0.1             # uneven words: chance that the sign touches its letter instead
    lead_in: float = 0.1                # a touching sign may reach this far past its letter's ink in the
    #                                     same row, in L; otherwise a lead-in may meet the letter, not cross it
    overhang: tuple = (0.15, 0.3)       # no part of the sign starts further than this left of its letter's
    #                                     rightmost ink (a lead-in over the letter), in L: in even words,
    #                                     in uneven words (where the sign hugs its letter)
    sign_margin: float = 0.04           # uneven words: the next letter this to twice this further from
    #                                     the sign than the sign is from its letter, in L
    baseline_jitter: float = 0.04       # drift of the baseline, in L
    mark_scale: tuple = (0.85, 1.3)     # size of the signs relative to print, per word
    mark_size_jitter: float = 0.1       # per sign, sd of log
    mark_jitter: float = 0.05           # position of a sign, in L
    slant: float = 0.12                 # sd of the shear (tan of the angle), clipped at 2.5 sd
    rotation: float = 1.5               # sd in degrees, clipped at 2.5 sd
    pen: tuple = (0.06, 0.14)           # pen width per word, in L (TUMMHCD's scanned strokes are about
    #                                     0.07; photos of handwriting look thicker; owner, 25 September
    #                                     2026); None keeps the images' strokes
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
    layout: list             # (character, x0, x1, bottom, top) in units of L, before slant and rotation


MEASURED = ASSETS / "glyph_sizes_tummhcd.json"  # copy of results/glyph_sizes_tummhcd.json, first run


def load_priors(path=None, sizes=MEASURED, clip=(0.5, 2.0)):
    """Priors by character: the font's positions, and by default the width and height
    measured on TUMMHCD (``scripts/glyph_sizes.py``; the owner's choice, 25 September 2026),
    where available, clipped to `clip` times the font's; sizes=None keeps the font's sizes.
    An above sign keeps its bottom, a sign below keeps its top, a letter or a sign standing
    on the baseline keeps its bottom."""
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


def line_gaps(layouts, prior):
    """Gaps between neighbours on the line, in units of L, from Sample.layout: letters,
    lonsum letters, digits and the signs written beside a letter (not above or below).
    -> {"letter to letter", "letter to sign beside it", "sign to next letter",
        "inside a syllable", "between syllables": [...]}"""
    out = {"letter to letter": [], "letter to sign beside it": [], "sign to next letter": [],
           "inside a syllable": [], "between syllables": []}
    for boxes in layouts:
        syl = syllables("".join(b[0] for b in boxes))
        line = [(b, n) for b, n in zip(boxes, syl) if prior[b[0]]["kind"] == "base" or prior[b[0]]["adv"] > 0.1]
        for (b1, n1), (b2, n2) in zip(line, line[1:]):
            k1, k2 = prior[b1[0]]["kind"], prior[b2[0]]["kind"]
            key = ("letter to letter" if k1 == k2 == "base" else
                   "letter to sign beside it" if k2 == "mark" else "sign to next letter")
            out[key].append(b2[1] - b1[2])
            out["inside a syllable" if n1 == n2 else "between syllables"].append(b2[1] - b1[2])
    return out


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


def contact(canvas, ink, qx, qy, reach):
    """How many pixels to move `ink` (to be pasted at column qx, row qy) to the left so that
    it just touches the ink already on the canvas; 0 if nothing is within `reach` pixels."""
    m = ink > 0.5
    best = None
    for r in np.flatnonzero(m.any(1)):
        left = qx + int(np.argmax(m[r]))
        lo = max(left - reach - 1, 0)
        seen = np.flatnonzero(canvas[qy + r, lo:left] > 0.5)
        if len(seen):
            gap = left - (lo + int(seen[-1])) - 1
            best = gap if best is None else min(best, gap)
    return max(best, 0) if best is not None else 0


def sign_body(ink, share=0.35):
    """The sign without its lead-in: handwritten ꯤ and ꯧ start with a thin stroke from the left
    (towards their letter), and the eye reads the sign where its stem or loop is. Columns to
    the left of the first column holding at least `share` of the fullest column's ink are
    cleared."""
    cols = (ink > 0.5).sum(0)
    if not cols.any():
        return ink
    first = int(np.argmax(cols >= share * cols.max()))
    out = ink.copy()
    out[:, :first] = 0
    return out


def reach_past(canvas, ink, qx, qy, reach):
    """How far, in pixels, `ink` (to be pasted at column qx, row qy) reaches to the left of the
    rightmost ink already on the canvas in the same row (looking up to `reach` to its left),
    at most over its rows; None if no row has ink before it."""
    lo = max(qx - reach, 0)
    window = canvas[qy:qy + ink.shape[0], lo:qx + ink.shape[1]] > 0.5
    mine = ink > 0.5
    rows = window.any(1) & mine.any(1)
    if not rows.any():
        return None
    right = window.shape[1] - 1 - np.argmax(window[rows, ::-1], 1)   # in window columns
    left = qx - lo + np.argmax(mine[rows], 1)
    return float((right - left + 1).max())


def ink_distance(canvas, ink, qx, qy, reach):
    """The shortest distance in pixels, in any direction, between `ink` (to be pasted at column
    qx, row qy) and the ink already on the canvas, looking up to `reach` to the left, above
    and below: 0 for neighbouring pixels, -1 if they overlap, None if nothing is there."""
    x0, y0 = max(qx - reach, 0), max(qy - reach, 0)
    window = canvas[y0:qy + ink.shape[0] + reach, x0:qx + ink.shape[1]] > 0.5
    mine = ink > 0.5
    if not window.any() or not mine.any():
        return None
    d = ndimage.distance_transform_edt(~window)[qy - y0:qy - y0 + ink.shape[0], qx - x0:][mine]
    return float(d.min()) - 1


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
                "touch": uniform(c.touch) if c.touch else 0.0,
                "uneven": uniform(c.uneven) if rng.random() < c.p_uneven else 0.0,
                "blur": uniform(c.blur),
                "paper": uniform(c.paper),
                "ink": uniform(c.ink) if c.ink else None}

    def layout(self, word, rng, st):
        """-> ([(character, x0, x1, bottom, top)] in units of L (y up, baseline at 0),
               [per character: "join" to be slid left until its ink meets the ink before it;
                "sign", "sign close" or "sign touching" for a sign beside its own letter (evenly
                spaced word; unevenly spaced word; the rare touching case); "after sign" for the
                first letter of the syllable after such a sign; or None],
               [per character on the line: the gap before it, in L; None for the others]).

        1. Every character is placed as the font places it, with its size jittered.
        2. The gaps between neighbours on the line (letters, lonsum letters, digits and the
           signs written beside a letter) are set: the font's spacing, plus the word's gap,
           plus jitter. A syllable is kept together (``charset.syllables``): no gap inside
           it is wider than the gaps between it and its neighbours, so a sign is never
           closer to the next letter than to its own. Most words are spaced evenly; in some
           (st["uneven"]) the syllables stand visibly apart.
        3. Joins: a character joins the one before it with chance st["touch"] inside a
           syllable, less often between syllables; when two syllables join, their insides
           are joined too. A sign beside its letter (ꯤ, ꯦ, ꯣ, ꯧ) and the letter after it are
           placed on the ink when drawn (render): evenly spaced, the sign never nearer the next
           letter than its own; in unevenly spaced words almost stuck to its letter.
        """
        c = self.cfg

        def jitter(sd):
            return float(np.exp(rng.normal(0, sd)))

        # 1. the font's placement
        boxes, line, pen, drift, base = [], [], 0.0, 0.0, None
        for i, ch in enumerate(word):
            p = self.prior[ch]
            if p["kind"] == "base" or base is None:
                drift = 0.6 * drift + float(rng.normal(0, c.baseline_jitter))
                w = p["w"] * st["width"] * jitter(c.glyph_width_jitter)
                h = (p["top"] - p["bottom"]) * jitter(c.glyph_height_jitter)
                x0 = pen + p.get("lsb", 0.05)
                bottom = p["bottom"] * h / max(p["top"] - p["bottom"], 1e-6) + drift if p["kind"] == "base" else drift
                box = [x0, x0 + w, bottom, bottom + h]
                pen = x0 + w + p.get("rsb", 0.05)
                base = box
                line.append(i)
            else:
                s = st["mark_scale"] * jitter(c.mark_size_jitter)
                w, h = p["w"] * s, (p["top"] - p["bottom"]) * s
                beside = p["adv"] > 0.1
                if p.get("span"):  # apun: under the whole letter before it
                    x0 = base[0] + float(rng.normal(0, c.mark_jitter))
                    x1 = base[1] + float(rng.normal(0, c.mark_jitter))
                    w = max(x1 - x0, 0.3)
                else:              # a sign beside the letter gets its spacing in step 2
                    x0 = pen + p["off"] + (0.0 if beside else float(rng.normal(0, c.mark_jitter)))
                dy = drift + float(rng.normal(0, c.mark_jitter))
                if p["bottom"] >= 0.9 or abs(p["bottom"]) < 0.05:  # above, or standing on the baseline
                    bottom = p["bottom"] + dy
                else:                                              # hanging from its top
                    bottom = p["top"] + dy - h
                box = [x0, x0 + w, bottom, bottom + h]
                if beside:         # the line goes on after it
                    pen = max(pen + p["adv"] * s, x0 + w)
                    line.append(i)
            boxes.append([ch, *box])

        # 2. gaps on the line, syllables kept together
        syl = syllables(word)
        pairs = list(zip(line, line[1:]))
        inside = [syl[a] == syl[b] for a, b in pairs]
        natural = [boxes[b][1] - boxes[a][2] for a, b in pairs]
        u = st.get("uneven", 0.0)
        gaps = [n + st["gap"] + (-u if k else u) + float(rng.normal(0, c.gap_jitter)) for n, k in zip(natural, inside)]
        edges = {}  # syllable -> the narrowest gap between it and a neighbour
        for (a, b), k, g in zip(pairs, inside, gaps):
            if not k:
                for n in (syl[a], syl[b]):
                    edges[n] = min(edges.get(n, np.inf), g)
        gaps = [max(min(g, edges.get(syl[a], np.inf)) if k else g, -c.max_overlap)
                for (a, b), k, g in zip(pairs, inside, gaps)]

        # 3. joins. A sign beside its own letter is placed against it on the ink when drawn
        #    (render), and the next syllable never joins onto such a sign.
        beside = [inside[j] and self.prior[word[b]]["kind"] == "mark" for j, (a, b) in enumerate(pairs)]
        after_sign = [not k and self.prior[word[a]]["kind"] == "mark" for (a, b), k in zip(pairs, inside)]
        joined = [rng.random() < st["touch"] * (1.0 if k else c.touch_between) and not s and not m
                  for k, s, m in zip(inside, after_sign, beside)]
        linked = {n for (a, b), k, j in zip(pairs, inside, joined) if j and not k for n in (syl[a], syl[b])}
        joined = [(j or (k and syl[a] in linked)) and not m for (a, b), k, j, m in zip(pairs, inside, joined, beside)]
        uneven = st.get("uneven", 0.0) > 0

        shift, roles, before = 0.0, [None] * len(word), [None] * len(word)
        second = {b: j for j, (a, b) in enumerate(pairs)}
        for i in range(len(word)):
            if i in second:
                j = second[i]
                shift += gaps[j] - natural[j]
                before[i] = gaps[j]
                if beside[j]:
                    roles[i] = ("sign" if not uneven else
                                "sign touching" if rng.random() < c.sign_touch else "sign close")
                else:
                    roles[i] = "after sign" if after_sign[j] else "join" if joined[j] else None
            boxes[i][1] += shift
            boxes[i][2] += shift
        return [tuple(b) for b in boxes], roles, before

    # ------------------------------------------------------------------ drawing

    def render(self, text, rng, trace=None):
        """-> Sample. trace: a list to append (character, column, row, ink map) to for every
        character as pasted, before slant and rotation (for checks such as scripts/check_signs.py)."""
        word = normalise(text)
        if not renderable(word):
            raise ValueError(f"cannot render {text!r}: characters outside the alphabet")
        c, st = self.cfg, self.word_style(rng)
        L = st["L"]
        anchor = self.store.anchor(rng) if c.style_k else None
        glyphs = [self.store.pick(ch, rng, anchor, c.style_k) for ch in word]
        boxes, roles, before = self.layout(word, rng, st)
        uneven = st.get("uneven", 0.0) > 0

        xmin = min(b[1] for b in boxes)
        xmax = max(b[2] for b in boxes)
        ymin = min(b[3] for b in boxes)
        ymax = max(b[4] for b in boxes)
        pad = 1.0 + 2.5 * c.slant + 0.5  # room for slant and rotation; cropped away below
        pushes = sum(r is not None and r != "join" for r in roles)  # each may move the rest right
        W = int(math.ceil((xmax - xmin + 2 * pad + 0.4 * pushes) * L))
        H = int(math.ceil((ymax - ymin + 2 * pad) * L))
        canvas = np.zeros((H, W), np.float32)
        placed, shift, final, sign_gap, letter_right = [], 0, [], 0, None
        for (ch, x0, x1, bottom, top), g, role, gap in zip(boxes, glyphs, roles, before):
            w = max(int(round((x1 - x0) * L)), 1)
            h = max(int(round((top - bottom) * L)), 1)
            ink = resize(self.store.alpha[g], w, h)
            px, py = int(round((x0 - xmin + pad) * L)) + shift, int(round((ymax + pad - top) * L))
            border = 0
            if st["pen"] is not None and min(w, h) >= 4:
                ink, border = set_pen(ink, st["pen"] * L * float(np.exp(rng.normal(0, 0.05))))
            reach = int(0.6 * L)
            if role == "join":  # slide left until the ink meets the ink before it; what follows moves too
                d = contact(canvas, ink, px - border, py - border, int(0.4 * L))
                px, shift = px - d, shift - d
            elif role in ("sign", "sign close", "sign touching"):  # placed by its body, from its letter
                if role == "sign touching":
                    want, low, lead = -1.0, -1.0, c.lead_in * L
                else:   # at least a pixel of paper between; a lead-in may meet the letter, not cross it
                    want = max((gap if role == "sign" else rng.uniform(*c.attach_gap)) * L, 1.0)
                    low, lead = 1.0, 0.0
                body, first = sign_body(ink), int(np.argmax((ink > 0.5).any(0)))
                for _ in range(8):
                    d = ink_distance(canvas, body, px - border, py - border, reach)
                    if d is None:
                        break
                    past = reach_past(canvas, ink, px - border, py - border, reach)
                    past = past if past is not None else -np.inf
                    over = letter_right - (px - border + first) if letter_right is not None else -np.inf
                    limit = c.overhang[uneven] * L
                    room = min(d - want, lead - past, limit - over)  # a step left of `room` keeps all limits
                    if room >= 1:
                        step = -int(room)
                    elif d < low or past > lead or over > limit:     # too close: back off
                        step = int(math.ceil(max(low - d, past - lead, over - limit)))
                    else:
                        break
                    px, shift = px + step, shift + step
                d = ink_distance(canvas, body, px - border, py - border, reach)
                sign_gap = d if d is not None else want
            elif role == "after sign":  # never nearer the sign than the sign is to its own letter
                if uneven:              # the syllables apart: sign_margin to twice that further
                    need = max(sign_gap, 0) + c.sign_margin * L
                    target = need + (rng.uniform(0, c.sign_margin) + st["uneven"]) * L
                else:                   # evenly spaced: the layout's gap, but no nearer than the sign's
                    need = max(sign_gap, 0) + 1
                    target = max(gap * L, need)
                for _ in range(8):
                    d = ink_distance(canvas, ink, px - border, py - border, reach)
                    if d is None:
                        break
                    if d > target + 1:      # closer: the distance shrinks by at most the step
                        step = -int(d - target)
                    elif d < need:          # further (a diagonal approach may need a second step)
                        step = int(math.ceil(target - d))
                    else:
                        break
                    px, shift = px + step, shift + step
            placed.append((ch, px, py, px + w, py + h))
            final.append((ch, x0 + (shift / L), x1 + (shift / L), bottom, top))
            qx, qy = px - border, py - border
            assert qx >= 0 and qy >= 0 and qx + ink.shape[1] <= W and qy + ink.shape[0] <= H, "canvas too small"
            region = canvas[qy:qy + ink.shape[0], qx:qx + ink.shape[1]]
            np.maximum(region, ink, out=region)
            if trace is not None:
                trace.append((ch, qx, qy, ink))
            if gap is not None or len(placed) == 1:  # on the line: its right edge, for a sign after it
                cols = np.flatnonzero((ink > 0.5).any(0))
                letter_right = qx + int(cols[-1]) if len(cols) else letter_right
        boxes = final

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
        return Sample(np.clip(np.round(img), 0, 255).astype(np.uint8), word, placed, glyphs, st, boxes)

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
    the seed sequence (seed, i). A word is drawn from the lexicon; with probability `numbers`
    it is a number in Meitei Mayek digits instead, and with probability `built` a word
    composed of real syllables (``lexicon.SyllableBank``: 1 to 6 syllables, every kind of
    syllable; 15% by default, the owner's choice); with probability `stop` a full stop
    (cheikhei) follows it."""

    def __init__(self, synth, lexicon, seed=0, numbers=0.03, stop=0.02, built=0.15):
        from .lexicon import SyllableBank

        self.synth, self.lexicon, self.seed = synth, lexicon, seed
        self.numbers, self.stop, self.built = numbers, stop, built
        self.bank = SyllableBank(lexicon) if built > 0 else None

    def text(self, rng):
        from .lexicon import number

        r = rng.random()
        text = (number(rng) if r < self.numbers else
                self.bank.word(rng) if r < self.numbers + self.built else self.lexicon.sample(rng))
        return text + CHEIKHEI if rng.random() < self.stop else text

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, int(i)])
        return self.synth.render(self.text(rng), rng)
