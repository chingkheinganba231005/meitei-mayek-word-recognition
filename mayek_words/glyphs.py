"""Character images to compose words from: TUMMHCD by split, or the font stand-ins.

A GlyphStore holds, for every image, its class, an ink map (0 paper, 1 ink) and a style
vector. Style features are measured on the 24 x 24 image and standardised within each
class, so that they can be compared across classes: slant, stroke width in the frame,
ink fraction and ink darkness. TUMMHCD has no writer information (Phase 0), so a word's
characters are matched by style instead of taken from one writer: ``pick`` chooses among
the k images of a class closest to the word's style anchor.

Which images may be used:
- synthetic training words: TUMMHCD train minus our validation part (``mayek.split``,
  15% per class, seed 42), the images the pretrained networks were trained on;
- synthetic validation words: that validation part;
- synthetic test words: TUMMHCD test, without the 469 images that have a pixel-identical
  train image;
- nowhere: the 48 images in groups of identical pixels with conflicting labels.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .charset import IMAGE_CLASSES

SIDE = 24
STYLE = ["slant", "stroke_width", "ink_fraction", "ink_level"]
ASSETS = Path(__file__).resolve().parent / "assets"


def otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.argmax(np.nan_to_num(between)))


def ink_map(gray):
    """uint8 greyscale, dark ink on light paper -> (ink map in [0, 1], style features as in STYLE).

    The ink map is 0 on paper, 0.5 at the image's Otsu threshold and 1 for a typical ink
    pixel, so that pale writing is kept whole (a cut at half the darkest ink erased much of
    it: 1.3% of characters lost more than half their ink).
    The polarity is fixed, not guessed: every TUMMHCD image is dark on light (Phase 0:
    paper level 247 or more in 95% of them), and a small sign stretched to the frame can
    be mostly ink. An image with (almost) no paper, such as apun, which is a line
    stretched to 24 x 24, counts as white paper under solid ink.
    """
    g = gray.astype(np.float32)
    t = float(otsu(gray))
    ink = gray <= t
    if (~ink).mean() < 0.05 or float(g.max() - g.min()) < 20:  # no paper to measure
        ink, t = g <= 127, 127.0
        paper = 255.0
    else:
        paper = float(np.median(g[~ink]))
    if not ink.any():
        return np.zeros_like(g), [0.0, 0.0, 0.0, 255.0]
    # 0 on paper, 0.5 half-way between the last ink level (the Otsu threshold) and the first
    # paper level, 1 from a typical ink pixel on: whatever the scan shows as ink stays ink
    # (alpha > 0.5, the pen step's cut), however pale the writing
    m, full = t + 0.5, float(np.median(g[ink]))
    alpha = np.where(g <= t, 0.5 + 0.5 * np.clip((m - g) / max(m - full, 0.5), 0, 1),
                     0.5 * np.clip((paper - g) / max(paper - m, 0.5), 0, 1)).astype(np.float32)
    ys, xs = np.nonzero(ink)
    dy, dx = ys - ys.mean(), xs - xs.mean()
    slant = float(-(dx * dy).sum() / max((dy * dy).sum(), 1e-6))  # > 0: top leans right
    pad = np.pad(ink, 1)
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    stroke = 2.0 * ink.sum() / max(int(ink.sum() - interior.sum()), 1)
    return alpha, [slant, stroke, float(ink.mean()), float(np.median(g[ink]))]


def path_key(path):
    """The part of an image path that identifies it: split folder / class folder / file."""
    return "/".join(Path(str(path).replace("\\", "/")).parts[-3:])


def excluded_paths(duplicates_csv, split):
    """Paths (as path_key) not to use for a split, from the Phase 0 duplicates file."""
    import pandas as pd

    d = pd.read_csv(duplicates_csv)
    conflicting = d.groupby("group").label.nunique()
    bad = set(d.loc[d.group.isin(conflicting[conflicting > 1].index), "path"])
    if split == "test":
        with_train = set(d.loc[d.split == "train", "group"])
        bad |= set(d.loc[(d.split == "test") & d.group.isin(with_train), "path"])
    return {path_key(p) for p in bad}


class GlyphStore:
    def __init__(self, images, labels, keys=None):
        self.gray = np.asarray(images, np.uint8)
        self.labels = np.asarray(labels, np.int64)
        self.keys = list(keys) if keys is not None else [str(i) for i in range(len(self.labels))]
        maps, feats = zip(*(ink_map(im) for im in self.gray))
        self.alpha = np.stack(maps).astype(np.float32)
        raw = np.array(feats, np.float64)
        self.raw_style = raw
        self.style = np.zeros_like(raw)
        self.by_class = {}
        for c in np.unique(self.labels):
            idx = np.flatnonzero(self.labels == c)
            self.by_class[int(c)] = idx
            sd = raw[idx].std(0)
            self.style[idx] = (raw[idx] - raw[idx].mean(0)) / np.where(sd > 1e-9, sd, 1.0)

    def __len__(self):
        return len(self.labels)

    # ------------------------------------------------------------ building and saving

    @classmethod
    def from_font(cls):
        """The 55 font characters (mayek_words/assets/font_glyphs.npz): for tests and previews."""
        f = np.load(ASSETS / "font_glyphs.npz")
        return cls(f["images"], f["labels"])

    @classmethod
    def from_split(cls, split_dir, split, duplicates_csv=None):
        """Images of one split of ``mayek.split`` (split_dir/{train,val,test}.csv)."""
        import pandas as pd

        split_dir = Path(split_dir)
        rows = pd.read_csv(split_dir / f"{split}.csv")
        skip = excluded_paths(duplicates_csv, split) if duplicates_csv else set()
        keep = [path_key(p) not in skip for p in rows.path]
        rows = rows[keep]
        images = []
        for p in rows.path:
            im = Image.open(split_dir / p).convert("L")
            if im.size != (SIDE, SIDE):
                im = im.resize((SIDE, SIDE), Image.BILINEAR)
            images.append(np.asarray(im))
        store = cls(np.stack(images), rows.label.to_numpy(), [path_key(p) for p in rows.path])
        store.dropped = int(len(keep) - sum(keep))
        return store

    def save(self, path):
        np.savez_compressed(path, images=self.gray, labels=self.labels, keys=np.array(self.keys))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        return cls(f["images"], f["labels"], f["keys"].tolist())

    # ------------------------------------------------------------ choosing characters

    def candidates(self, char):
        """Indices of the images that can draw a character of the alphabet."""
        parts = [self.by_class.get(c) for c in IMAGE_CLASSES[char]]
        parts = [p for p in parts if p is not None]
        if not parts:
            raise KeyError(f"no images for {char!r} (U+{ord(char):04X})")
        return np.concatenate(parts)

    def anchor(self, rng):
        """A style to match a word's characters to: that of a random image."""
        return self.style[rng.integers(len(self))]

    def pick(self, char, rng, anchor=None, k=16):
        """An image for the character: one of the k closest in style to the anchor (any, if None)."""
        idx = self.candidates(char)
        if anchor is None or k is None or len(idx) <= k:
            return int(idx[rng.integers(len(idx))])
        d = ((self.style[idx] - anchor) ** 2).sum(1)
        near = np.argpartition(d, k - 1)[:k]
        return int(idx[near[rng.integers(k)]])
