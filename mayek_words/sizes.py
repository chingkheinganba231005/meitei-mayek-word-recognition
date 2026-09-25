"""Recovering the size and proportions TUMMHCD lost, from stroke thickness.

Every TUMMHCD image is the character's ink box stretched to 24 x 24 (Phase 0). The
stretch leaves a trace: a stroke drawn with a pen of width p becomes p * 24 / w pixels
thick across a vertical stroke and p * 24 / h pixels across a horizontal one, where w and
h are the character's original width and height. So, if writers use about the same pen for
every character, the thicknesses t_x (of vertical strokes) and t_y (of horizontal strokes)
give back the original box, relative to a letter:

    h / L = T / t_y,    w / L = T / t_x,

where L is the height of a letter and T the median t_y over the 27 letters.

Thickness is measured with chords: for every ink pixel, the length of the horizontal and
the vertical run of ink through it. Across a vertical stroke the horizontal chord is short
(the stroke's thickness) and the vertical one long, and the other way round across a
horizontal stroke. Each ink pixel votes with its shorter chord for the direction it is
short in; a direction needs at least MIN_SHARE of the ink pixels, otherwise the character
has no strokes in that direction and the estimate is None. So is a thickness of more than
MAX_SHARE of the frame: that is a solid blob, not a stroke (apun, a line stretched to the
frame, becomes one).
"""

import numpy as np

from .charset import CLASS_OF, LETTERS

MIN_SHARE = 0.15
MAX_SHARE = 0.6


def _runs(mask):
    """Length of the horizontal run of ink through each pixel (0 on paper)."""
    out = np.zeros(mask.shape, np.int32)
    for r, row in enumerate(mask):
        padded = np.concatenate([[False], row, [False]])
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for start, stop in zip(edges[::2], edges[1::2]):
            out[r, start:stop] = stop - start
    return out


def chord_thickness(mask):
    """-> (t_x, t_y): median thickness in pixels of vertical and horizontal strokes (None if too few)."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if n == 0:
        return None, None
    h, v = _runs(mask), _runs(mask.T).T
    hs, vs = h[mask], v[mask]
    vertical = hs < vs      # short horizontal chord: the pixel is in a vertical stroke
    horizontal = vs < hs
    t_x = float(np.median(hs[vertical])) if vertical.sum() >= MIN_SHARE * n else None
    t_y = float(np.median(vs[horizontal])) if horizontal.sum() >= MIN_SHARE * n else None
    t_x = t_x if t_x is not None and t_x <= MAX_SHARE * mask.shape[1] else None
    t_y = t_y if t_y is not None and t_y <= MAX_SHARE * mask.shape[0] else None
    return t_x, t_y


def estimate_boxes(masks, labels):
    """Per class: mean thicknesses over its images and the box (w, h) in units of L.

    masks: (N, H, W) bool ink masks of images stretched to the frame; labels: TUMMHCD classes.
    """
    labels = np.asarray(labels)
    per_class = {}
    for c in np.unique(labels):
        tx, ty = zip(*(chord_thickness(m) for m in masks[labels == c]))
        tx = [t for t in tx if t is not None]
        ty = [t for t in ty if t is not None]
        per_class[int(c)] = {"images": int((labels == c).sum()),
                             "t_x": float(np.mean(tx)) if len(tx) >= 0.5 * len(labels[labels == c]) else None,
                             "t_y": float(np.mean(ty)) if len(ty) >= 0.5 * len(labels[labels == c]) else None}
    letter_ty = [per_class[CLASS_OF[ch]]["t_y"] for ch in LETTERS if CLASS_OF[ch] in per_class]
    letter_ty = [t for t in letter_ty if t is not None]
    T = float(np.median(letter_ty))
    for r in per_class.values():
        r["h"] = T / r["t_y"] if r["t_y"] else None
        r["w"] = T / r["t_x"] if r["t_x"] else None
    return T, per_class
