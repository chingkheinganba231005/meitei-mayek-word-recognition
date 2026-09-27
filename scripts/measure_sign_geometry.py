"""Where the small vowel signs stand in word images, measured on the ink.

    python scripts/measure_sign_geometry.py --sets real=work/real/real_test.tar \\
        print=work/synth/val.tar beside=work/synth/val_beside.tar --out results/round3_sign_geometry.json

A set is a folder or .tar with images/ and labels.tsv (as ``mayek_htr.data.FixedSet``
reads it): a real set cut from written pages, or a synthetic one. For each word whose text
has a sign written above its letter or beside it at the top (ꯥ ꯩ ꯪ ꯣ ꯦ ꯧ):

1. The pieces of ink (8-connected, darker than grey level 128, at least 20 pixels).
2. The letters' band: the top line and the baseline are the medians of the tops and
   bottoms of the large pieces (at least 0.6 of the tallest); L is the band's height.
3. The small pieces (under 0.75 L high and 0.9 L wide) centred in the upper 55% of the
   band or above it are matched, left to right, to those signs of the text. A word where
   the counts differ (a sign touching its letter or split in two, a letter in pieces) is
   skipped and counted.
4. For each sign, in L: `centre` and `left`, the horizontal distance from its centre and
   from its left end to the rightmost ink before it (its letter), + to the right; `top`
   and `bottom`, its highest and lowest ink relative to the top line, + above; `w` and
   `h`, its size; and `angle`, the direction of its main axis in degrees (0 horizontal,
   45 falling to the right like a backslash, 90 upright, 135 rising to the right), and
   `elongation`, the spread of its ink along that axis over the spread across it (about 1
   for a ring, 3 or more for a straight stroke).

The report gives the 10th, 50th and 90th percentiles per sign and set, and how many words
were measured and skipped. Only statistics are written: no word texts.

--glyphs store.npz adds the shapes of the character images the synthesiser draws these
signs with (TUMMHCD): each image resized to its class's measured width and height (as the
synthesiser draws it; ``synth.load_priors``), the direction of its main axis and how
elongated it is (the ratio of the axes' spreads), so that a writer's sign can be compared
with them in shape as well as in place.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.data import _read_set  # noqa: E402
from mayek_words.synth import MEASURED, load_priors, resize  # noqa: E402

SIGNS = "".join(chr(c) for c in (0xABE5, 0xABE9, 0xABEA, 0xABE3, 0xABE6, 0xABE7))  # ꯥ ꯩ ꯪ ꯣ ꯦ ꯧ
MEASURES = ("centre", "left", "top", "bottom", "w", "h", "angle", "elongation")


def pieces(gray, ink=128, min_px=20):
    """The 8-connected pieces of ink: dicts with the box (y0, y1, x0, x1) and pixel coordinates."""
    lab, _ = ndimage.label(gray < ink, structure=np.ones((3, 3)))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        if len(ys) >= min_px:
            out.append({"y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start, "x1": sl[1].stop,
                        "ys": ys + sl[0].start, "xs": xs + sl[1].start})
    return out


def axis_angle(xs, ys):
    """The direction of the main axis of a set of pixels, in degrees in [0, 180), image y
    pointing down: 45 falls to the right (a backslash), 135 rises to the right."""
    return axes(xs, ys)[0]


def axes(xs, ys):
    """-> (direction of the main axis in degrees, as axis_angle; elongation: the spread along
    the main axis over the spread across it, 1 for a ring, large for a straight stroke)."""
    if len(xs) < 3:
        return float("nan"), float("nan")
    w, v = np.linalg.eigh(np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()])))
    ax = v[:, -1]
    return float(math.degrees(math.atan2(ax[1], ax[0])) % 180.0), float(math.sqrt(w[-1] / max(w[0], 1e-6)))


def measure(gray, text, signs=SIGNS):
    """-> (list of per-sign measures in L, None) or (None, reason the word was skipped)."""
    ps = pieces(gray)
    if not ps:
        return None, "no ink"
    tallest = max(p["y1"] - p["y0"] for p in ps)
    big = [p for p in ps if p["y1"] - p["y0"] >= 0.6 * tallest]
    top = float(np.median([p["y0"] for p in big]))
    L = float(np.median([p["y1"] for p in big])) - top
    if L < 8:
        return None, "no band"
    small = [p for p in ps if all(p is not q for q in big) and p["y1"] - p["y0"] < 0.75 * L
             and p["x1"] - p["x0"] < 0.9 * L and (p["y0"] + p["y1"]) / 2 < top + 0.55 * L]
    small.sort(key=lambda p: (p["x0"] + p["x1"]) / 2)
    want = [c for c in text if c in signs]
    if len(small) != len(want):
        return None, "count"
    out = []
    for ch, p in zip(want, small):
        cx = (p["x0"] + p["x1"]) / 2
        before = [q for q in ps if all(q is not s for s in small) and (q["x0"] + q["x1"]) / 2 < cx]
        if not before:
            return None, "nothing before a sign"
        edge = max(q["x1"] for q in before)
        angle, elongation = axes(p["xs"].astype(float), p["ys"].astype(float))
        out.append({"char": ch, "centre": (cx - edge) / L, "left": (p["x0"] - edge) / L,
                    "top": (top - p["y0"]) / L, "bottom": (top - p["y1"]) / L,
                    "w": (p["x1"] - p["x0"]) / L, "h": (p["y1"] - p["y0"]) / L,
                    "angle": angle, "elongation": elongation})
    return out, None


def measure_set(path, limit=None):
    names, texts, grays = _read_set(path)
    if limit:
        names, texts, grays = names[:limit], texts[:limit], grays[:limit]
    rows, words, skipped = [], 0, {}
    for text, gray in zip(texts, grays):
        if not any(c in SIGNS for c in text):
            continue
        words += 1
        m, why = measure(gray, text)
        if m is None:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            rows.extend(m)
    report = {"words_with_signs": words, "measured": words - sum(skipped.values()), "skipped": skipped, "signs": {}}
    for ch in SIGNS:
        r = [x for x in rows if x["char"] == ch]
        if r:
            report["signs"][ch] = {"n": len(r), **{k: [round(float(v), 3) for v in np.nanpercentile(
                [x[k] for x in r], [10, 50, 90])] for k in MEASURES}}
    return report


def image_shapes(store, sizes=MEASURED):
    """-> {sign: {"n", "angle", "elongation"}} over the store's images of each sign, drawn at
    the width and height the synthesiser gives them."""
    prior = load_priors(sizes=sizes)
    out = {}
    for ch in SIGNS:
        p = prior[ch]
        w, h = 48, max(int(round(48 * (p["top"] - p["bottom"]) / p["w"])), 1)
        angles, elong = [], []
        for i in store.candidates(ch):
            ys, xs = np.nonzero(resize(store.alpha[i], w, h) > 0.5)
            if len(xs) < 3:
                continue
            a, e = axes(xs.astype(float), ys.astype(float))
            angles.append(a)
            elong.append(e)
        if angles:
            out[ch] = {"n": len(angles), "angle": [round(float(v), 1) for v in np.percentile(angles, [10, 50, 90])],
                       "elongation": [round(float(v), 2) for v in np.percentile(elong, [10, 50, 90])],
                       "falling_right_20_to_70": round(float(np.mean((np.array(angles) > 20) & (np.array(angles) < 70))), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="name=path (folder or .tar)")
    ap.add_argument("--limit", type=int, help="only the first n words of each set")
    ap.add_argument("--glyphs", help="glyph store .npz (or 'font'): the shapes of the images of these signs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"percentiles": [10, 50, 90], "units": "L (the letters' band height); angle in degrees",
              "sets": {}}
    for item in args.sets:
        name, path = item.split("=", 1)
        report["sets"][name] = {"path": Path(path).name, **measure_set(path, args.limit)}
        s = report["sets"][name]
        print(f"{name}: {s['measured']} of {s['words_with_signs']} words with signs measured (skipped {s['skipped']})")
        for ch, v in s["signs"].items():
            print(f"  {ch} {v['n']:4d}: centre {v['centre']}, top {v['top']}, bottom {v['bottom']}, "
                  f"h {v['h']}, angle {v['angle']}")
    if args.glyphs:
        from mayek_words.glyphs import GlyphStore

        store = GlyphStore.from_font() if args.glyphs == "font" else GlyphStore.load(args.glyphs)
        report["images"] = {"glyphs": Path(args.glyphs).name, "signs": image_shapes(store)}
        for ch, v in report["images"]["signs"].items():
            print(f"  images of {ch} ({v['n']}): angle {v['angle']}, elongation {v['elongation']}, "
                  f"falling to the right {v['falling_right_20_to_70']:.0%}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
