"""How thick the strokes of word sets are, as the recogniser sees them (words normalised
to 64 px high; mayek_htr.images.strokes): to compare a pen with the training words.

    python scripts/measure_strokes.py --set synthetic=work/synth/val.tar --set trial=work/real/trial \\
        --out results/phase3_trial_strokes.json

For each set (a folder or .tar with images/ and labels.tsv): the stroke width in px
(median, 10th and 90th percentiles), the ink band's height in px, width / band (the pen
relative to the writing's size) and the share of ink pixels; also the stroke width at
the images' own scale (for words cut from 200 dpi pages, 1 px = 0.127 mm).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.data import _read_set  # noqa: E402
from mayek_htr.images import strokes  # noqa: E402


def describe(grays, limit=None):
    grays = [g for g in grays[:limit] if strokes(g) is not None]
    rows = np.array([strokes(g) for g in grays])
    w, band, ink = rows[:, 0], rows[:, 1], rows[:, 2]
    native = np.array([strokes(g, None)[0] for g in grays])
    return {"words": len(rows), "stroke_px": {"median": round(float(np.median(w)), 2),
                                              "p10": round(float(np.percentile(w, 10)), 2),
                                              "p90": round(float(np.percentile(w, 90)), 2)},
            "band_px": round(float(np.median(band)), 1), "stroke_per_band": round(float(np.median(w / band)), 3),
            "ink_share": round(float(np.median(ink)), 3),
            "stroke_px_own_scale": round(float(np.median(native)), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--limit", type=int, help="at most this many words per set")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for spec in args.set:
        name, _, path = spec.partition("=")
        _, _, grays = _read_set(path)
        out[name] = {"set": Path(path).name, **describe(grays, args.limit)}
        s = out[name]
        print(f"{name}: {s['words']} words, strokes {s['stroke_px']['median']} px, "
              f"{s['stroke_per_band']} of the ink band, ink {s['ink_share']:.1%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
