"""How large people write each character: TUMMHCD's lost sizes, recovered from stroke thickness.

    python scripts/glyph_sizes.py work/glyphs/train.npz --out results/glyph_sizes_tummhcd.json

Method in mayek_words/sizes.py: every image is the ink box stretched to 24 x 24, so the
thickness of vertical and horizontal strokes gives the character's original width and
height relative to a letter, if the pen is the same. The script first checks the method
on the font's own characters, stretched the same way, where the true sizes are known,
then applies it to the store and compares with the font.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words.charset import TUMMHCD  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402
from mayek_words.sizes import estimate_boxes  # noqa: E402
from mayek_words.synth import load_priors  # noqa: E402


def compare(est, prior):
    rows, ratios_w, ratios_h = [], [], []
    for c, ch in enumerate(TUMMHCD):
        e, p = est.get(c), prior[ch]
        if e is None:
            continue
        fw, fh = p["w"], p["top"] - p["bottom"]
        row = {"class": c, "char": ch, "kind": p["kind"], "images": e["images"],
               "t_x": e["t_x"], "t_y": e["t_y"],
               "w": e["w"], "h": e["h"], "font_w": round(fw, 3), "font_h": round(fh, 3),
               "w_over_font": round(e["w"] / fw, 3) if e["w"] else None,
               "h_over_font": round(e["h"] / fh, 3) if e["h"] else None}
        rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        if e["w"]:
            ratios_w.append(e["w"] / fw)
        if e["h"]:
            ratios_h.append(e["h"] / fh)
    summary = {name: {"classes": len(r), "median": round(float(np.median(r)), 3),
                      "p10": round(float(np.percentile(r, 10)), 3), "p90": round(float(np.percentile(r, 90)), 3)}
               for name, r in (("w_over_font", ratios_w), ("h_over_font", ratios_h))}
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="a glyph store .npz (scripts/build_glyphs.py); normally train")
    ap.add_argument("--out", default="results/glyph_sizes_tummhcd.json")
    args = ap.parse_args()

    prior = load_priors()
    font = GlyphStore.from_font()
    _, font_est = estimate_boxes(font.alpha > 0.5, font.labels)
    _, check = compare(font_est, prior)

    store = GlyphStore.load(args.store)
    T, est = estimate_boxes(store.alpha > 0.5, store.labels)
    rows, summary = compare(est, prior)
    out = {"store": Path(args.store).name, "images": len(store),
           "letter_stroke_across_horizontal_px": round(T, 3),
           "pen_width_over_letter_height": round(T / 24, 4),
           "check_on_font_characters": check, "tummhcd_vs_font": summary, "classes": rows,
           "note": "w and h in units of L (letter height); None where a character has too few "
                   "strokes in that direction. check_on_font_characters: the method on the font's "
                   "own characters stretched to 24 x 24, where the answer is the font's size."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"check on the font: {check}")
    print(f"TUMMHCD against the font: {summary}")
    print(f"{'':6}{'w':>6}{'font':>6}{'h':>7}{'font':>6}")
    for r in rows:
        if r["kind"] == "mark" or r["class"] in (9, 25, 44):
            print(f"{r['class']:3d} {r['char']} {r['w'] or '-':>6}{r['font_w']:>6} {r['h'] or '-':>6}{r['font_h']:>6}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
