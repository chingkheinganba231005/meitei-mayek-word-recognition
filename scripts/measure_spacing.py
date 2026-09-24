"""How close together handwritten letters are: gaps and touching, in letter heights, from images.

    python scripts/measure_spacing.py page.png other.png --out results/spacing.json \\
        [--invert] [--drop-coloured] [--drop-ruled] [--source "where the images come from"]
    python scripts/measure_spacing.py --synthetic 10 --glyphs font --lexicon words.tsv \\
        --out results/spacing_synthetic_font.json

For photos or scans of handwriting (dark ink on light paper; --invert for light on dark):
ink is found against the local paper level, and connected pieces of ink are measured. The
letter height L is the median height of the taller half of the pieces; pieces at least
0.6 L tall are letters (possibly several joined), smaller ones are signs and dots. Pieces
are grouped into lines, and the gap between neighbouring letters on a line is measured;
a gap over 0.6 L is a space between words.

Touching letters form one piece, so they have no gap to measure. Their number is
estimated from the width of each piece: a piece k times as wide as a single letter (the
median width of pieces narrower than 1.5 L) holds about k letters, joined k - 1 times.
"Touching" is the share of neighbouring letters joined this way. At low resolution, blur
joins letters closer than about a pixel, so compare with synthetic words using gaps under
0.03 L, not 0.

--drop-coloured removes coloured ink (lines drawn over a figure), --drop-ruled removes
long thin pieces (ruled lines). Rough by design: meant to compare the synthesiser's
spacing with real handwriting, not to segment it.

--synthetic N renders N pages of synthetic words (default settings, no rotation, so that
lines stay level) and measures them in exactly the same way, for a like-for-like check of
the synthesiser against real handwriting.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

WORD_GAP = 0.6


def load(path, invert=False, drop_coloured=False):
    rgb = Image.open(path).convert("RGB")
    gray = rgb.convert("L")
    if invert:
        gray = ImageOps.invert(gray)
    g = np.asarray(gray).astype(np.float32)
    if drop_coloured:
        a = np.asarray(rgb).astype(np.float32)
        g[(a.max(2) - a.min(2)) > 40] = 255
    return g


def pieces(g, min_area=12):
    """Boxes (x0, x1, y0, y1) of the connected pieces of ink."""
    paper = ndimage.uniform_filter(ndimage.maximum_filter(g, size=15), size=31)  # lightest value nearby
    diff = paper - g
    ink = diff > max(25.0, 0.35 * np.percentile(diff, 99.5))
    lab, _ = ndimage.label(ink, structure=np.ones((3, 3)))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if int((lab[sl] == i).sum()) >= min_area:
            boxes.append((sl[1].start, sl[1].stop, sl[0].start, sl[0].stop))
    return boxes


def letter_height(boxes):
    h = np.sort([b[3] - b[2] for b in boxes])
    return float(np.median(h[len(h) // 2:]))


def measure(boxes, drop_ruled=False):
    L = letter_height(boxes)
    if drop_ruled:
        boxes = [b for b in boxes if not ((b[1] - b[0]) > 4 * L and (b[3] - b[2]) < 0.5 * L)]
        L = letter_height(boxes)
    letters = [b for b in boxes if b[3] - b[2] >= 0.6 * L]
    lines = []
    for b in sorted(letters, key=lambda b: (b[2] + b[3]) / 2):
        mid = (b[2] + b[3]) / 2
        for line in lines:
            if abs(mid - line["mid"]) < 0.5 * L:
                line["boxes"].append(b)
                break
        else:
            lines.append({"mid": mid, "boxes": [b]})
    gaps = []
    for line in lines:
        row = sorted(line["boxes"], key=lambda b: b[0])
        gaps += [(b[0] - a[1]) / L for a, b in zip(row, row[1:])]
    gaps = np.array(gaps)
    visible = gaps[gaps <= WORD_GAP]
    widths = np.array([(b[1] - b[0]) / L for b in letters])
    single = float(np.median(widths[widths < 1.5]))
    joined = int((np.maximum(np.round(widths / single), 1) - 1).sum())
    return {"letter_height_px": round(L, 1), "letters_found": len(letters), "lines": len(lines),
            "single_letter_width_in_L": round(single, 3),
            "visible_gaps_in_L": {"n": int(len(visible)), "p10": round(float(np.percentile(visible, 10)), 3),
                                  "median": round(float(np.median(visible)), 3),
                                  "p90": round(float(np.percentile(visible, 90)), 3)} if len(visible) else None,
            "joined_letters": joined,
            "touching": round(joined / max(joined + len(visible), 1), 3)}


def synthetic_page(words, rng, rows=8, width=1300, row_h=95, space=38):
    """A page of synthetic words (words: synth.Words), dark ink on white."""
    page = np.full((rows * row_h, width), 255, np.uint8)
    for r in range(rows):
        x = 10
        while True:
            smp = words.synth.render(words.text(rng), rng)
            h, w = smp.image.shape
            if x + w > width - 10:
                break
            if h <= row_h:
                sub = page[r * row_h:r * row_h + h, x:x + w]
                np.minimum(sub, smp.image, out=sub)
                x += w + space
    return page


def measure_synthetic(n, glyphs, lexicon, seed=0):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mayek_words.glyphs import GlyphStore
    from mayek_words.lexicon import Lexicon
    from mayek_words.synth import Config, Words, WordSynth

    store = GlyphStore.from_font() if glyphs == "font" else GlyphStore.load(glyphs)
    cfg = Config(rotation=0)
    words = Words(WordSynth(store, config=cfg), Lexicon.load(lexicon), numbers=0, stop=0)
    rng = np.random.default_rng(seed)
    pages = [measure(pieces(synthetic_page(words, rng).astype(np.float32))) for _ in range(n)]
    return {"pages": n, "glyphs": Path(glyphs).name, "lexicon": Path(lexicon).name,
            "gap": cfg.gap, "gap_jitter": cfg.gap_jitter, "touch": cfg.touch,
            "touching_median": round(float(np.median([p["touching"] for p in pages])), 3),
            "visible_gap_median_in_L": round(float(np.median([p["visible_gaps_in_L"]["median"] for p in pages])), 3),
            "letter_height_px_median": float(np.median([p["letter_height_px"] for p in pages]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--drop-coloured", action="store_true")
    ap.add_argument("--drop-ruled", action="store_true")
    ap.add_argument("--source", default="", help="where the images come from, recorded in the output")
    ap.add_argument("--synthetic", type=int, default=0, help="measure this many pages of synthetic words")
    ap.add_argument("--glyphs", default="font", help="with --synthetic: glyph store .npz, or 'font'")
    ap.add_argument("--lexicon", help="with --synthetic: word<TAB>count list")
    args = ap.parse_args()

    if args.synthetic:
        r = measure_synthetic(args.synthetic, args.glyphs, args.lexicon)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"synthetic": r}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"synthetic pages: touching {r['touching_median']:.0%}, visible gap median "
              f"{r['visible_gap_median_in_L']} L; written to {out}")
        return

    rows = {}
    for path in args.images:
        r = measure(pieces(load(path, args.invert, args.drop_coloured)), args.drop_ruled)
        rows[Path(path).name] = r
        v = r["visible_gaps_in_L"] or {}
        print(f"{Path(path).name}: L {r['letter_height_px']} px, visible gaps median {v.get('median')} L "
              f"(n={v.get('n')}), touching {r['touching']:.0%}")
    out = Path(args.out)
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"samples": {}}
    report["samples"].update({name: {**r, "source": args.source,
                                     "options": {"invert": args.invert, "drop_coloured": args.drop_coloured,
                                                 "drop_ruled": args.drop_ruled}} for name, r in rows.items()})
    samples = list(report["samples"].values())
    report["median_over_samples"] = {
        "touching": round(float(np.median([s["touching"] for s in samples])), 3),
        "visible_gap_median_in_L": round(float(np.median([s["visible_gaps_in_L"]["median"] for s in samples
                                                          if s["visible_gaps_in_L"]])), 3),
        "samples": len(samples)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"median over {len(samples)} samples: {report['median_over_samples']}; written to {out}")


if __name__ == "__main__":
    main()
