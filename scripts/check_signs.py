"""Does a sign beside its letter (ꯤ, ꯦ, ꯣ, ꯧ) sit nearer that letter than the next one, on the ink?

    python scripts/check_signs.py --glyphs work/glyphs/val.npz --lexicon work/lexicon/train.tsv \\
        --out results/sign_placement_tummhcd.json [--sizes measured|font|file.json] [--words 300]

For each sign, words from the lexicon in which the sign is followed by the letter of another
syllable are rendered (default settings, no rotation), and two things are measured on the
ink as drawn, in letter heights L:
- body: the shortest distance from the sign's body (its lead-in stroke cleared,
  ``synth.sign_body``) to its own letter, against the shortest distance from the sign to
  the next letter. This is the rule the synthesiser enforces.
- centre: the horizontal distance from the sign's centre of ink to the rightmost ink of its
  letter, against that to the leftmost ink of the next letter. A cruder check that does not
  depend on the rule's own definitions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words.charset import syllables  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402
from mayek_words.lexicon import read_counts  # noqa: E402
from mayek_words.synth import MEASURED, Config, WordSynth, load_priors, sign_body  # noqa: E402

BESIDE = [chr(c) for c in (0xABE4, 0xABE6, 0xABE3, 0xABE7)]  # ꯤ ꯦ ꯣ ꯧ


def distance(a, b):
    """Shortest distance in pixels between the ink (> 0.5) of two placed glyphs (x, y, ink):
    0 for neighbouring pixels, -1 if they overlap."""
    (ax, ay, ai), (bx, by, bi) = a, b
    x0, y0 = min(ax, bx), min(ay, by)
    H = max(ay + ai.shape[0], by + bi.shape[0]) - y0
    W = max(ax + ai.shape[1], bx + bi.shape[1]) - x0
    A, B = np.zeros((H, W), bool), np.zeros((H, W), bool)
    A[ay - y0:ay - y0 + ai.shape[0], ax - x0:ax - x0 + ai.shape[1]] = ai > 0.5
    B[by - y0:by - y0 + bi.shape[0], bx - x0:bx - x0 + bi.shape[1]] = bi > 0.5
    if not A.any() or not B.any():
        return np.nan
    return -1.0 if (A & B).any() else float(ndimage.distance_transform_edt(~A)[B].min()) - 1


def columns(g):
    x, _, ink = g
    return x + np.nonzero(ink > 0.5)[1]


def check(synth, words, seed=0):
    """-> {sign: summary} over the words (lists per sign)."""
    out = {}
    for sign, ws in words.items():
        rows = []
        for i, w in enumerate(ws):
            trace = []
            smp = synth.render(w, np.random.default_rng(seed + i), trace)
            L, placed = smp.style["L"], [t[1:] for t in trace]
            syl = syllables(smp.text)
            line = [k for k, c in enumerate(smp.text)
                    if synth.prior[c]["kind"] == "base" or synth.prior[c]["adv"] > 0.1]
            for a, k, b in zip(line, line[1:], line[2:]):
                if smp.text[k] != sign or syl[a] != syl[k] or syl[b] == syl[k]:
                    continue
                x, y, ink = placed[k]
                weight = (ink * (ink > 0.5)).sum(0)
                cx = x + float(weight @ np.arange(ink.shape[1]) / max(weight.sum(), 1e-6))
                rows.append((distance(placed[a], (x, y, sign_body(ink))) / L,
                             distance(placed[k], placed[b]) / L,
                             (cx - columns(placed[a]).max()) / L, (columns(placed[b]).min() - cx) / L))
        r = np.array(rows)
        if not len(r):
            continue
        out[sign] = {"cases": len(r),
                     "body_nearer_next_letter": round(float(np.mean(r[:, 1] <= r[:, 0])), 4),
                     "next_letter_touching_sign": round(float(np.mean(r[:, 1] < 0)), 4),
                     "centre_nearer_next_letter": round(float(np.mean(r[:, 3] <= r[:, 2])), 4),
                     "median_in_L": {"body_to_letter": round(float(np.median(r[:, 0])), 3),
                                     "sign_to_next_letter": round(float(np.median(r[:, 1])), 3),
                                     "centre_to_letter": round(float(np.median(r[:, 2])), 3),
                                     "centre_to_next_letter": round(float(np.median(r[:, 3])), 3)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glyphs", required=True, help="glyph store .npz, or 'font'")
    ap.add_argument("--lexicon", required=True, help="word<TAB>count list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="measured", help="'measured', 'font' or a glyph_sizes_*.json file")
    ap.add_argument("--words", type=int, default=300, help="words per sign (the most frequent)")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    store = GlyphStore.from_font() if args.glyphs == "font" else GlyphStore.load(args.glyphs)
    priors = load_priors(sizes={"measured": MEASURED, "font": None}.get(args.sizes, args.sizes))
    synth = WordSynth(store, priors, Config(rotation=0))
    counts = read_counts(args.lexicon)
    ranked = [w for w, _ in counts.most_common()]
    words = {}
    for s in BESIDE:
        ws = []
        for w in ranked:
            syl = syllables(w)
            if any(c == s and k + 1 < len(w) and syl[k + 1] != syl[k] for k, c in enumerate(w)):
                ws.append(w)
                if len(ws) == args.words:
                    break
        words[s] = ws
    report = {"glyphs": Path(args.glyphs).name, "lexicon": Path(args.lexicon).name,
              "sizes": args.sizes if args.sizes in ("measured", "font") else Path(args.sizes).name,
              "words_per_sign": args.words, "seed": args.seed, "signs": check(synth, words, args.seed)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for s, v in report["signs"].items():
        print(f"{s}: {v['cases']} cases; body nearer the next letter {v['body_nearer_next_letter']:.0%}, "
              f"centre nearer the next letter {v['centre_nearer_next_letter']:.0%}; medians {v['median_in_L']}")


if __name__ == "__main__":
    main()
