"""Does the pen step keep every stroke of a character, faint ones included?

    python scripts/check_strokes.py --glyphs work/glyphs/val.npz --out results/pen_strokes_tummhcd_val.json \
        [--lexicon work/lexicon/train.tsv]

The pen step (``synth.set_pen``) redraws each character at the word's pen width from a mask
of its strokes. A scan's faint strokes (grey, under the ink map's 0.5) were lost when the
mask was the pixels over 0.5; ``synth.strokes`` adds faint strokes joined to the dark ones.
For a sample of characters resized to 32 x 32 and redrawn with a 3.2 px pen, this counts
the stroke pixels (over 0.2 and joined to pixels over 0.5) left with no ink nearby, allowing
for the thinning the pen step intends, under the old mask ("cut at 0.5") and the new one.

With --lexicon, it also renders words and measures how dark their strokes are against the
paper (paper minus the 2nd percentile of the image, in grey levels), with the minimum stroke
contrast (``Config.min_contrast``) and without it: pale scans on grey paper, blurred, looked
as if characters were disappearing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words import synth  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402


def visible(alpha):
    """Pixels over 0.2 joined to pixels over 0.5: what the eye sees as strokes."""
    lab, n = ndimage.label(alpha > 0.2, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[alpha > 0.5])] = True
    keep[0] = False
    return keep[lab]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glyphs", required=True, help="glyph store .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lexicon", help="word<TAB>count list: also measure the contrast of rendered words")
    ap.add_argument("--words", type=int, default=1000)
    args = ap.parse_args()

    store = GlyphStore.load(args.glyphs)
    idx = np.random.default_rng(args.seed).choice(len(store), min(args.n, len(store)), replace=False)
    masks = {"cut at 0.5": lambda a: a > 0.5, "strokes": synth.strokes}
    missing = {k: [] for k in masks}
    original = synth.strokes
    for i in idx:
        a = synth.resize(store.alpha[i], 32, 32)
        ref = visible(a)
        have = synth.stroke_width(a > 0.5) if (a > 0.5).sum() >= 3 else 1.0
        allow = int(math.ceil(max(0.0, (have - 3.2) / 2))) + 2
        for name, mask in masks.items():
            synth.strokes = mask
            try:
                out, border = synth.set_pen(a, 3.2)
            finally:
                synth.strokes = original
            out = out[border:border + 32, border:border + 32] if border else out
            cover = ndimage.binary_dilation(out > 0.5, iterations=allow)
            missing[name].append(1 - (ref & cover).sum() / max(ref.sum(), 1))
    report = {"glyphs": Path(args.glyphs).name, "characters": len(idx), "seed": args.seed,
              "share_of_characters_missing_strokes": {
                  name: {f"over_{t}": round(float(np.mean(np.array(v) > t)), 4) for t in (0.05, 0.1, 0.2)}
                  for name, v in missing.items()}}
    if args.lexicon:
        from mayek_words.lexicon import Lexicon

        lexicon = Lexicon.load(args.lexicon)
        report["word_contrast"] = {}
        for name, floor in (("with minimum contrast", synth.Config().min_contrast), ("without", None)):
            words = synth.Words(synth.WordSynth(store, config=synth.Config(min_contrast=floor)), lexicon, seed=args.seed)
            c = np.array([s.style["paper"] - np.percentile(s.image, 2) for s in (words[i] for i in range(args.words))])
            report["word_contrast"][name] = {"min_contrast": floor, "words": args.words,
                                             "median": round(float(np.median(c)), 1),
                                             "p5": round(float(np.percentile(c, 5)), 1),
                                             **{f"under_{t}": round(float(np.mean(c < t)), 4) for t in (80, 100, 120)}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ("share_of_characters_missing_strokes", "word_contrast")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
