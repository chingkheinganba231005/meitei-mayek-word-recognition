"""The character language model for decoding (mayek_htr/lm.py): interpolated Kneser-Ney
n-grams over the training lexicon's words, order and weighting chosen by perplexity on the
validation lexicon.

    python scripts/build_char_lm.py --train work/lexicon/train.tsv --val work/lexicon/val.tsv \\
        --out work/lm/char_lm.pkl --results results/phase2_lm.json

Training text: every training word weighted by count ** power, plus numbers and words
followed by a full stop in the synthesiser's shares (lm.training_counts). Validation: the
validation words weighted as the synthetic validation words are drawn (count ** 0.5), with
numbers and full stops in the same shares. The words composed of syllables (15% of the
synthetic words) are left out of both: the model is meant for real words. The model with
the lowest perplexity per character is saved.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.lm import CharLM, training_counts  # noqa: E402
from mayek_words.lexicon import read_counts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="training lexicon (word<TAB>count)")
    ap.add_argument("--val", required=True, help="validation lexicon")
    ap.add_argument("--out", required=True, help="the chosen model (.pkl)")
    ap.add_argument("--results", default="results/phase2_lm.json")
    ap.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="training words weighted by count ** power (0: every distinct word once)")
    ap.add_argument("--discount", type=float, default=0.75)
    args = ap.parse_args()

    train, val = read_counts(args.train), read_counts(args.val)
    overlap = len(set(train) & set(val))
    held = training_counts(val, power=0.5, seed=1)
    words, weights = list(held), list(held.values())
    grid, best = [], None
    for power in args.powers:
        counts = training_counts(train, power=power, seed=0)
        for order in args.orders:
            t0 = time.time()
            lm = CharLM(order, args.discount).fit(counts)
            ppl = lm.perplexity(words, weights)
            row = {"order": order, "power": power, "val_perplexity": round(ppl, 4),
                   "ngrams": sum(len(t) for t in lm.counts.values()), "seconds": round(time.time() - t0, 1)}
            grid.append(row)
            print(json.dumps(row))
            if best is None or ppl < best[0]["val_perplexity"]:
                best = (row, lm)
    row, lm = best
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lm.save(args.out)
    out = {"chosen": row, "discount": args.discount,
           "train": {"file": Path(args.train).name, "distinct_words": len(train)},
           "val": {"file": Path(args.val).name, "distinct_words": len(val), "in_train": overlap,
                   "weighting": "count ** 0.5, numbers 3% and full stops 2% of the weight, as the synthetic sets"},
           "note": "perplexity per character, the end of the word counted as a character",
           "grid": grid}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"chosen: order {row['order']}, power {row['power']}, validation perplexity {row['val_perplexity']}; "
          f"saved to {args.out}")


if __name__ == "__main__":
    main()
