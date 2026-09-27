"""Make the pages on which the real word set is written by hand (Phase 3): a PDF of A4
pages, each word printed above an empty box, to be written on an iPad (or printed and
photographed) and cut up by scripts/cut_writing_pages.py.

    # the real test set: 500 words and numbers from the test split of the lexicon
    python scripts/make_writing_pages.py --lexicon work/lexicon/test.tsv --n 500 \\
        --title "Meitei Mayek words" --out work/real/real_words_pages.pdf --summary results/phase3_pages.json

    # a trial page from validation words, to check the writing and the cutting first
    python scripts/make_writing_pages.py --lexicon work/lexicon/val.tsv --split val --n 20 \\
        --title "Trial page" --out work/real/trial_page.pdf

Words: distinct words of the given split (a hash of the word, as in mayek_words.lexicon,
so test words are never training words), drawn with probability proportional to
count ** 0.5; every letter and sign as often as the lexicon allows up to --cover; 5%
numbers; 2% of the words ending with a full stop (mayek_htr.pages.choose_items). The PDF
carries every item's text and box (its embedded manifest.json). --summary writes the
counts only (no words): the lexicon's size, the pages, the kinds, each character's count.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.pages import choose_items, make_manifest, write_pdf  # noqa: E402
from mayek_words.charset import ALPHABET  # noqa: E402
from mayek_words.lexicon import read_counts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", required=True, help="word<TAB>count (the lexicon of the split)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=500, help="items (words and numbers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--numbers", type=float, default=0.05)
    ap.add_argument("--stop", type=float, default=0.02)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=14)
    ap.add_argument("--cover", type=int, default=8)
    ap.add_argument("--title", default="Meitei Mayek words")
    ap.add_argument("--out", required=True, help="the template .pdf")
    ap.add_argument("--summary", help="counts .json")
    args = ap.parse_args()

    counts = read_counts(args.lexicon)
    items = choose_items(counts, args.n, np.random.default_rng(args.seed), split=args.split, numbers=args.numbers,
                         stop=args.stop, min_count=args.min_count, max_len=args.max_len, cover=args.cover)
    sha1 = hashlib.sha1(Path(args.lexicon).read_bytes()).hexdigest()
    selection = {"split": args.split, "n": args.n, "seed": args.seed, "numbers": args.numbers, "stop": args.stop,
                 "min_count": args.min_count, "max_len": args.max_len, "cover": args.cover}
    manifest = make_manifest(items, args.title, lexicon=Path(args.lexicon).name, lexicon_sha1=sha1,
                             selection=selection)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_pdf(args.out, manifest)

    chars = Counter(ch for it in items for ch in it["text"])
    summary = {"template": Path(args.out).name, "title": args.title, "lexicon": manifest["lexicon"],
               "lexicon_sha1": sha1, "lexicon_words": len(counts), "selection": selection,
               "pages": manifest["pages"], "items": len(items), "kinds": dict(Counter(it["kind"] for it in items)),
               "with_full_stop": sum(it["text"].endswith("꯫") for it in items),
               "characters": sum(chars.values()),
               "length": {"mean": round(float(np.mean([len(it["text"]) for it in items])), 2),
                          "max": max(len(it["text"]) for it in items)},
               "per_page": [sum(it["page"] == p for it in manifest["items"]) for p in range(1, manifest["pages"] + 1)],
               "character_counts": {ch: chars[ch] for ch in ALPHABET},
               "fewer_than_cover": [ch for ch in ALPHABET if chars[ch] < args.cover]}
    print(f"{args.out}: {len(items)} items on {manifest['pages']} pages; {summary['kinds']}; "
          f"characters under {args.cover}: {' '.join(summary['fewer_than_cover']) or 'none'}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
