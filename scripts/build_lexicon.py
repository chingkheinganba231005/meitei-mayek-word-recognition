"""The words to write, in everyday spelling, split by word into train / validation / test.

    python scripts/build_lexicon.py wikipedia=work/wordlists/wikipedia.tsv \\
        fineweb2=work/wordlists/fineweb2.tsv fineweb2_removed=work/wordlists/fineweb2_removed.tsv \\
        --out-dir work/lexicon --stats results/lexicon_stats.json [--exclude prompts.txt]

Inputs are the word-frequency lists of the Phase 0 run (scripts/corpus_stats.py
--words-out). Rules in mayek_words/lexicon.py. Writes train.tsv, val.tsv and test.tsv
(word<TAB>count) to --out-dir, and the statistics to --stats. The word lists come from
CC BY-SA and ODC-By text: keep them out of the repository; the statistics are fine.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words import lexicon as lx  # noqa: E402
from mayek_words.charset import normalise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="name=path of a word-frequency list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats", default="results/lexicon_stats.json")
    ap.add_argument("--max-len", type=int, default=lx.MAX_LEN)
    ap.add_argument("--exclude", help="file with words (one per line) to keep out of training")
    args = ap.parse_args()

    sources, stats = {}, {"sources": {}}
    for item in args.sources:
        name, _, path = item.partition("=")
        counts = lx.read_counts(path)
        sources[name] = counts
        stats["sources"][name] = {"file": Path(path).name, "distinct_words": len(counts),
                                  "running_words": sum(counts.values())}
    combined = lx.combine(sources)
    kept, dropped = lx.clean(combined, args.max_len)
    exclude = []
    if args.exclude:
        exclude = [normalise(w.strip()) for w in Path(args.exclude).read_text(encoding="utf-8").split() if w.strip()]
    parts = lx.split(kept, exclude)
    plain, boosted = lx.Lexicon(parts["train"], rare_share=0.0), lx.Lexicon(parts["train"])
    shares = (lx.sampled_shares(plain), lx.sampled_shares(boosted))
    stats["sampling_train"] = {
        "note": "share of each character among the characters of 200,000 drawn training words, without and "
                "with the draws for rare characters (rare_share 0.1, rare_below 0.005)",
        "rare_characters": boosted.rare_chars,
        "shares": {ch: {"plain": round(shares[0][ch], 5), "with_rare_draws": round(shares[1][ch], 5)}
                   for ch in sorted(shares[0], key=lambda c: shares[0][c])}}
    stats.update({
        "combined": {"distinct_words": len(combined), "running_words": sum(combined.values()),
                     "note": "largest count over the sources"},
        "kept": {"distinct_words": len(kept), "running_words": sum(kept.values())},
        "dropped": {why: {"distinct_words": d, "running_words": r} for why, (d, r) in dropped.items()},
        "excluded_from_train": len(exclude),
        "splits": {name: lx.describe(c) for name, c in parts.items()},
    })
    out = Path(args.out_dir)
    for name, counts in parts.items():
        lx.write_counts(counts, out / f"{name}.tsv")
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, s in stats["splits"].items():
        print(f"{name:5}: {s['distinct_words']:>7,} distinct words, {s['running_words']:>9,} running")
    print(f"dropped: {stats['dropped']}")
    print(f"written to {out} and {args.stats}")


if __name__ == "__main__":
    main()
