"""Compare runs word by word on the same set, from their predictions files
(scripts/eval_recogniser.py --predictions: file, reference, kind, greedy, with_lm).

    python scripts/compare_runs.py --predictions a=work/runs/a/real_predictions.tsv \\
        b=work/runs/b/real_predictions.tsv --focus ꯥ --out results/phase3_real_comparison.json

For every run: the words misread greedily and with the language model, and how many the
language model puts right and wrong. For every two runs: the words only one of them reads
right, and the exact (McNemar) test of the difference. Across runs: the words every run
misreads and the words every run reads right. For each --focus character: how many words
contain it (for several characters: any of them), and each run's misread words among them
and among the others. Counts only, no
words: the predictions stay with the set.
"""

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.metrics import paired_exact  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f, delimiter="\t")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True, help="name=predictions.tsv")
    ap.add_argument("--focus", nargs="*", default=[], help="characters to look at (ꯥꯩ: either)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = dict(p.split("=", 1) for p in args.predictions)
    P = {name: read(path) for name, path in runs.items()}
    files = sorted(next(iter(P.values())))
    for name, rows in P.items():
        if sorted(rows) != files or any(rows[f]["reference"] != P[next(iter(P))][f]["reference"] for f in files):
            raise SystemExit(f"{name}: not the same words")

    def right(name, f, key="with_lm"):
        return P[name][f][key] == P[name][f]["reference"]

    out = {"words": len(files), "runs": {}, "pairs": {}, "focus": {}}
    for name in P:
        fixed = sum(right(name, f) and not right(name, f, "greedy") for f in files)
        broken = sum(right(name, f, "greedy") and not right(name, f) for f in files)
        out["runs"][name] = {"misread_greedy": sum(not right(name, f, "greedy") for f in files),
                             "misread_with_lm": sum(not right(name, f) for f in files),
                             "lm_puts_right": fixed, "lm_puts_wrong": broken, "p": round(paired_exact(fixed, broken), 4)}
    for a, b in combinations(P, 2):
        for key in ("with_lm", "greedy"):
            x = sum(right(a, f, key) and not right(b, f, key) for f in files)
            y = sum(right(b, f, key) and not right(a, f, key) for f in files)
            out["pairs"].setdefault(f"{a} | {b}", {})[key] = {"only_first_right": x, "only_second_right": y,
                                                               "p": round(paired_exact(x, y), 4)}
    out["misread_by_every_run"] = sum(all(not right(n, f) for n in P) for f in files)
    out["read_right_by_every_run"] = sum(all(right(n, f) for n in P) for f in files)
    for ch in args.focus:                       # words with any of the item's characters
        has = [f for f in files if set(ch) & set(P[next(iter(P))][f]["reference"])]
        rest = [f for f in files if f not in has]
        out["focus"][ch] = {"words_with_it": len(has), "words_without": len(rest),
                            "misread_with_lm": {n: [sum(not right(n, f) for f in has), sum(not right(n, f) for f in rest)]
                                                for n in P}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for n, r in out["runs"].items():
        print(f"{n}: misread {r['misread_greedy']} greedy, {r['misread_with_lm']} with the language model "
              f"(it puts {r['lm_puts_right']} right, {r['lm_puts_wrong']} wrong; p = {r['p']})")
    print(f"misread by every run: {out['misread_by_every_run']}; read right by every run: {out['read_right_by_every_run']}")


if __name__ == "__main__":
    main()
