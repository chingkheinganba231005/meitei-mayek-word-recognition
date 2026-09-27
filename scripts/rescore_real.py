"""Score the saved readings of the real word set again after its labels are corrected. The
networks are not run again.

    python scripts/rescore_real.py --set real_words --runs work/runs --results results \\
        --predictions-out work/rescored

Each run's real_predictions.tsv (file, reference, kind, greedy, with_lm; written by
scripts/eval_recogniser.py --predictions) is first scored against its own references and
must give exactly the scores in results/phase3_real_<run>.json, so the readings are the ones
behind those results. Then the same readings are scored against the set's labels
(labels.tsv), with the same code as eval_recogniser.py, and the results file is written again:
its other fields are kept, the scores replaced, and "labels" records which labels changed.
With --predictions-out, each predictions file is written again there with the corrected
references, for scripts/compare_runs.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr import metrics  # noqa: E402

SCORES = ("greedy", "greedy_by_kind", "with_lm", "with_lm_by_kind")


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def scores(refs, rows):
    kinds = [r["kind"] for r in rows] if all(r.get("kind") for r in rows) else None
    greedy = [r["greedy"] for r in rows]
    with_lm = [r["with_lm"] for r in rows]
    return {"greedy": metrics.score(refs, greedy), "greedy_by_kind": metrics.by_kind(refs, greedy, kinds),
            "with_lm": metrics.score(refs, with_lm), "with_lm_by_kind": metrics.by_kind(refs, with_lm, kinds)}


def rescore(result, rows, labels, corrections):
    """Stored result, predictions rows, labels {file: text} -> the result scored on the labels."""
    old = scores([r["reference"] for r in rows], rows)
    for key in SCORES:
        if result.get(key) != json.loads(json.dumps(old[key])):
            raise ValueError(f"the predictions do not reproduce the stored {key}")
    missing = [r["file"] for r in rows if r["file"] not in labels]
    if missing:
        raise ValueError(f"not in the set: {missing[:5]}")
    new = scores([labels[r["file"]] for r in rows], rows)
    out = dict(result)
    out.update(new)
    changed = [r["file"] for r in rows if labels[r["file"]] != r["reference"]]
    out["labels"] = {"corrected": [{"file": c["file"], "reason": c["reason"]} for c in corrections
                                   if c["file"] in changed],
                     "rescored_by": "scripts/rescore_real.py (readings unchanged)"}
    if len(out["labels"]["corrected"]) != len(changed):
        raise ValueError("a label differs from the predictions' reference without a recorded correction")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="the released set: labels.tsv and manifest.json")
    ap.add_argument("--runs", required=True, help="folder of run folders with real_predictions.tsv")
    ap.add_argument("--results", required=True, help="folder of phase3_real_<run>.json")
    ap.add_argument("--predictions-out", help="folder for the predictions with corrected references")
    args = ap.parse_args()

    root = Path(args.set)
    with open(root / "labels.tsv", encoding="utf-8") as f:
        labels = {r[0]: r[1] for r in csv.reader(f, delimiter="\t") if r}
    corrections = json.loads((root / "manifest.json").read_text(encoding="utf-8")).get("corrections", [])
    for pred in sorted(Path(args.runs).glob("*/real_predictions.tsv")):
        run = pred.parent.name
        path = Path(args.results) / f"phase3_real_{run}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = read_predictions(pred)
        out = rescore(result, rows, labels, corrections)
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        if args.predictions_out:
            dest = Path(args.predictions_out) / run / "real_predictions.tsv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
                w.writeheader()
                w.writerows({**r, "reference": labels[r["file"]]} for r in rows)
        g, l = out["greedy"], out["with_lm"]
        print(f"{run}: greedy CER {result['greedy']['cer']:.5f} -> {g['cer']:.5f}, WER {result['greedy']['wer']:.2f} -> "
              f"{g['wer']:.2f}; with LM CER {result['with_lm']['cer']:.5f} -> {l['cer']:.5f}, "
              f"WER {result['with_lm']['wer']:.2f} -> {l['wer']:.2f}")


if __name__ == "__main__":
    main()
