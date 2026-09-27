"""Build the Hugging Face model repository and the browser demo (a static Space) from a
trained run, as the first project did for its character network.

    python scripts/build_demo.py --runs work/runs --results work/results --lm work/lm/char_lm.pkl \\
        --check-set work/synth/val.tar --model-dir work/hf/model --space-dir work/hf/space \\
        --hf-id Chingkheinganba/handwritten-meitei-mayek-word-recognition --summary results/phase3_demo.json

The run: of the final recipe's runs (--candidates), the one with the lowest CER, then WER,
with the language model on the synthetic validation set (phase2_val_<run>.json); its
decoding weights come from the same file. The ONNX export is checked against PyTorch on
--check-n words of the synthetic validation set; if float16 weights change more than 1%
of the greedy readings, the weights are kept in float32. The cards quote the synthetic test
(phase2_test_<run>.json) and, if present, the real words (phase3_real_<run>.json).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from mayek_htr.data import FixedSet  # noqa: E402
from mayek_htr.lm import CharLM  # noqa: E402
from mayek_htr.train import load_checkpoint  # noqa: E402
from mayek_htr.web import build_site, check_onnx, export_lm, export_onnx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/chingkheinganba231005/meitei-mayek-word-recognition"


def pct(x):
    return f"{100 * x:.2f}%"


def results_text(test, real):
    """The scores quoted on the cards and the page."""
    parts = []
    if test:
        t = test["with_lm"]
        parts.append(f"On {t['words']:,} synthetic test words (TUMMHCD test characters) it reaches a character error "
                     f"rate of {pct(t['cer'])} and a word error rate of {pct(t['wer'])} with the language model "
                     f"({pct(test['greedy']['cer'])} and {pct(test['greedy']['wer'])} without).")
    if real:
        r = real["with_lm"]
        lo, hi = r["word_accuracy_95ci"]
        parts.append(f"On {r['words']} real handwritten words (one native writer, on a tablet; words it never saw in "
                     f"training): character error rate {pct(r['cer'])}, word error rate {pct(r['wer'])} with the "
                     f"language model (words read right {pct(r['word_accuracy'])}, 95% interval "
                     f"{pct(lo)}-{pct(hi)}); without it {pct(real['greedy']['cer'])} and {pct(real['greedy']['wer'])}.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with the runs (<run>/best.pt)")
    ap.add_argument("--results", required=True, help="folder with phase2_val_*.json (and test, real)")
    ap.add_argument("--candidates", nargs="+", default=["round2_convnext_tummhcd", "round2_convnext_tummhcd_seed1"])
    ap.add_argument("--lm", required=True)
    ap.add_argument("--check-set", required=True, help="synthetic validation set, for the ONNX check")
    ap.add_argument("--check-n", type=int, default=300)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--space-dir", required=True)
    ap.add_argument("--hf-id", default="Chingkheinganba/handwritten-meitei-mayek-word-recognition")
    ap.add_argument("--no-lm-in-browser", action="store_true", help="the page decodes greedily")
    ap.add_argument("--summary", help="what was built, as .json")
    args = ap.parse_args()

    res = Path(args.results)
    val = {r: json.loads((res / f"phase2_val_{r}.json").read_text(encoding="utf-8")) for r in args.candidates}
    run = min(args.candidates, key=lambda r: (val[r]["with_lm"]["cer"], val[r]["with_lm"]["wer"]))
    test_f, real_f = res / f"phase2_test_{run}.json", res / f"phase3_real_{run}.json"
    test = json.loads(test_f.read_text(encoding="utf-8")) if test_f.exists() else None
    real = json.loads(real_f.read_text(encoding="utf-8")) if real_f.exists() else None
    chosen = val[run]["lm"]
    decoding = {"alpha": chosen["alpha"], "beta": chosen["beta"], "beam": chosen["beam"],
                "chosen_on": f"phase2_val_{run}.json (synthetic validation, {val[run]['words']} words)"}
    print(f"demo run: {run} (validation with the language model: "
          + ", ".join(f"{r} CER {val[r]['with_lm']['cer']:.4g} WER {val[r]['with_lm']['wer']:.4g}" for r in args.candidates)
          + f"); alpha {decoding['alpha']}, beta {decoding['beta']}")

    model, ck = load_checkpoint(Path(args.runs) / run / "best.pt")
    out = Path(args.model_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "web").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "step": ck.get("step")}, out / "recogniser.pt")
    shutil.copy(args.lm, out / "char_lm.pkl")
    (out / "decoding.json").write_text(json.dumps(decoding, indent=1), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"run": run, "validation": val[run].get("with_lm"),
                                                  "synthetic_test": test and {k: test[k] for k in ("greedy", "with_lm")},
                                                  "real_words": real and {k: real[k] for k in ("greedy", "with_lm")}},
                                                 indent=1, ensure_ascii=False), encoding="utf-8")

    check = FixedSet(args.check_set, limit=args.check_n)
    onnx_path = export_onnx(model, out / "web" / "model.onnx")
    agree, diff = check_onnx(model, onnx_path, check.images)
    half = agree >= 0.99
    if not half:
        onnx_path = export_onnx(model, out / "web" / "model.onnx", half_weights=False)
        agree, diff = check_onnx(model, onnx_path, check.images)
    print(f"ONNX ({'float16' if half else 'float32'} weights): the same greedy reading as PyTorch for "
          f"{agree:.1%} of {len(check)} validation words (largest log probability difference {diff:.4f})")
    lm_info = None
    if not args.no_lm_in_browser:
        lm_info = export_lm(CharLM.load(args.lm), out / "web" / "lm.bin.gz")
        print(f"language model for the browser: {lm_info['entries']:,} entries, {lm_info['file_mb']} MB")

    text = results_text(test, real)
    fill = {"hf_id": args.hf_id, "repo_url": REPO_URL, "results": text, "run": run, "step": ck.get("step")}
    (out / "README.md").write_text((ROOT / "space" / "MODEL_CARD.md").read_text(encoding="utf-8").format(**fill),
                                   encoding="utf-8")

    base = f"https://huggingface.co/{args.hf_id}/resolve/main/web/"
    model_mb = round(onnx_path.stat().st_size / 1e6, 1)
    about = ("ConvNeXt-T (from the first project's TUMMHCD network) with a bidirectional LSTM and CTC, trained on "
             "synthetic words composed from TUMMHCD characters, decoded with a character 6-gram language model. "
             + text + " Handwriting unlike the training words (very thin or very thick pens, photos with clutter) "
             "will be read less reliably.")
    site = build_site(args.space_dir, decoding, {"about": about, "repo_url": REPO_URL, "run": run},
                      model_url=base + "model.onnx", lm_url=None if lm_info is None else base + "lm.bin.gz",
                      model_mb=model_mb, lm_mb=lm_info and lm_info["file_mb"])
    (site / "README.md").write_text((ROOT / "space" / "README.md").read_text(encoding="utf-8").format(**fill),
                                    encoding="utf-8")
    print(f"model repository in {out}, Space in {site}")
    if args.summary:
        summary = {"run": run, "decoding": decoding, "onnx": {"float16": half, "same_greedy_reading": round(agree, 4),
                                                              "max_logprob_diff": round(diff, 5), "checked_on": len(check),
                                                              "mb": model_mb},
                   "lm_in_browser": lm_info, "hf_id": args.hf_id, "results_quoted": text}
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
