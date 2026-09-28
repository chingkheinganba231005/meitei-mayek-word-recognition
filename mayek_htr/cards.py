"""Texts of the Hugging Face model card, the Space card and the demo page, filled from the
results files. Used by ``scripts/build_demo.py`` (a full build) and ``scripts/update_cards.py``
(the cards and the quoted scores only, without the trained network)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/chingkheinganba231005/meitei-mayek-word-recognition"
HF_ID = "Chingkheinganba/handwritten-meitei-mayek-word-recognition"
CANDIDATES = ("round2_convnext_tummhcd", "round2_convnext_tummhcd_seed1")
PAPER_RUN = "round2_convnext_tummhcd_seed1"    # the demo's run when the paper was written
PAPER_REVISION = "paper-v1"                     # the Hugging Face tag that keeps that version


def pct(x):
    return f"{100 * x:.2f}%"


def load_results(results, candidates=CANDIDATES, run=None):
    """The demo's run (the candidate with the lowest CER, then WER, with the language model on
    synthetic validation, unless given) and its validation, synthetic test and real-word results."""
    res = Path(results)
    read = lambda name: json.loads((res / name).read_text(encoding="utf-8")) if (res / name).exists() else None
    val = {r: read(f"phase2_val_{r}.json") for r in candidates}
    if run is None:
        run = min(candidates, key=lambda r: (val[r]["with_lm"]["cer"], val[r]["with_lm"]["wer"]))
    return (run, val.get(run) or read(f"phase2_val_{run}.json"), read(f"phase2_test_{run}.json"),
            read(f"phase3_real_{run}.json"))


def load_extras(results, run):
    """For a run: its synthetic test words with every sign above written beside its letter
    (round3_test_beside_<run>.json, or None) and the share of its training words written that
    way (from phase2_train_<run>.json; 0 when not recorded)."""
    res = Path(results)
    read = lambda name: json.loads((res / name).read_text(encoding="utf-8")) if (res / name).exists() else None
    train = read(f"phase2_train_{run}.json") or {}
    marks_beside = train.get("marks_beside", train.get("cfg", {}).get("marks_beside", 0.0)) or 0.0
    return read(f"round3_test_beside_{run}.json"), float(marks_beside)


def results_text(test, real, beside=None):
    """The scores quoted on the cards and the page."""
    parts = []
    if test:
        t = test["with_lm"]
        parts.append(f"On {t['words']:,} synthetic test words (TUMMHCD test characters) it reaches a character error "
                     f"rate of {pct(t['cer'])} and a word error rate of {pct(t['wer'])} with the language model "
                     f"({pct(test['greedy']['cer'])} and {pct(test['greedy']['wer'])} without).")
    if beside:
        b = beside["with_lm"]
        parts.append(f"On the same words with every sign that print places above a letter written beside it at the "
                     f"top: character error rate {pct(b['cer'])}, word error rate {pct(b['wer'])} with the language "
                     f"model.")
    if real:
        r = real["with_lm"]
        lo, hi = r["word_accuracy_95ci"]
        parts.append(f"On {r['words']} real handwritten words (one native writer, on a tablet; words it never saw in "
                     f"training): character error rate {pct(r['cer'])}, word error rate {pct(r['wer'])} with the "
                     f"language model (words read right {pct(r['word_accuracy'])}, 95% interval "
                     f"{pct(lo)}-{pct(hi)}); without it {pct(real['greedy']['cer'])} and {pct(real['greedy']['wer'])}.")
    return " ".join(parts)


def training_text(marks_beside):
    """What the cards say about where the training words put the signs above a letter."""
    if not marks_beside:
        return ""
    share = "half" if abs(marks_beside - 0.5) < 1e-9 else f"{100 * marks_beside:.0f}%"
    return (f" In {share} of the training words, the signs that print places above a letter (ꯥ, ꯩ, ꯪ) are "
            "written beside it at the top, where handwriting often puts them, as with ꯦ; in the others they "
            "stand above it, as in print.")


def versions_text(run, hf_id=HF_ID):
    """A pointer to the version described in the paper, when the demo runs another model."""
    if run == PAPER_RUN:
        return ""
    return (f"\n\nThe earlier model (run `{PAPER_RUN}`, trained with every sign in its printed place), which the "
            f"paper describes, is kept at revision [`{PAPER_REVISION}`](https://huggingface.co/{hf_id}/tree/"
            f"{PAPER_REVISION}) of this repository.")


def about_text(text, training=""):
    """The paragraph under the demo page."""
    return ("The first segmentation-free recogniser of handwritten Meitei Mayek words: ConvNeXt-T (started from "
            "a network trained on the TUMMHCD characters) with a bidirectional LSTM and CTC, trained on synthetic "
            "words composed from TUMMHCD characters, decoded with a character 6-gram language model." + training +
            " " + text + " Handwriting unlike the training words (very thin or very thick pens, photos with clutter) "
            "will be read less reliably.")


def cards(text, run, step, hf_id=HF_ID, training=""):
    """(model card, Space card) as Markdown, from the templates in space/."""
    fill = {"hf_id": hf_id, "repo_url": REPO_URL, "results": text, "run": run, "step": step, "training": training,
            "versions": versions_text(run, hf_id)}
    model = (ROOT / "space" / "MODEL_CARD.md").read_text(encoding="utf-8").format(**fill)
    space = (ROOT / "space" / "README.md").read_text(encoding="utf-8").format(**fill)
    return model, space


def results_json(run, val, test, real, beside=None):
    """results.json of the model repository."""
    out = {"run": run, "validation": val and val.get("with_lm"),
           "synthetic_test": test and {k: test[k] for k in ("greedy", "with_lm")},
           "real_words": real and {k: real[k] for k in ("greedy", "with_lm")}}
    if beside:
        out["synthetic_test_signs_beside"] = {k: beside[k] for k in ("greedy", "with_lm")}
    return out
