import importlib.util
import json
from pathlib import Path

from mayek_htr.cards import PAPER_REVISION, PAPER_RUN, load_extras, training_text, versions_text

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scores(cer, wer, words=5000):
    return {"cer": cer, "wer": wer, "words": words, "characters": 6 * words}


def test_training_and_versions_text():
    assert training_text(0.0) == "" and training_text(None) == ""
    assert "In half of the training words" in training_text(0.5) and "ꯥ, ꯩ, ꯪ" in training_text(0.5)
    assert "In 30% of the training words" in training_text(0.3)
    assert versions_text(PAPER_RUN) == ""                      # the paper's model needs no pointer to itself
    assert f"tree/{PAPER_REVISION}" in versions_text("round3_convnext_tummhcd")


def test_cards_of_a_run_trained_with_signs_beside(tmp_path):
    run, res = "round3_convnext_tummhcd", tmp_path / "results"
    res.mkdir()
    write = lambda name, obj: (res / name).write_text(json.dumps(obj), encoding="utf-8")
    write(f"phase2_val_{run}.json", {"step": 58000, "greedy": scores(0.003, 0.02), "with_lm": scores(0.0025, 0.017),
                                     "lm": {"alpha": 0.5, "beta": 0.0, "beam": 16}})
    write(f"phase2_test_{run}.json", {"greedy": scores(0.0034, 0.022), "with_lm": scores(0.0024, 0.0156)})
    write(f"round3_test_beside_{run}.json", {"greedy": scores(0.0041, 0.026), "with_lm": scores(0.0031, 0.0198)})
    write(f"phase2_train_{run}.json", {"cfg": {"marks_beside": 0.5}})
    beside, marks_beside = load_extras(res, run)
    assert marks_beside == 0.5 and beside["with_lm"]["cer"] == 0.0031

    paths = load("update_cards").write(res, tmp_path / "cards", run=run)
    model = paths["model/README.md"].read_text(encoding="utf-8")
    space = paths["space/README.md"].read_text(encoding="utf-8")
    about = paths["space/about.txt"].read_text(encoding="utf-8")
    for text in (model, space, about):
        assert "In half of the training words" in text
        assert "{" not in text.split("---", 2)[-1].replace("{\\", "")      # every placeholder filled
    assert "0.24%" in model and "0.31%" in model and "written beside it at the top" in model
    assert f"tree/{PAPER_REVISION}" in model and run in model
    saved = json.loads(paths["model/results.json"].read_text(encoding="utf-8"))
    assert saved["synthetic_test_signs_beside"]["with_lm"]["cer"] == 0.0031 and saved["real_words"] is None


def test_cards_of_the_paper_run_are_unchanged_in_substance(tmp_path):
    paths = load("update_cards").write(ROOT / "results", tmp_path, run=PAPER_RUN)
    model = paths["model/README.md"].read_text(encoding="utf-8")
    assert "training words, the signs" not in model and PAPER_REVISION not in model
    assert "3.72%" in model                                         # its real-word figure, as before
