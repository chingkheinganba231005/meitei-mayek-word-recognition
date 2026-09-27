import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


release_mod, rescore_mod = load("release_real_set"), load("rescore_real")
K, A, T, STOP = chr(0xABC0), chr(0xABE5), chr(0xABC7), chr(0xABEB)


def fake_set(root):
    (root / "images").mkdir(parents=True)
    items = [{"item": 1, "page": 1, "text": K + A, "kind": "lexicon", "status": "written", "flags": [],
              "file": "000001.png", "size": [20, 40]},
             {"item": 2, "page": 1, "text": T + K + STOP, "kind": "lexicon", "status": "written", "flags": ["outside"],
              "file": "000002.png", "size": [20, 40]},
             {"item": 3, "page": 2, "text": "SECRET", "kind": "lexicon", "status": "page not given"}]
    for it in items[:2]:
        Image.new("L", (40, 20), 255).save(root / "images" / it["file"])
    (root / "labels.tsv").write_text("".join(f"{it['file']}\t{it['text']}\n" for it in items[:2]), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"format": "mayek-real-words/1", "title": "t", "lexicon": "test.tsv",
                                                    "selection": {"split": "test"}, "pages": [{"page": 1}],
                                                    "items": items}, ensure_ascii=False), encoding="utf-8")


def test_release_keeps_written_items_and_records_corrections(tmp_path):
    fake_set(tmp_path / "set")
    fix = {"000002.png": {"text": T + K, "reason": "full stop not written"}}
    info = release_mod.release(*release_mod.read_set(tmp_path / "set"), fix, tmp_path / "out")
    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "images").iterdir()) == ["000001.png", "000002.png"]
    assert (out / "labels.tsv").read_text(encoding="utf-8") == f"000001.png\t{K + A}\n000002.png\t{T + K}\n"
    everything = "".join(p.read_text(encoding="utf-8") for p in out.glob("*.*"))
    assert "SECRET" not in everything                         # items not written stay private
    assert info["words"] == 2 and info["characters"] == 4
    assert info["corrections"] == [{"item": 2, "file": "000002.png", "was": T + K + STOP, "now": T + K,
                                    "reason": "full stop not written"}]
    s = release_mod.summary(info)
    assert s["ending_with_full_stop"] == 0 and s["flagged"] == {"outside": 1} and s["characters"] == 4
    with pytest.raises(ValueError):
        release_mod.release(*release_mod.read_set(tmp_path / "set"), {"000009.png": {"text": K, "reason": "x"}},
                            tmp_path / "out2")


def test_rescore_checks_the_readings_and_applies_the_labels(tmp_path):
    rows = [{"file": "000001.png", "reference": K + A, "kind": "lexicon", "greedy": K, "with_lm": K + A},
            {"file": "000002.png", "reference": T + K + STOP, "kind": "lexicon", "greedy": T + K, "with_lm": T + K}]
    stored = {"checkpoint": "run/best.pt", "lm": {"alpha": 0.5}, **rescore_mod.scores([r["reference"] for r in rows], rows)}
    stored = json.loads(json.dumps(stored))
    assert stored["with_lm"]["wer"] == 0.5
    labels = {"000001.png": K + A, "000002.png": T + K}
    corrections = [{"file": "000002.png", "reason": "full stop not written"}]
    out = rescore_mod.rescore(stored, rows, labels, corrections)
    assert out["with_lm"]["wer"] == 0.0 and out["with_lm"]["characters"] == 4
    assert out["greedy"]["wer"] == 0.5                        # the greedy reading of item 1 is still wrong
    assert out["lm"] == {"alpha": 0.5} and out["labels"]["corrected"] == corrections
    with pytest.raises(ValueError):                           # readings that do not give the stored scores
        rescore_mod.rescore(stored, [dict(rows[0], with_lm=K), rows[1]], labels, corrections)
    with pytest.raises(ValueError):                           # a changed label without a recorded reason
        rescore_mod.rescore(stored, rows, labels, [])


def test_rescore_main_writes_results_and_predictions(tmp_path):
    fake_set(tmp_path / "set")
    release_mod.release(*release_mod.read_set(tmp_path / "set"),
                        {"000002.png": {"text": T + K, "reason": "full stop not written"}}, tmp_path / "rel")
    rows = [{"file": "000001.png", "reference": K + A, "kind": "lexicon", "greedy": K + A, "with_lm": K + A},
            {"file": "000002.png", "reference": T + K + STOP, "kind": "lexicon", "greedy": T + K, "with_lm": T + K}]
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    with open(run / "real_predictions.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "results").mkdir()
    stored = json.loads(json.dumps(rescore_mod.scores([r["reference"] for r in rows], rows)))
    (tmp_path / "results" / "phase3_real_r1.json").write_text(json.dumps(stored), encoding="utf-8")
    import sys
    argv = sys.argv
    sys.argv = ["rescore_real.py", "--set", str(tmp_path / "rel"), "--runs", str(tmp_path / "runs"),
                "--results", str(tmp_path / "results"), "--predictions-out", str(tmp_path / "new")]
    try:
        rescore_mod.main()
    finally:
        sys.argv = argv
    res = json.loads((tmp_path / "results" / "phase3_real_r1.json").read_text(encoding="utf-8"))
    assert res["with_lm"]["wer"] == 0.0 and res["labels"]["corrected"][0]["file"] == "000002.png"
    new = list(csv.DictReader(open(tmp_path / "new" / "r1" / "real_predictions.tsv", encoding="utf-8"), delimiter="\t"))
    assert [r["reference"] for r in new] == [K + A, T + K]


def test_update_cards_writes_the_quoted_scores(tmp_path):
    cards_mod = load("update_cards")
    paths = cards_mod.write(ROOT / "results", tmp_path, run="round2_convnext_tummhcd_seed1")
    real = json.loads((ROOT / "results" / "phase3_real_round2_convnext_tummhcd_seed1.json").read_text(encoding="utf-8"))
    cer = f"{100 * real['with_lm']['cer']:.2f}%"
    for name in ("model/README.md", "space/README.md", "space/about.txt"):
        text = paths[name].read_text(encoding="utf-8")
        assert cer in text and "{" not in text.split("---", 2)[-1].replace("{\\", "")
    saved = json.loads(paths["model/results.json"].read_text(encoding="utf-8"))
    assert saved["run"] == "round2_convnext_tummhcd_seed1" and saved["real_words"]["with_lm"] == real["with_lm"]
