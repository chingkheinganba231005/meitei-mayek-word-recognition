"""The browser demo's JavaScript (web/*.js, run with Node) against the Python code it ports,
the ONNX export against PyTorch, and the assembled site."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mayek_htr import images, labels
from mayek_htr.decode import beam_search, greedy
from mayek_htr.lm import EOS, CharLM, training_counts
from mayek_htr.web import SYMBOLS, build_site, export_lm
from mayek_words.charset import ALPHABET, LETTERS, VOWEL_SIGNS, standard_spelling

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node.js")


def run_js(job, tmp_path):
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    subprocess.run([NODE, str(ROOT / "tests" / "js_harness.mjs"), str(tmp_path / "job.json"),
                    str(tmp_path / "out.json")], check=True, capture_output=True, text=True)
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def word_image(h, w, seed, pen=3, paper=235, ink=40):
    """A word-like image: strokes of pen px on grey paper with noise."""
    rng = np.random.default_rng(seed)
    g = np.full((h, w), paper, np.float32)
    for _ in range(12):
        y, x = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 8, 7 * w // 8)
        dy, dx = rng.integers(-h // 4, h // 4), rng.integers(-w // 10, w // 10)
        for t in np.linspace(0, 1, 200):
            yy, xx = int(y + t * dy), int(x + t * dx)
            g[max(yy - pen // 2, 0):yy + pen - pen // 2, max(xx - pen // 2, 0):xx + pen - pen // 2] = ink
    return np.clip(g + rng.normal(0, 6, g.shape), 0, 255).astype(np.uint8)


@needs_node
def test_preprocessing_matches_python(tmp_path):
    cases = [(64, 300, False), (167, 495, True), (300, 900, True), (40, 150, False), (520, 1400, True),
             (90, 2400, False), (1000, 700, True)]
    grays = [word_image(h, w, k, pen=max(2, h // 25)) for k, (h, w, _) in enumerate(cases)]
    job = {"images": [{"w": int(g.shape[1]), "h": int(g.shape[0]), "data": g.ravel().tolist(), "crop": c}
                      for g, (_, _, c) in zip(grays, cases)]}
    out = run_js(job, tmp_path)["images"]
    for g, (_, _, crop), js in zip(grays, cases, out):
        src = images.crop_ink(g) if crop else g
        assert js["crop"] == [src.shape[1], src.shape[0]]
        ref = images.normalise(src)
        got = np.array(js["data"], np.uint8).reshape(js["h"], js["w"])
        assert got.shape == ref.shape
        diff = np.abs(got.astype(int) - ref.astype(int))
        assert diff.max() <= 1 and (diff > 0).mean() < 0.005, (g.shape, diff.max(), (diff > 0).mean())


def small_lm(order=6):
    rng = np.random.default_rng(0)
    letters, signs = sorted(LETTERS), sorted(VOWEL_SIGNS)
    words = {}
    while len(words) < 400:
        w = "".join(letters[rng.integers(len(letters))] + (signs[rng.integers(len(signs))] if rng.random() < 0.4
                                                          else "") for _ in range(rng.integers(1, 5)))
        words[w] = int(rng.integers(1, 50))
    return CharLM(order).fit(training_counts(words, power=0.0))


def peaked(T, rng):
    """Log probabilities of a word-like CTC output: a class per frame, with rivals."""
    C = labels.NUM_CLASSES
    x = rng.normal(0, 1.5, (T, C))
    for t in range(T):
        x[t, 0 if rng.random() < 0.45 else rng.integers(1, C)] += 7
        if rng.random() < 0.3:
            x[t, rng.integers(1, C)] += 6.5                   # a close rival
    return x - np.log(np.exp(x).sum(1, keepdims=True))


@needs_node
def test_decoding_matches_python(tmp_path):
    lm = small_lm()
    info = export_lm(lm, tmp_path / "lm.bin.gz")
    assert info["entries"] == sum(len(t) for t in lm.counts.values()) + sum(
        len(lm.context[n]) for n in range(2, lm.order + 1))
    rng = np.random.default_rng(1)
    chars = [c for c in ALPHABET]
    queries = []
    for _ in range(300):
        hist = [int(k) for k in rng.integers(1, len(ALPHABET) + 1, rng.integers(0, 8))]
        w = int(rng.integers(1, len(ALPHABET) + 2))        # a character or the end
        queries.append([w, hist])
    mats = [peaked(int(rng.integers(8, 60)), rng) for _ in range(40)]
    settings = {"alpha": 0.7, "beta": 1.0, "beam": 16, "prune": -10.0, "topk": 8}
    job = {"decode": {"lm": str(tmp_path / "lm.bin.gz"), "queries": queries, "settings": settings,
                      "matrices": [{"T": len(m), "C": m.shape[1], "data": m.ravel().tolist()} for m in mats]}}
    out = run_js(job, tmp_path)
    for (w, hist), got in zip(queries, out["logprobs"]):
        sym = EOS if w == SYMBOLS[EOS] else chars[w - 1]
        assert got == pytest.approx(lm.logprob(sym, "".join(chars[k - 1] for k in hist)), abs=2e-5)

    def text(ids):
        return "".join(chars[k - 1] for k in ids)

    for m, got in zip(mats, out["decode"]):
        assert text(got["greedy"]) == greedy(m)
        assert text(got["plain"]) == beam_search(m, None, alpha=0.0, beta=0.0)
        assert text(got["lm"]) == beam_search(m, lm, alpha=0.7, beta=1.0)


@needs_node
def test_standard_spelling_matches_python(tmp_path):
    words = ["ꯑꯥꯏ", "ꯂꯣꯏꯅ", "ꯃꯤꯇꯩꯏ", "ꯏꯃꯥ", "ꯍꯨꯏꯁꯤꯡ", ""]
    assert run_js({"spelling": words}, tmp_path)["spelling"] == [standard_spelling(w) for w in words]


def test_onnx_export(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("timm")
    torch = pytest.importorskip("torch")
    from mayek_htr.model import build
    from mayek_htr.web import check_onnx, export_onnx

    torch.manual_seed(0)
    model = build("convnext_tiny", init="none").eval()
    norms = [images.normalise(word_image(64, w, k)) for k, w in enumerate((100, 320, 700))]
    full = export_onnx(model, tmp_path / "full.onnx", half_weights=False)
    agree, diff = check_onnx(model, full, norms)
    assert agree == 1.0 and diff < 1e-4
    half = export_onnx(model, tmp_path / "half.onnx")
    assert half.stat().st_size < 0.6 * full.stat().st_size
    assert check_onnx(model, half, norms)[1] < 0.5


def test_build_site(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "lm.bin.gz").write_bytes(b"lm")
    site = build_site(tmp_path / "site", {"alpha": 0.5, "beta": 1.0, "beam": 16}, {"model_name": "test"},
                      onnx_path=tmp_path / "model.onnx", lm_path=tmp_path / "lm.bin.gz")
    for name in ("index.html", "app.js", "pipeline.js", "decode.js", "style.css", "model.onnx", "lm.bin.gz"):
        assert (site / name).exists(), name
    config = json.loads((site / "config.json").read_text(encoding="utf-8"))
    assert config["alphabet"] == list(ALPHABET) and config["model_url"] == "model.onnx"
    assert config["decoding"]["alpha"] == 0.5 and config["model_name"] == "test"


def test_build_demo_and_reader(tmp_path):
    """scripts/build_demo.py on a stand-in run: the run chosen on validation, the model
    repository and the Space; the Reader reads a word from the repository's files."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("timm")
    torch = pytest.importorskip("torch")
    from dataclasses import asdict

    from mayek_htr.model import build
    from mayek_htr.reader import Reader
    from mayek_htr.train import TrainConfig

    torch.manual_seed(0)
    runs, res = tmp_path / "runs", tmp_path / "results"
    res.mkdir()
    cfg = asdict(TrainConfig(encoder="convnext_tiny", init="none"))
    for k, run in enumerate(("a", "b")):
        (runs / run).mkdir(parents=True)
        torch.save({"model": build("convnext_tiny", init="none").state_dict(), "cfg": cfg, "step": 10 + k},
                   runs / run / "best.pt")
        score = {"words": 3, "cer": 0.01, "wer": 0.05 - 0.01 * k, "word_accuracy": 0.95, "word_accuracy_95ci": [0.9, 1]}
        (res / f"phase2_val_{run}.json").write_text(json.dumps(
            {"words": 3, "with_lm": score, "greedy": score, "lm": {"alpha": 0.25 * (k + 1), "beta": 1.0, "beam": 16}}))
    (res / "phase2_test_b.json").write_text(json.dumps({"with_lm": score, "greedy": score}))
    (res / "phase3_real_b.json").write_text(json.dumps({"with_lm": score, "greedy": score}))
    small_lm().save(tmp_path / "lm.pkl")
    check = tmp_path / "check"
    (check / "images").mkdir(parents=True)
    from PIL import Image
    for k in range(3):
        Image.fromarray(word_image(64, 200 + 60 * k, k)).save(check / "images" / f"{k:06d}.png")
    (check / "labels.tsv").write_text("".join(f"{k:06d}.png\tꯀ\n" for k in range(3)), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_demo.py"),
                        "--runs", str(runs), "--results", str(res), "--candidates", "a", "b", "--lm", str(tmp_path / "lm.pkl"),
                        "--check-set", str(check), "--check-n", "3", "--model-dir", str(tmp_path / "model"),
                        "--space-dir", str(tmp_path / "space"), "--hf-id", "someone/words",
                        "--summary", str(tmp_path / "summary.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run"] == "b" and summary["decoding"]["alpha"] == 0.5          # lower validation WER
    for name in ("recogniser.pt", "char_lm.pkl", "decoding.json", "results.json", "README.md", "web/model.onnx",
                 "web/lm.bin.gz"):
        assert (tmp_path / "model" / name).exists(), name
    config = json.loads((tmp_path / "space" / "config.json").read_text(encoding="utf-8"))
    assert config["model_url"] == "https://huggingface.co/someone/words/resolve/main/web/model.onnx"
    assert config["lm_url"].endswith("/web/lm.bin.gz") and config["decoding"]["alpha"] == 0.5
    card = (tmp_path / "space" / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---") and "sdk: static" in card and "someone/words" in card and "{" not in card
    assert "{" not in (tmp_path / "model" / "README.md").read_text(encoding="utf-8").replace("```", "")
    reader = Reader(tmp_path / "model")
    assert reader.decoding["alpha"] == 0.5 and reader.lm is not None
    text = reader.read(word_image(90, 400, 7))
    assert isinstance(text, str) and set(text) <= set(ALPHABET)
