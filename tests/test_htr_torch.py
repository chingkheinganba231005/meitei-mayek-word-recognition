"""The recogniser's PyTorch parts: model shapes, loading the first paper's weights, data
streams and sets, augmentation, and that training learns and resumes. Skipped without
torch (and timm for the ConvNeXt tests)."""

import json
import tarfile

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mayek_htr import data, labels  # noqa: E402
from mayek_htr.augment import PRESETS, augment  # noqa: E402
from mayek_htr.model import Recogniser, SmallCNN, build, load_tummhcd  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402
from mayek_words.lexicon import Lexicon  # noqa: E402
from mayek_words.synth import Words, WordSynth, load_priors  # noqa: E402

K, AA, LAI, INAP, MIT = chr(0xABC0), chr(0xABE5), chr(0xABC2), chr(0xABE4), chr(0xABC3)
WORDS = {K + AA: 50, LAI + INAP: 30, MIT + AA + K: 20, K + INAP + LAI: 10}


@pytest.fixture(scope="module")
def words():
    return Words(WordSynth(GlyphStore.from_font(), load_priors(sizes=None)), Lexicon(WORDS), seed=3)


def write_set(words, folder, n=6):
    """A fixed set as scripts/render_words.py writes it."""
    (folder / "images").mkdir(parents=True)
    lines = []
    for i in range(n):
        s = words[i]
        Image.fromarray(s.image).save(folder / "images" / f"{i:06d}.png")
        lines.append(f"{i:06d}.png\t{s.text}\n")
    (folder / "labels.tsv").write_text("".join(lines), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps({"n": n, "seed": words.seed, "numbers": words.numbers,
                                                    "stop": words.stop, "built": words.built}), encoding="utf-8")
    return folder


def test_small_cnn_shapes():
    model = build("small_cnn", init="none").eval()
    x = torch.zeros(2, 1, 64, 96)
    logits, lengths = model(x, torch.tensor([96, 50]))
    assert logits.shape == (2, 24, labels.NUM_CLASSES)
    assert lengths.tolist() == [24, 13]                      # one column per 4 px, rounded up


def test_convnext_shapes_and_tummhcd_weights(tmp_path):
    timm = pytest.importorskip("timm")
    model = build("convnext_tiny", init="none").eval()
    f = model.encoder(torch.zeros(1, 1, 64, 80))
    assert f.shape == (1, 384, 4, 10)                        # 4 rows, one column per 8 px
    logits, lengths = model(torch.zeros(1, 1, 64, 80), torch.tensor([80]))
    assert logits.shape == (1, 10, labels.NUM_CLASSES) and lengths.tolist() == [10]

    # a model folder as the first paper exports it: config.json and state dicts with backbone.* keys
    net = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, in_chans=1)
    state = {"backbone." + k: v for k, v in net.state_dict().items()}
    state["head.weight"] = torch.zeros(55, 768)
    folder = tmp_path / "release" / "model"
    folder.mkdir(parents=True)
    torch.save(state, folder / "convnext_t.pt")
    torch.save({}, folder / "convnext_t_meta.pt")
    members = [{"name": "convnext_t_meta", "file": "convnext_t_meta.pt",
                "cfg": {"arch": "convnext_tiny.fb_in22k_ft_in1k", "channels": "gray", "meta": True}},
               {"name": "convnext_t", "file": "convnext_t.pt",
                "cfg": {"arch": "convnext_tiny.fb_in22k_ft_in1k", "channels": "gray"}}]
    (folder / "config.json").write_text(json.dumps({"img": 128, "members": members}))
    loaded = load_tummhcd(tmp_path)                           # found below the folder given; plain member
    enc = build("convnext_tiny", init="none").encoder
    assert enc.load_backbone(loaded) == []
    w = net.state_dict()["stages.2.downsample.1.weight"]
    assert torch.equal(enc.state_dict()["stages.2.downsample.1.conv.weight"], w)
    assert torch.equal(enc.state_dict()["stem.0.weight"], net.state_dict()["stem.0.weight"])


def test_synth_stream_batches(words):
    stream = data.SynthStream(words, batch_size=4, pool=2, start=10)
    it = iter(stream)
    a, b = next(it), next(it)
    for batch in (a, b):
        assert batch["x"].dtype == torch.uint8 and batch["x"].shape[:2] == (4, 1) and batch["x"].shape[2] == 64
        assert int(batch["target_lengths"].sum()) == len(batch["targets"])
        assert [labels.decode(labels.encode(t)) for t in batch["texts"]] == batch["texts"]
        assert batch["widths"].max() <= batch["x"].shape[3]
    texts = sorted(a["texts"] + b["texts"])
    assert texts == sorted(words[i].text for i in range(10, 18))   # items start, start + 1, ...
    again = next(iter(data.SynthStream(words, batch_size=4, pool=2, start=10)))
    assert torch.equal(again["x"], a["x"])                   # the same words, the same order


def test_synth_stream_skips_a_word_it_cannot_draw(words, capsys):
    class Failing:
        seed = words.seed

        def __getitem__(self, i):
            if i == 2:
                raise AssertionError("canvas too small")
            return words[i]

    batch = next(iter(data.SynthStream(Failing(), batch_size=4, pool=1)))
    assert sorted(batch["texts"]) == sorted(words[i].text for i in (0, 1, 3))
    assert "skipped training word 2" in capsys.readouterr().err


def test_fixed_set_folder_tar_and_kinds(words, tmp_path):
    folder = write_set(words, tmp_path / "set", n=6)
    with tarfile.open(tmp_path / "set.tar", "w") as tar:
        tar.add(folder, arcname=".")
    a, b = data.FixedSet(folder), data.FixedSet(tmp_path / "set.tar")
    assert a.texts == b.texts == [words[i].text for i in range(6)] and a.names == b.names
    assert all(np.array_equal(x, y) for x, y in zip(a.images, b.images))
    batches = list(a.batches(batch_size=4))
    assert sorted(i for bt in batches for i in bt["index"].tolist()) == list(range(6))
    # the kind of each word, told from the seed as Words draws it
    w = Words(words.synth, words.lexicon, seed=5, numbers=0.3, stop=0.0, built=0.3)
    info = {"seed": 5, "numbers": 0.3, "built": 0.3}
    got = data.kinds(info, [f"images/{i:06d}.png" for i in range(40)])
    assert {"number", "syllables", "lexicon"} <= set(got)
    for i, kind in enumerate(got):
        text = w.text(np.random.default_rng([5, i]))
        if kind == "number":
            assert all(0xABF0 <= ord(c) <= 0xABF9 for c in text)
        elif kind == "lexicon":
            assert text in WORDS


def test_augment_shapes_and_range():
    torch.manual_seed(0)
    x = torch.zeros(3, 1, 64, 128)
    x[:, :, 20:44, 10:60] = 1.0
    widths = torch.tensor([128, 70, 60])
    for name, aug in PRESETS.items():
        y = augment(x.clone(), widths, aug)
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
        if aug is None:
            assert torch.equal(y, x)
        else:
            assert float((y > 0.5).float().sum()) > 0.3 * float(x.sum())   # the writing survives


def test_ctc_learns_a_batch(words):
    torch.manual_seed(0)
    batch = data.make_batch([(data.images.normalise(words[i].image), words[i].text) for i in range(4)])
    model = Recogniser(SmallCNN(), hidden=64, layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctc = torch.nn.CTCLoss(zero_infinity=True)
    x = batch["x"].float() / 255
    losses = []
    for _ in range(60):
        logits, lengths = model(x, batch["widths"])
        loss = ctc(logits.log_softmax(-1).transpose(0, 1), batch["targets"], lengths, batch["target_lengths"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_train_writes_and_resumes(words, tmp_path):
    from mayek_htr.train import TrainConfig, load_checkpoint, train

    val = data.FixedSet(write_set(words, tmp_path / "val", n=4))
    cfg = TrainConfig(encoder="small_cnn", init="none", steps=4, batch=4, pool=1, workers=0, warmup=2,
                      eval_every=2, save_every=2, log_every=2, hidden=32, layers=1)
    run = tmp_path / "run"
    history = train(cfg, words, val, run, device="cpu", log=lambda m: None)
    assert [h["step"] for h in history] == [2, 4]
    assert all((run / f).exists() for f in ("best.pt", "final.pt", "last.pt", "history.json"))
    cfg.steps = 6
    history = train(cfg, words, val, run, device="cpu", log=lambda m: None)
    assert [h["step"] for h in history] == [2, 4, 6]         # went on from step 4
    model, ck = load_checkpoint(run / "final.pt")
    assert ck["step"] == 6 and not model.training
    for f in run.iterdir():                                   # the run folder emptied: a new run,
        f.unlink()                                            # not the local copy of the old one
    history = train(cfg, words, val, run, device="cpu", log=lambda m: None)
    assert [h["step"] for h in history] == [2, 4, 6] and (run / "best.pt").exists()


def test_dev_ensemble(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "dev_ensemble.py"
    release = tmp_path / "release" / "model"
    release.mkdir(parents=True)
    members = [{"name": "net_a", "file": "net_a.pt", "cfg": {"arch": "convnext_tiny.fb_in22k_ft_in1k",
                                                              "channels": "gray"}, "views": ["id"], "weight": 0.5},
               {"name": "net_a_meta", "file": "net_a_meta.pt", "cfg": {"arch": "convnext_tiny", "channels": "gray",
                                                                       "meta": True}, "views": ["id"], "weight": 0.5}]
    (release / "config.json").write_text(json.dumps({"img": 128, "stats": {}, "members": members}))
    runs = tmp_path / "runs"
    for m in members:
        weights = {"w": torch.randn(3, 2), "steps": torch.tensor(5)}
        torch.save(weights, release / m["file"])
        (runs / m["name"] / "dev").mkdir(parents=True)
        dev = {"w": weights["w"] + 1.0, "steps": torch.tensor(5)}
        torch.save({"model": dev, "cfg": m["cfg"], "history": [{"epoch": 1, "val_acc": 0.98}]},
                   runs / m["name"] / "dev" / "final.pt")

    def make(out):
        return subprocess.run([sys.executable, str(script), "--release", str(tmp_path / "release"),
                               "--runs", str(runs), "--out", str(out)], capture_output=True, text=True)

    r = make(tmp_path / "dev")
    assert r.returncode == 0, r.stderr
    config = json.loads((tmp_path / "dev" / "config.json").read_text())
    assert [m["name"] for m in config["members"]] == ["net_a", "net_a_meta"] and "validation" in config["trained_on"]
    assert torch.equal(torch.load(tmp_path / "dev" / "net_a.pt")["w"], torch.load(release / "net_a.pt")["w"] + 1.0)

    torch.save({"model": torch.load(release / "net_a.pt"), "cfg": members[0]["cfg"]},
               runs / "net_a" / "dev" / "final.pt")                      # the released weights: refused
    r = make(tmp_path / "bad")
    assert r.returncode != 0 and "released weights themselves" in r.stderr
    (runs / "net_a" / "dev" / "final.pt").unlink()                         # a member missing: refused
    r = make(tmp_path / "bad")
    assert r.returncode != 0 and "missing" in r.stderr
