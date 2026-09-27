"""The browser demo: the recogniser exported to ONNX, the language model in a compact
binary file, and the static page in ``web/``.

Everything runs in the visitor's browser with ONNX Runtime Web, as in the first project's
character demo, so the demo can be hosted for free as a static Hugging Face Space. The
page reimplements the preprocessing of ``images.py`` and the decoding of ``decode.py``
and ``lm.py`` in JavaScript (``web/pipeline.js``, ``web/decode.js``);
``tests/test_web.py`` checks them against the Python versions.

- ``export_onnx``: one word in, (image (1, 1, 64, W) with ink in [0, 1], padded with paper
  to a multiple of 32 px; length: the columns the word fills) -> natural log
  probabilities (length, 55), as ``train.predict`` computes them for one word; weights
  stored as float16 to halve the download.
- ``export_lm``: the character n-gram model in back-off form (every n-gram seen with its
  log probability, every context seen with its log back-off weight), which gives exactly
  the probabilities of ``lm.CharLM``; gzip-compressed.
- ``build_site``: the page, its config.json and, if given, the model files.
"""

import gzip
import io
import json
import math
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .labels import ALPHABET, NUM_CLASSES
from .lm import BOS, EOS

ORT_VERSION = "1.30.0"
ORT_CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
ORT_FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
WEB_SRC = Path(__file__).resolve().parents[1] / "web"
SYMBOLS = {BOS: 0, **{ch: i + 1 for i, ch in enumerate(ALPHABET)}, EOS: NUM_CLASSES}   # as the class ids


def _single(model):
    import torch

    from .model import MEAN, STD

    class Single(torch.nn.Module):
        """One word through Recogniser.forward without packing: the LSTM reads the first
        `length` columns only, as the packed sequence does in the evaluation."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, length):
            f = self.m.encoder((image - MEAN) / STD)
            B, C, h, T = f.shape
            f = self.m.proj(f.permute(0, 3, 1, 2).reshape(B, T, C * h))[:, :length[0]]
            out, _ = self.m.rnn(f)
            return self.m.head(out).log_softmax(-1)[0]

    return Single(model.eval().float().cpu()).eval()


def export_onnx(model, path, half_weights=True):
    """Export a trained Recogniser for one word of any width."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = _single(model)
    args = (torch.zeros(1, 1, 64, 256), torch.tensor([32]))
    kwargs = dict(input_names=["image", "length"], output_names=["logp"], opset_version=17,
                  dynamic_axes={"image": {3: "width"}, "logp": {0: "frames"}})
    try:
        torch.onnx.export(net, args, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(net, args, str(path), **kwargs)
    if half_weights:
        _store_half(path)
    return path


def _store_half(path):
    """Large float32 initializers become float16 plus a Cast node (as in the first project)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    casts, halves = [], []
    for init in list(g.initializer):
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 256:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            halves.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(halves)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def model_input(norm, stride=8, multiple=32):
    """A normalised word (uint8, ink 255, 64 px high) -> the ONNX inputs, as the page makes
    them (web/pipeline.js modelInput)."""
    h, w = norm.shape
    W = -(-w // multiple) * multiple
    x = np.zeros((1, 1, h, W), np.float32)
    x[0, 0, :, :w] = norm / np.float32(255)
    frames = min(max(-(-w // stride), 1), W // stride)
    return x, np.array([frames], np.int64)


def check_onnx(model, path, norms):
    """The exported network against PyTorch on normalised words: (share of words read the
    same by greedy decoding, largest difference of a log probability)."""
    import onnxruntime as ort
    import torch

    from .decode import greedy

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    net = _single(model)
    same, diff = 0, 0.0
    for norm in norms:
        x, n = model_input(norm)
        got = sess.run(None, {"image": x, "length": n})[0]
        with torch.no_grad():
            ref = net(torch.from_numpy(x), torch.from_numpy(n)).numpy()
        same += greedy(got) == greedy(ref)
        diff = max(diff, float(np.abs(got - ref).max()))
    return same / max(len(norms), 1), diff


def export_lm(lm, path):
    """A lm.CharLM -> a gzip-compressed binary file for web/decode.js (CharLM)."""
    S = len(SYMBOLS)
    D = lm.discount
    tables, arrays = [], []

    def key(symbols):
        k = 0
        for s in symbols:
            k = k * S + SYMBOLS[s]
        return k

    for n in range(1, lm.order + 1):
        items = [(key(h + (w,)), math.log(lm._p(w, h, n))) for (h, w) in lm.counts[n]]
        items.sort()
        tables.append({"kind": "ngram", "n": n, "count": len(items)})
        arrays.append((np.array([k for k, _ in items], "<f8"), np.array([v for _, v in items], "<f4")))
        if n >= 2:
            ctx = sorted((key(h), math.log(D * types / tot)) for h, (tot, types) in lm.context[n].items())
            tables.append({"kind": "context", "n": n, "count": len(ctx)})
            arrays.append((np.array([k for k, _ in ctx], "<f8"), np.array([v for _, v in ctx], "<f4")))
    tot, types = lm.context[1][()]
    head = {"order": lm.order, "symbols": S, "eos": SYMBOLS[EOS], "discount": D,
            "unigram_floor": math.log(D * types / tot / len(lm.vocab)), "tables": tables}

    # layout: "MMLM", version, header length, 0, the header (JSON, padded with spaces),
    # then per table its keys (float64) and values (float32), each at a multiple of 8 bytes
    def layout(body_at):
        at = body_at
        for t, (k, v) in zip(tables, arrays):
            t["keys_at"] = at
            at += k.nbytes
            t["vals_at"] = at
            at += v.nbytes + (-v.nbytes % 8)
        return json.dumps(head).encode()

    body_at = 16 + len(layout(0)) + 512
    body_at += -body_at % 8
    header = layout(body_at)
    if 16 + len(header) > body_at:
        raise RuntimeError("language model header does not fit")
    header += b" " * (body_at - 16 - len(header))
    buf = io.BytesIO()
    buf.write(b"MMLM")
    buf.write(np.array([1, len(header), 0], "<u4").tobytes())
    buf.write(header)
    for (k, v) in arrays:
        buf.write(k.tobytes())
        buf.write(v.tobytes())
        buf.write(b"\0" * (-v.nbytes % 8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = buf.getvalue()
    path.write_bytes(gzip.compress(data, 9) if path.suffix == ".gz" else data)
    return {"entries": int(sum(t["count"] for t in tables)), "mb": round(len(data) / 1e6, 2),
            "file_mb": round(path.stat().st_size / 1e6, 2)}


def fetch_ort(dest, tarball=None):
    """Copy the ONNX Runtime Web files the page needs (from npm, or a local npm tarball)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if tarball is None:
        url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
        data = urllib.request.urlopen(url, timeout=300).read()
    else:
        data = Path(tarball).read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name in ORT_FILES:
            (dest / name).write_bytes(tar.extractfile(f"package/dist/{name}").read())


def build_site(out_dir, decoding, info, onnx_path=None, lm_path=None, model_url="model.onnx", lm_url="lm.bin.gz",
               bundle_ort=False, ort_tarball=None, model_mb=None, lm_mb=None):
    """Assemble the static site: the page, config.json and, if given, the model and the
    language model (copied next to the page; else the page downloads them from model_url
    and lm_url, lm_url None for greedy decoding only). decoding: alpha, beta, beam as chosen
    on validation. info: shown on the page (model, results, links). bundle_ort: copy ONNX
    Runtime Web next to the page (from npm or ort_tarball) instead of the jsDelivr CDN."""
    from .images import HEIGHT, MAX_WIDTH

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(WEB_SRC, out_dir)
    if onnx_path is not None:
        shutil.copy(onnx_path, out_dir / "model.onnx")
        model_url = "model.onnx"
    if lm_path is not None:
        shutil.copy(lm_path, out_dir / "lm.bin.gz")
        lm_url = "lm.bin.gz"
    ort_base = ORT_CDN
    if bundle_ort or ort_tarball is not None:
        fetch_ort(out_dir / "ort", ort_tarball)
        ort_base = "ort/"
    config = {"alphabet": list(ALPHABET), "height": HEIGHT, "max_width": MAX_WIDTH, "stride": 8,
              "model_url": model_url, "lm_url": lm_url, "ort_base": ort_base, "model_mb": model_mb, "lm_mb": lm_mb,
              "decoding": {"alpha": decoding["alpha"], "beta": decoding["beta"], "beam": decoding.get("beam", 16),
                           "prune": decoding.get("prune", -10.0), "topk": decoding.get("topk", 8)},
              **info}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    return out_dir
