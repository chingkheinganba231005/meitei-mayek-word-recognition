"""Batches of word images for training and evaluation (PyTorch).

``SynthStream``: endless training batches of synthetic words, rendered on the fly by the
DataLoader's worker processes (``mayek_words.synth.Words``: item i is always the same
image). Worker w of n renders items start + w, start + w + n, ...; it renders `pool`
batches' worth of words at a time and groups them by width, so a batch pads little.
Build the Words (and its glyph store) before the DataLoader: the workers are forked and
share it. A word the synthesiser fails to draw is skipped with a message (item i always
fails the same way, so it would otherwise stop every resumed run at the same place).

``FixedSet``: a fixed set of word images with their text: a folder or .tar written by
``scripts/render_words.py --out-dir`` (images/NNNNNN.png and labels.tsv, file<TAB>text),
such as the synthetic validation and test sets, or later the real set in the same form.
A synthetic set also holds config.json (how it was rendered); ``kinds`` tells from it which
words are lexicon words, words composed of syllables, or numbers.

A batch is a dict: x uint8 (B, 1, H, W) ink high, widths (B,), targets (all label ids
concatenated), target_lengths (B,), texts, and index (the items' positions).
"""

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from . import images, labels


def make_batch(items, index=None):
    """[(uint8 normalised image, text)] -> batch dict."""
    x, widths = images.pad_batch([im for im, _ in items])
    ids = [labels.encode(t) for _, t in items]
    return {"x": torch.from_numpy(x).unsqueeze(1), "widths": torch.from_numpy(widths),
            "targets": torch.tensor([k for s in ids for k in s], dtype=torch.long),
            "target_lengths": torch.tensor([len(s) for s in ids], dtype=torch.long),
            "texts": [t for _, t in items],
            "index": torch.as_tensor(index if index is not None else np.arange(len(items)))}


class SynthStream(IterableDataset):
    def __init__(self, words, batch_size=64, pool=8, start=0, height=images.HEIGHT, max_width=images.MAX_WIDTH):
        self.words, self.batch_size, self.pool, self.start = words, batch_size, pool, start
        self.height, self.max_width = height, max_width

    def __iter__(self):
        info = get_worker_info()
        w, n = (info.id, info.num_workers) if info is not None else (0, 1)
        i = self.start + w
        order = np.random.default_rng([self.words.seed, self.start, w])
        while True:
            items, failed = [], 0
            for _ in range(self.batch_size * self.pool):
                try:
                    s = self.words[i]
                    items.append((images.normalise(s.image, self.height, self.max_width), s.text))
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"skipped training word {i}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    if failed > self.batch_size * self.pool // 2:
                        raise
                i += n
            items.sort(key=lambda it: it[0].shape[1])
            chunks = [items[k:k + self.batch_size] for k in range(0, len(items), self.batch_size)]
            for c in order.permutation(len(chunks)):
                yield make_batch(chunks[c])


def _read_set(path):
    """-> (names, texts, uint8 greyscale images) of a folder or .tar with labels.tsv."""
    path = Path(path)
    if path.is_dir():
        rows = [line.split("\t") for line in (path / "labels.tsv").read_text(encoding="utf-8").splitlines() if line]
        grays = [np.asarray(Image.open(path / "images" / name).convert("L")) for name, _ in rows]
        return [r[0] for r in rows], [r[1] for r in rows], grays
    with tarfile.open(path) as tar:
        members = {m.name.lstrip("./"): m for m in tar.getmembers() if m.isfile()}
        text = tar.extractfile(members["labels.tsv"]).read().decode("utf-8")
        rows = [line.split("\t") for line in text.splitlines() if line]
        grays = [np.asarray(Image.open(io.BytesIO(tar.extractfile(members["images/" + name]).read())).convert("L"))
                 for name, _ in rows]
    return [r[0] for r in rows], [r[1] for r in rows], grays


def set_info(path):
    """The config.json of a set written by scripts/render_words.py, or None (a real set)."""
    path = Path(path)
    if path.is_dir():
        f = path / "config.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lstrip("./") == "config.json":
                return json.loads(tar.extractfile(m).read().decode("utf-8"))
    return None


def kinds(info, names):
    """The kind of each word of a synthetic set as ``mayek_words.synth.Words.text`` drew it:
    'number', 'syllables' (composed of real syllables), 'scrambled' (random letters in a
    word's shape) or 'lexicon'. Item i is the file images/{i:06d}.png; its first random
    number decides."""
    out, scrambled = [], info.get("scrambled", 0.0)
    for name in names:
        r = np.random.default_rng([info["seed"], int(Path(name).stem)]).random()
        out.append("number" if r < info["numbers"] else
                   "syllables" if r < info["numbers"] + info["built"] else
                   "scrambled" if r < info["numbers"] + info["built"] + scrambled else "lexicon")
    return out


class FixedSet:
    def __init__(self, path, height=images.HEIGHT, max_width=images.MAX_WIDTH, limit=None):
        names, texts, grays = _read_set(path)
        if limit:
            names, texts, grays = names[:limit], texts[:limit], grays[:limit]
        self.path, self.names = str(path), names
        self.texts = [labels.clean(t)[0] for t in texts]
        self.grays = grays
        self.images = [images.normalise(g, height, max_width) for g in grays]
        self.info = set_info(path)
        self.kinds = kinds(self.info, names) if self.info else None

    def __len__(self):
        return len(self.texts)

    def batches(self, batch_size=128):
        """Batches in order of width (little padding); `index` gives each item's position."""
        order = np.argsort([im.shape[1] for im in self.images], kind="stable")
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            yield make_batch([(self.images[i], self.texts[i]) for i in idx], idx)
