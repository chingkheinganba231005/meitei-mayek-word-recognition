"""Training the recogniser (PyTorch): CTC on endless synthetic words; checkpoints chosen on
the fixed synthetic validation set by greedy CER.

An exponential moving average of the weights (EMA) is what gets evaluated and kept, as
in the first project. The learning rate warms up, then follows a cosine down to
lr * lr_min. Mixed precision (bfloat16) for the encoder on a GPU; the LSTM runs in float32.

Resumable, since Colab disconnects: the state (weights, EMA, optimiser, step, history)
is saved to local disk every `save_every` steps and copied to the run folder (e.g. on
Google Drive). Run again with the same folder to go on. After a resume the training
words go on from item step x batch of the stream (a few may be seen again).
"""

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import metrics
from .augment import PRESETS, augment
from .data import SynthStream
from .decode import greedy
from .model import MEAN, STD, build


@dataclass
class TrainConfig:
    encoder: str = "convnext_tiny"       # or "small_cnn"
    init: str = "tummhcd"                # "tummhcd", "imagenet" or "none"
    steps: int = 60000
    batch: int = 64
    lr: float = 3e-4
    lr_min: float = 0.01                 # the cosine ends at lr * lr_min
    warmup: int = 2000
    wd: float = 0.05
    ema: float = 0.999
    clip: float = 5.0
    aug: str = "full"                    # augment.PRESETS
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    eval_every: int = 2000
    save_every: int = 1000
    log_every: int = 200
    workers: int = 8
    pool: int = 8                        # batches rendered at once by a worker, grouped by width
    seed: int = 0                        # weights, augmentation
    data_seed: int = 1000                # the training words (the fixed sets use seeds 1 and 2)
    scrambled: float = 0.1               # share of training words with random letters of each kind (owner,
    #                                      26 September 2026; the first round had none)


class EMA:
    """Exponential moving average of the weights (buffers copied)."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay, self.updates = decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(m, 1 - d)
            else:
                e.copy_(m)


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def save(obj, path, tries=4):
    """torch.save made on local disk, then copied (mounted drives refuse writes now and then)."""
    path = Path(path)
    fd, local = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, local)
        for k in range(tries):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, path.with_name(path.name + ".tmp"))
                os.replace(path.with_name(path.name + ".tmp"), path)
                return
            except OSError:
                if k == tries - 1:
                    raise
                time.sleep(5 * (k + 1))
    finally:
        os.unlink(local)


def amp_dtype(device):
    return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None


def prepare(batch, device):
    x = batch["x"].to(device, non_blocking=True).float().div_(255)
    return x, batch["widths"].to(device, non_blocking=True)


@torch.no_grad()
def predict(model, fixed_set, device, batch_size=128):
    """-> per word: natural log probabilities (T, classes) as float16 numpy arrays."""
    model.eval()
    amp = amp_dtype(device)
    out = [None] * len(fixed_set)
    for b in fixed_set.batches(batch_size):
        x, widths = prepare(b, device)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            logits, lengths = model((x - MEAN) / STD, widths)
        lp = logits.float().log_softmax(-1).cpu()
        for j, i in enumerate(b["index"].tolist()):
            out[i] = lp[j, :int(lengths[j])].numpy().astype(np.float16)
    return out


def evaluate(model, fixed_set, device, batch_size=128):
    """Greedy decoding -> (scores, hypotheses, log probabilities)."""
    logps = predict(model, fixed_set, device, batch_size)
    hyps = [greedy(lp.astype(np.float32)) for lp in logps]
    return metrics.score(fixed_set.texts, hyps), hyps, logps


def load_checkpoint(path, device="cpu"):
    """A saved model (best.pt / final.pt: EMA weights and config) -> model in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    model = build(cfg.encoder, init="none", hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def train(cfg, words, val_set, run_dir, tummhcd_dir=None, device=None, log=print):
    """Train (or resume) into run_dir; returns the history. words: mayek_words.synth.Words
    over the training characters and lexicon, made with seed cfg.data_seed and share
    cfg.scrambled of scrambled words."""
    if words.seed != cfg.data_seed or abs(getattr(words, "scrambled", 0.0) - cfg.scrambled) > 1e-12:
        raise ValueError(f"the training words (seed {words.seed}, scrambled {getattr(words, 'scrambled', 0.0)}) "
                         f"do not match the settings (data_seed {cfg.data_seed}, scrambled {cfg.scrambled})")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:10]    # one local copy per run folder
    local = Path(tempfile.gettempdir()) / "mayek_htr" / f"{run_dir.name}-{key}" / "last.pt"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build(cfg.encoder, init=cfg.init, tummhcd_dir=tummhcd_dir, hidden=cfg.hidden, layers=cfg.layers,
                  dropout=cfg.dropout).to(device)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(param_groups(model, cfg.wd), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    amp = amp_dtype(device)
    step, history, best = 0, [], {"cer": math.inf}

    ck = None
    if not (run_dir / "last.pt").exists() and local.exists():  # the run folder is new or was emptied:
        local.unlink()                                         # start afresh, not from an old local copy
    for p in (run_dir / "last.pt", local):
        if p.exists():
            try:
                c = torch.load(p, map_location=device, weights_only=False)
            except Exception as e:  # a half-written file after a crash
                log(f"ignoring unreadable checkpoint {p} ({type(e).__name__})")
                continue
            if ck is None or c["step"] > ck["step"]:
                ck = c
    if ck is not None:
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        ema.updates, step, history, best = ck["ema_updates"], ck["step"], ck["history"], ck["best"]
        log(f"resuming at step {step}")
        now = asdict(cfg)
        changed = {k: (ck["cfg"].get(k), now.get(k)) for k in set(ck["cfg"]) | set(now)
                   if ck["cfg"].get(k) != now.get(k)}
        if changed:
            log(f"settings changed since the checkpoint (then, now): {changed}")

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        t = (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (cfg.lr_min + (1 - cfg.lr_min) * 0.5 * (1 + math.cos(math.pi * t)))

    if step < cfg.steps:
        stream = SynthStream(words, cfg.batch, cfg.pool, start=step * cfg.batch)
        loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=cfg.workers,
                                             pin_memory=device == "cuda", persistent_workers=cfg.workers > 0,
                                             prefetch_factor=4 if cfg.workers else None)
        batches = iter(loader)
        aug = PRESETS[cfg.aug]
        t0, waited, seen = time.time(), 0.0, 0
        running, count, recent = 0.0, 0, []         # the loss since the last validation; the last log_every
        model.train()
        while step < cfg.steps:
            tw = time.time()
            b = next(batches)
            waited += time.time() - tw
            x, widths = prepare(b, device)
            x = (augment(x, widths, aug) - MEAN) / STD
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits, lengths = model(x, widths)
            logp = logits.float().log_softmax(-1).transpose(0, 1)
            loss = ctc(logp, b["targets"].to(device), lengths, b["target_lengths"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            ema.update(model)
            step += 1
            seen += len(b["texts"])
            running += loss.item()
            count += 1
            recent.append(loss.item())

            if step % cfg.log_every == 0:
                dt = time.time() - t0
                log(f"step {step}  loss {np.mean(recent):.4f}  lr {lr_at(step):.2e}  "
                    f"{seen / dt:.0f} words/s  waiting for data {100 * waited / dt:.0f}%")
                recent = []
            if step % cfg.eval_every == 0 or step == cfg.steps:
                res, _, _ = evaluate(ema.module, val_set, device)
                model.train()
                row = {"step": step, "loss": round(running / max(count, 1), 5), "lr": lr_at(step - 1),
                       "val_cer": res["cer"], "val_wer": res["wer"],
                       "words_per_s": round(seen / (time.time() - t0), 1),
                       "waiting_for_data": round(waited / (time.time() - t0), 3)}
                history.append(row)
                log(f"validation at step {step}: CER {res['cer']:.4f}  WER {res['wer']:.4f}")
                if res["cer"] < best["cer"]:
                    best = {"cer": res["cer"], "wer": res["wer"], "step": step}
                    save({"model": ema.module.state_dict(), "cfg": asdict(cfg), "step": step, "val": res},
                         run_dir / "best.pt")
                running, count = 0.0, 0
            if step % cfg.save_every == 0 or step == cfg.steps:
                state = {"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                         "ema_updates": ema.updates, "step": step, "history": history, "best": best,
                         "cfg": asdict(cfg)}
                save(state, local)
                try:
                    shutil.copyfile(local, run_dir / "last.pt.tmp")
                    os.replace(run_dir / "last.pt.tmp", run_dir / "last.pt")
                except OSError as e:
                    log(f"could not copy the checkpoint to {run_dir} ({e}); training goes on")

    save({"model": ema.module.state_dict(), "cfg": asdict(cfg), "step": step}, run_dir / "final.pt")
    (run_dir / "history.json").write_text(json.dumps({"cfg": asdict(cfg), "best": best, "history": history},
                                                     indent=1), encoding="utf-8")
    return history
