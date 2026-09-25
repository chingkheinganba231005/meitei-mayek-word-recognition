"""Evaluate a trained recogniser on a set of word images: greedy decoding, and CTC beam
search with the character language model.

    # validation: choose the language model's weight (alpha) and the bonus per character (beta)
    python scripts/eval_recogniser.py --checkpoint work/runs/convnext_tummhcd/best.pt \\
        --set work/synth/val.tar --lm work/lm/char_lm.pkl --tune \\
        --out results/phase2_val_convnext_tummhcd.json \\
        --predictions work/runs/convnext_tummhcd/val_predictions.tsv --sheet work/runs/convnext_tummhcd/val_errors.png

    # test, once, with alpha and beta as chosen on validation
    python scripts/eval_recogniser.py --checkpoint work/runs/convnext_tummhcd/best.pt \\
        --set work/synth/test.tar --lm work/lm/char_lm.pkl --tuned results/phase2_val_convnext_tummhcd.json \\
        --out results/phase2_test_convnext_tummhcd.json

Scores (mayek_htr/metrics.py): CER, WER, word accuracy with its 95% interval, the confusable
pairs, the commonest errors; for a synthetic set also by kind of word (lexicon words, words
composed of syllables, numbers). --tune tries every alpha and beta of the grid on the set
and keeps the pair with the lowest CER (then WER). --predictions writes every word
(file, reference, kind, greedy, with the language model); --sheet draws the first 48
misread words.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from mayek_htr import metrics  # noqa: E402
from mayek_htr.data import FixedSet  # noqa: E402
from mayek_htr.decode import beam_search, greedy  # noqa: E402
from mayek_htr.lm import CharLM  # noqa: E402
from mayek_htr.train import load_checkpoint, predict  # noqa: E402
from mayek_words.glyphs import ASSETS  # noqa: E402

LM, LOGPS = None, None          # shared with the forked decoding processes


def _decode(job):
    alpha, beta, beam, idx = job
    return [beam_search(LOGPS[i].astype(np.float32), LM, alpha, beta, beam=beam) for i in idx]


def decode_grid(settings, idx, beam, processes):
    """{(alpha, beta): hypotheses for the words idx}, the words decoded in parallel."""
    chunks = np.array_split(np.asarray(idx), max(1, len(idx) // 100))
    jobs = [(a, b, beam, c) for a, b in settings for c in chunks]
    if processes > 1:
        with mp.get_context("fork").Pool(processes) as pool:
            parts = pool.map(_decode, jobs, chunksize=1)
    else:
        parts = [_decode(j) for j in jobs]
    out, k = {}, 0
    for s in settings:
        out[s] = [h for part in parts[k:k + len(chunks)] for h in part]
        k += len(chunks)
    return out


def by_kind(refs, hyps, kinds):
    if kinds is None:
        return None
    out = {}
    for kind in sorted(set(kinds)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        s = metrics.score([refs[i] for i in idx], [hyps[i] for i in idx])
        out[kind] = {k: s[k] for k in ("words", "characters", "cer", "wer", "word_accuracy", "word_accuracy_95ci")}
    return out


def meitei_font(size):
    path = str(ASSETS / "NotoSansMeeteiMayek-Regular.ttf")
    try:
        return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    except (OSError, KeyError, ImportError):
        return ImageFont.truetype(path, size)


def error_sheet(grays, refs, hyps, path, cols=4, cell=(320, 150)):
    """Misread words: the image, the reference (black) and what was read (below it)."""
    font, latin = meitei_font(20), ImageFont.load_default()
    rows = max((len(grays) + cols - 1) // cols, 1)
    sheet = Image.new("L", (cols * cell[0], rows * cell[1]), 255)
    draw = ImageDraw.Draw(sheet)
    for k, (g, r, h) in enumerate(zip(grays, refs, hyps)):
        y, x = divmod(k, cols)
        im = Image.fromarray(g)
        scale = min((cell[0] - 10) / im.width, (cell[1] - 62) / im.height, 2.0)
        im = im.resize((max(int(im.width * scale), 1), max(int(im.height * scale), 1)), Image.BILINEAR)
        sheet.paste(im, (x * cell[0] + 5, y * cell[1] + 4))
        draw.text((x * cell[0] + 5, (y + 1) * cell[1] - 34), "ref", font=latin, fill=0, anchor="ls")
        draw.text((x * cell[0] + 35, (y + 1) * cell[1] - 34), r, font=font, fill=0, anchor="ls")
        draw.text((x * cell[0] + 5, (y + 1) * cell[1] - 8), "read", font=latin, fill=0, anchor="ls")
        draw.text((x * cell[0] + 35, (y + 1) * cell[1] - 8), h or " ", font=font, fill=0, anchor="ls")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    global LM, LOGPS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best.pt or final.pt of a run")
    ap.add_argument("--set", required=True, help="fixed set of word images (folder or .tar)")
    ap.add_argument("--limit", type=int, help="only the first n words")
    ap.add_argument("--out", required=True, help="results .json")
    ap.add_argument("--lm", help="character language model (scripts/build_char_lm.py)")
    ap.add_argument("--tune", action="store_true", help="choose alpha and beta on this set")
    ap.add_argument("--tune-n", type=int, help="tune on the first n words only")
    ap.add_argument("--tuned", help="results .json of the validation set, for its alpha and beta")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--predictions", help="per-word .tsv")
    ap.add_argument("--sheet", help="misread words .png")
    ap.add_argument("--device")
    args = ap.parse_args()
    if args.tune and args.tuned:
        ap.error("--tune or --tuned, not both")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, ck = load_checkpoint(args.checkpoint, device)
    data = FixedSet(args.set, limit=args.limit)
    LOGPS = predict(model, data, device)
    seconds = {"network": round(time.time() - t0, 1)}
    refs = data.texts
    hyps = [greedy(lp.astype(np.float32)) for lp in LOGPS]
    out = {"checkpoint": str(Path(args.checkpoint).parent.name) + "/" + Path(args.checkpoint).name,
           "step": ck.get("step"), "cfg": ck.get("cfg"), "set": Path(args.set).name, "words": len(data),
           "set_info": {k: data.info.get(k) for k in ("n", "seed", "glyphs", "lexicon", "numbers", "stop", "built")}
           if data.info else None,
           "greedy": metrics.score(refs, hyps), "greedy_by_kind": by_kind(refs, hyps, data.kinds)}
    final = hyps

    if args.lm:
        LM = CharLM.load(args.lm)
        t1 = time.time()
        alpha, beta, beam, grid = args.alpha, args.beta, args.beam, None
        if args.tuned:
            chosen = json.loads(Path(args.tuned).read_text(encoding="utf-8"))["lm"]
            alpha, beta, beam = chosen["alpha"], chosen["beta"], chosen["beam"]
            if chosen["file"] != Path(args.lm).name:
                print(f"note: tuned with {chosen['file']}, now {Path(args.lm).name}")
        if args.tune:
            idx = list(range(min(args.tune_n or len(data), len(data))))
            settings = [(0.0, 0.0)] + [(a, b) for a in args.alphas for b in args.betas]
            found = decode_grid(settings, idx, beam, args.processes)
            grid = []
            for (a, b), h in found.items():
                s = metrics.score([refs[i] for i in idx], h)
                grid.append({"alpha": a, "beta": b, "cer": s["cer"], "wer": s["wer"]})
            best = min(grid, key=lambda r: (r["cer"], r["wer"]))
            alpha, beta = best["alpha"], best["beta"]
            print(f"chosen on {len(idx)} words: alpha {alpha}, beta {beta} (CER {best['cer']}, WER {best['wer']}); "
                  f"without the language model: CER {grid[0]['cer']}, WER {grid[0]['wer']}")
        final = decode_grid([(alpha, beta)], range(len(data)), beam, args.processes)[(alpha, beta)]
        out["lm"] = {"file": Path(args.lm).name, "order": LM.order, "alpha": alpha, "beta": beta, "beam": beam,
                     "chosen_on": (f"{out['set']} ({len(idx)} words)" if args.tune else
                                   Path(args.tuned).name if args.tuned else "given"),
                     "grid": grid}
        out["with_lm"] = metrics.score(refs, final)
        out["with_lm_by_kind"] = by_kind(refs, final, data.kinds)
        seconds["decoding"] = round(time.time() - t1, 1)
    out["seconds"] = seconds

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    g = out["greedy"]
    print(f"{out['set']}: {len(data)} words; greedy CER {g['cer']:.4f} WER {g['wer']:.4f}", end="")
    if args.lm:
        w = out["with_lm"]
        print(f"; with the language model CER {w['cer']:.4f} WER {w['wer']:.4f}", end="")
    print(f"\nwritten to {args.out}")

    kinds = data.kinds or [""] * len(data)
    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).write_text(
            "file\treference\tkind\tgreedy\twith_lm\n" + "".join(
                f"{n}\t{r}\t{k}\t{h}\t{f if args.lm else ''}\n"
                for n, r, k, h, f in zip(data.names, refs, kinds, hyps, final)), encoding="utf-8")
    if args.sheet:
        wrong = [i for i in range(len(data)) if final[i] != refs[i]][:48]
        error_sheet([data.grays[i] for i in wrong], [refs[i] for i in wrong], [final[i] for i in wrong], args.sheet)


if __name__ == "__main__":
    main()
