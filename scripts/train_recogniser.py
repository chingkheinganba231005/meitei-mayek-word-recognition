"""Train the word recogniser on synthetic words rendered on the fly (mayek_htr/train.py).

    python scripts/train_recogniser.py --glyphs work/glyphs/train.npz --lexicon work/lexicon/train.tsv \\
        --sizes results/glyph_sizes_tummhcd.json --val work/synth/val.tar \\
        --run-dir work/runs/convnext_tummhcd --init tummhcd --tummhcd-dir work/weights \\
        --results results/phase2_train_convnext_tummhcd.json [--steps 60000 --batch 64 ...]

Every field of mayek_htr.train.TrainConfig can be set (--encoder, --init, --steps, --lr,
--seed, --data-seed ...). Run again with the same --run-dir to go on after an interruption.
The training words are drawn as the fixed sets were (scripts/render_words.py: numbers 3%,
full stops 2%, count ** 0.5, rare characters 10%, words composed of syllables 15%), plus
--scrambled (10% by default) of lexicon words with their letters replaced by random ones of
the same kind, from the training lexicon and the training characters, with their own seed
(--data-seed).
The run folder gets best.pt (the weights with the lowest validation CER), final.pt,
last.pt (to resume) and history.json; --results gets a summary.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from mayek_htr.data import FixedSet  # noqa: E402
from mayek_htr.train import TrainConfig, load_checkpoint, train  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402
from mayek_words.lexicon import Lexicon  # noqa: E402
from mayek_words.synth import MEASURED, Words, WordSynth, load_priors  # noqa: E402


def config_args(ap):
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default,
                        help=f"default {f.default}")


def git_commit():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glyphs", required=True, help="training characters (glyph store .npz), or 'font'")
    ap.add_argument("--lexicon", required=True, help="training lexicon (word<TAB>count)")
    ap.add_argument("--sizes", default="measured", help="as scripts/render_words.py")
    ap.add_argument("--val", required=True, help="fixed validation set (folder or .tar)")
    ap.add_argument("--val-limit", type=int, help="use only the first n validation words (quick runs)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tummhcd-dir", help="the first paper's model folder, for --init tummhcd")
    ap.add_argument("--results", help="summary .json")
    ap.add_argument("--device")
    config_args(ap)
    args = ap.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})
    if cfg.init == "tummhcd" and cfg.encoder != "small_cnn" and not args.tummhcd_dir:
        ap.error("--init tummhcd needs --tummhcd-dir")

    store = GlyphStore.from_font() if args.glyphs == "font" else GlyphStore.load(args.glyphs)
    sizes = {"measured": MEASURED, "font": None}.get(args.sizes, args.sizes)
    words = Words(WordSynth(store, load_priors(sizes=sizes)), Lexicon.load(args.lexicon), seed=cfg.data_seed,
                  scrambled=cfg.scrambled)
    val = FixedSet(args.val, limit=args.val_limit)
    print(f"training characters: {len(store)} images; lexicon: {len(words.lexicon)} words; "
          f"validation: {len(val)} words; device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}",
          flush=True)

    t0 = time.time()
    history = train(cfg, words, val, args.run_dir, tummhcd_dir=args.tummhcd_dir, device=args.device,
                    log=lambda m: print(m, flush=True))
    if args.results:
        run = Path(args.run_dir)
        saved = json.loads((run / "history.json").read_text(encoding="utf-8"))
        model, best = load_checkpoint(run / "best.pt")
        summary = {"run": run.name, "cfg": dataclasses.asdict(cfg), "commit": git_commit(),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                   "training": {"glyphs": Path(args.glyphs).name, "lexicon": Path(args.lexicon).name,
                                "sizes": args.sizes if args.sizes in ("measured", "font") else Path(args.sizes).name,
                                "scrambled": cfg.scrambled, "words_seen": cfg.steps * cfg.batch},
                   "validation": {"set": Path(args.val).name, "words": len(val)},
                   "best": saved["best"], "val_at_best": best.get("val"),
                   "history": history,
                   "hours_this_session": round((time.time() - t0) / 3600, 2),
                   "note": "val CER and WER by greedy decoding of the EMA weights; best.pt is the step with the "
                           "lowest CER"}
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"summary written to {args.results}")


if __name__ == "__main__":
    main()
