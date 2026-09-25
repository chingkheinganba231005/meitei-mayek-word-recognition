"""Baseline: cut the word into characters, then classify each with the first paper's
ensemble. The cuts are perfect (the synthesiser's own boxes), so this is the best a
segment-then-classify recogniser built on the first paper could do on the same words.

    python scripts/oracle_baseline.py --model-dir work/weights --set work/synth/val.tar \\
        --glyphs work/glyphs/val.npz --lexicon work/lexicon/val.tsv --sizes results/glyph_sizes_tummhcd.json \\
        --lm work/lm/char_lm.pkl --tune --out results/phase2_baseline_val.json

    # test, once, with the language model's weights chosen on validation
    python scripts/oracle_baseline.py ... --set work/synth/test.tar --glyphs work/glyphs/test.npz \\
        --lexicon work/lexicon/test.tsv --tuned results/phase2_baseline_val.json --out results/phase2_baseline_test.json

The set's words are rendered again from its config.json (seed, shares and synthesiser
settings) to recover every character's box and image; the text must come out the same and
the image the same size (the script stops otherwise), and how many images are identical
is reported. Each character is classified in two forms:
- isolated: its original TUMMHCD image, as in the first paper (no neighbours, no resizing:
  an upper bound);
- cut: its box cut from the word image and stretched to 24 x 24 like a TUMMHCD image
  (strokes of neighbours that reach into the box stay).
ꯢ (044) and ꯏ (025) are one letter: their probabilities are added. Decoding: the most
probable class of each character; with zones, after Hijam and Saharia (2024): a character
written above the line can only be a sign written above (ꯥ ꯩ ꯪ), one below the line only
a sign written below (ꯨ, apun), a small sign raised beside its letter only ꯦ ꯣ ꯧ, and the
rest only what is written on the line (zones from the font's placement, mayek_words
priors). The zones are perfect: each character's own (their second stage estimates them
from the word image; here the boxes alone would place 1% of characters in the wrong zone,
as a raised ꯣ can sit higher than a low ꯥ). Each of these also with the character language
model, its weight chosen on validation. Hypotheses have as many characters as the truth,
so CER is the share of characters misread.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr import metrics  # noqa: E402
from mayek_htr.data import _read_set, kinds, set_info  # noqa: E402
from mayek_htr.decode import segments  # noqa: E402
from mayek_htr.lm import CharLM  # noqa: E402
from mayek_htr.model import find_model_dir  # noqa: E402
from mayek_words.charset import ALPHABET, CLASS_OF, I_LETTER, I_LONSUM, TUMMHCD  # noqa: E402
from mayek_words.glyphs import GlyphStore  # noqa: E402
from mayek_words.lexicon import Lexicon  # noqa: E402
from mayek_words.synth import MEASURED, Config, Words, WordSynth, load_priors  # noqa: E402

ZONES = ("upper", "raised", "middle", "lower")
SIDE = 24                                    # TUMMHCD images are 24 x 24


def zone(bottom, top):
    """Zone of a character placed from bottom to top (in letter heights above the baseline)."""
    c = (bottom + top) / 2
    return "upper" if c > 1.0 else "lower" if c < 0.0 else "raised" if bottom > 0.3 else "middle"


def zone_masks(prior):
    """({zone: boolean mask over ALPHABET of the characters written in it}, {character: zone}),
    from the font's placement in the priors."""
    of = {ch: zone(prior[ch]["bottom"], prior[ch]["top"]) for ch in ALPHABET}
    return {z: np.array([of[ch] == z for ch in ALPHABET]) for z in ZONES}, of


def merged(p):
    """(n, 55) TUMMHCD class probabilities -> (n, 54) over ALPHABET, ꯢ added to ꯏ."""
    keep = [CLASS_OF[ch] for ch in ALPHABET]
    out = p[:, keep].copy()
    out[:, ALPHABET.index(I_LETTER)] += p[:, CLASS_OF[I_LONSUM]]
    return out


def cut(image, box):
    """A character's box cut from the word image, stretched to SIDE x SIDE."""
    _, x0, y0, x1, y1 = box
    H, W = image.shape
    x0, x1 = int(np.clip(np.floor(x0), 0, W - 1)), int(np.clip(np.ceil(x1), 1, W))
    y0, y1 = int(np.clip(np.floor(y0), 0, H - 1)), int(np.clip(np.ceil(y1), 1, H))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize((SIDE, SIDE), Image.BILINEAR))


def rerender(info, glyphs, lexicon, sizes):
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in info["config"].items()})
    store = GlyphStore.load(glyphs)
    synth = WordSynth(store, load_priors(sizes=sizes), cfg)
    words = Words(synth, Lexicon.load(lexicon, info["alpha"], info["rare_share"]), info["seed"],
                  info["numbers"], info["stop"], info["built"])
    return words, store, synth.prior


def decode_all(logp, lm, alpha, spans):
    return ["" if a == b else segments(logp[a:b], lm, alpha) for a, b in spans]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="the first paper's model folder (or a folder above it)")
    ap.add_argument("--set", required=True, help="synthetic set rendered by scripts/render_words.py")
    ap.add_argument("--glyphs", required=True, help="the glyph store it was rendered with")
    ap.add_argument("--lexicon", required=True, help="the lexicon it was drawn from")
    ap.add_argument("--sizes", default="measured", help="as it was rendered with (scripts/render_words.py)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm", help="character language model")
    ap.add_argument("--tune", action="store_true", help="choose the language model's weight on this set")
    ap.add_argument("--tuned", help="results .json of the validation set, for its weights")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--no-tta", action="store_true", help="without the test-time views of the first paper")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--predictions", help="per-word .tsv")
    ap.add_argument("--device")
    args = ap.parse_args()
    import mayek.charset
    import torch
    from mayek.recognizer import Recognizer

    assert [c.char for c in mayek.charset.CLASSES] == list(TUMMHCD), "class order differs from the first project"
    info = set_info(args.set)
    if info is None:
        sys.exit(f"{args.set} has no config.json: not a set rendered by scripts/render_words.py")
    sizes = {"measured": MEASURED, "font": None}.get(args.sizes, args.sizes)
    for what, given, used in (("glyphs", Path(args.glyphs).name, info["glyphs"]),
                              ("lexicon", Path(args.lexicon).name, info["lexicon"]),
                              ("sizes", args.sizes if args.sizes in ("measured", "font") else Path(args.sizes).name,
                               info["sizes"])):
        if given != used:
            print(f"warning: --{what} is {given}, the set was rendered with {used}")
    names, texts, grays = _read_set(args.set)
    if args.limit:
        names, texts, grays = names[:args.limit], texts[:args.limit], grays[:args.limit]
    word_kinds = kinds(info, names)

    t0 = time.time()
    words, store, prior = rerender(info, args.glyphs, args.lexicon, sizes)
    masks, zone_of = zone_masks(prior)
    iso, cuts, zs, spans, refs, same, diffs = [], [], [], [], [], 0, []
    for name, text, gray in zip(names, texts, grays):
        s = words[int(Path(name).stem)]
        if s.text != text or s.image.shape != gray.shape:
            sys.exit(f"{name}: rendered again as {s.text!r} {s.image.shape}, stored {text!r} {gray.shape}: "
                     "the set cannot be reproduced with these files and this code")
        d = float(np.abs(s.image.astype(np.int16) - gray).mean())
        same += d == 0
        diffs.append(d)
        a = len(iso)
        for ch, g, box in zip(s.text, s.glyphs, s.boxes):
            iso.append(store.gray[g])
            cuts.append(cut(gray, box))
            zs.append(zone_of[ch])
        spans.append((a, len(iso)))
        refs.append(s.text)
    seconds = {"rendering": round(time.time() - t0, 1)}

    t0 = time.time()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    folder = find_model_dir(args.model_dir)
    rec = Recognizer(folder, device=device)
    probs = {}
    for form, images in (("isolated", iso), ("cut", cuts)):
        p = np.concatenate([rec.probs(images[k:k + args.batch], tta=not args.no_tta)
                            for k in range(0, len(images), args.batch)])
        probs[form] = merged(p)
    seconds["classifier"] = round(time.time() - t0, 1)

    zone_mask = np.stack([masks[z] for z in zs])
    logps = {}
    for form, p in probs.items():
        lp = np.log(np.maximum(p, 1e-12))
        logps[(form, "classifier")] = lp
        logps[(form, "zones")] = np.where(zone_mask, lp, -1e9)
    lm, chosen, grids = None, {}, {}
    if args.lm:
        lm = CharLM.load(args.lm)
        if args.tuned:
            chosen = {tuple(k.split("+")): v for k, v in
                      json.loads(Path(args.tuned).read_text(encoding="utf-8"))["lm"]["alpha"].items()}
    t0 = time.time()
    out_variants, hyps_all = {}, {}
    for (form, how), lp in logps.items():
        hyps = decode_all(lp, None, 0.0, spans)
        out_variants.setdefault(form, {})[how] = metrics.score(refs, hyps)
        hyps_all[f"{form} {how}"] = hyps
        if lm is None:
            continue
        if args.tune:
            grid = []
            for a in args.alphas:
                s = metrics.score(refs, decode_all(lp, lm, a, spans))
                grid.append({"alpha": a, "cer": s["cer"], "wer": s["wer"]})
            best = min(grid, key=lambda r: (r["cer"], r["wer"]))
            chosen[(form, how)] = best["alpha"]
            grids[f"{form}+{how}"] = grid
        a = chosen.get((form, how), 0.5)
        hyps = decode_all(lp, lm, a, spans)
        out_variants[form][how + "+lm"] = metrics.score(refs, hyps)
        hyps_all[f"{form} {how}+lm"] = hyps
    seconds["decoding"] = round(time.time() - t0, 1)

    n_chars = len(iso)
    out = {"set": Path(args.set).name, "words": len(refs), "characters": n_chars,
           "model": {"folder": folder.name, "members": [m["name"] for m in rec.config["members"]],
                     "tta": not args.no_tta},
           "reproduced": {"identical_images": int(same), "of": len(refs),
                          "largest_mean_difference_grey_levels": round(max(diffs), 3) if diffs else None},
           "zones": {"from": "each character's own zone (perfect zones)",
                     "characters_by_zone": {z: [ch for ch in ALPHABET if zone_of[ch] == z] for z in ZONES}},
           "note": ("CER = share of characters misread (as many characters as the truth); WER = share of words "
                    "with an error; +lm: with the character language model, weight alpha"),
           "variants": out_variants,
           "by_kind": {k: _by_kind(refs, h, word_kinds) for k, h in hyps_all.items()},
           "seconds": seconds}
    if lm is not None:
        out["lm"] = {"file": Path(args.lm).name, "order": lm.order,
                     "alpha": {f"{f}+{h}": a for (f, h), a in chosen.items()},
                     "chosen_on": (out["set"] if args.tune else Path(args.tuned).name if args.tuned else "default 0.5"),
                     "grid": grids or None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for form, v in out_variants.items():
        print(form + ": " + "; ".join(f"{how} CER {s['cer']:.4f} WER {s['wer']:.4f}" for how, s in v.items()))
    print(f"images identical when rendered again: {same} of {len(refs)}; written to {args.out}")
    if args.predictions:
        keys = list(hyps_all)
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).write_text(
            "file\treference\tkind\t" + "\t".join(keys) + "\n" + "".join(
                f"{n}\t{r}\t{k}\t" + "\t".join(hyps_all[key][i] for key in keys) + "\n"
                for i, (n, r, k) in enumerate(zip(names, refs, word_kinds))), encoding="utf-8")


def _by_kind(refs, hyps, word_kinds):
    out = {}
    for kind in sorted(set(word_kinds)):
        idx = [i for i, k in enumerate(word_kinds) if k == kind]
        s = metrics.score([refs[i] for i in idx], [hyps[i] for i in idx])
        out[kind] = {k: s[k] for k in ("words", "cer", "wer")}
    return out


if __name__ == "__main__":
    main()
