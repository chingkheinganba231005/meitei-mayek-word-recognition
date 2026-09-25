"""Render a fixed, seeded set of synthetic word images, and a contact sheet to look at.

    python scripts/render_words.py --glyphs work/glyphs/val.npz --lexicon work/lexicon/val.tsv \\
        --n 5000 --seed 1 --out-dir work/synth/val [--sizes measured|font|file.json]

    python scripts/render_words.py --glyphs font --lexicon words.tsv --n 48 --sheet sheet.png

Writes out-dir/images/NNNNNN.png, out-dir/labels.tsv (file<TAB>text) and
out-dir/config.json (everything needed to render the same set again). --sheet writes
the first --sheet-n images with their text underneath. --glyphs font uses the font's own
characters instead of TUMMHCD (for a quick look without the dataset).
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words.glyphs import ASSETS, GlyphStore  # noqa: E402
from mayek_words.lexicon import Lexicon  # noqa: E402
from mayek_words.synth import MEASURED, Config, Words, WordSynth, line_gaps, load_priors  # noqa: E402


def contact_sheet(samples, path, cols=4, cell=(300, 120)):
    try:
        font = ImageFont.truetype(str(ASSETS / "NotoSansMeeteiMayek-Regular.ttf"), 20,
                                  layout_engine=ImageFont.Layout.RAQM)
    except (OSError, KeyError, ImportError):
        font = ImageFont.truetype(str(ASSETS / "NotoSansMeeteiMayek-Regular.ttf"), 20)
    latin = ImageFont.load_default()  # the Meitei Mayek font has no Latin digits
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("L", (cols * cell[0], rows * cell[1]), 255)
    draw = ImageDraw.Draw(sheet)
    for k, s in enumerate(samples):
        r, c = divmod(k, cols)
        im = Image.fromarray(s.image)
        scale = min((cell[0] - 10) / im.width, (cell[1] - 36) / im.height, 2.0)
        im = im.resize((max(int(im.width * scale), 1), max(int(im.height * scale), 1)), Image.BILINEAR)
        sheet.paste(im, (c * cell[0] + 5, r * cell[1] + 4))
        draw.text((c * cell[0] + 5, (r + 1) * cell[1] - 8), str(k), font=latin, fill=0, anchor="ls")
        draw.text((c * cell[0] + 35, (r + 1) * cell[1] - 8), s.text, font=font, fill=0, anchor="ls")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glyphs", required=True, help="glyph store .npz, or 'font'")
    ap.add_argument("--lexicon", required=True, help="word<TAB>count list (scripts/build_lexicon.py)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", help="write the images and labels here")
    ap.add_argument("--sizes", default="measured",
                    help="'measured' (on TUMMHCD, the default), 'font', or a glyph_sizes_*.json file")
    ap.add_argument("--numbers", type=float, default=0.03)
    ap.add_argument("--stop", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5, help="sampling weight = count ** alpha")
    ap.add_argument("--rare-share", type=float, default=0.1, help="share of draws for rare characters")
    ap.add_argument("--built", type=float, default=0.15,
                    help="share of words composed of real syllables (1-6 syllables, every kind)")
    ap.add_argument("--sheet", help="contact sheet .png")
    ap.add_argument("--sheet-n", type=int, default=48)
    args = ap.parse_args()

    store = GlyphStore.from_font() if args.glyphs == "font" else GlyphStore.load(args.glyphs)
    cfg = Config()
    sizes = {"measured": MEASURED, "font": None}.get(args.sizes, args.sizes)
    synth = WordSynth(store, load_priors(sizes=sizes), cfg)
    lexicon = Lexicon.load(args.lexicon, args.alpha, args.rare_share)
    words = Words(synth, lexicon, args.seed, args.numbers, args.stop, args.built)

    t0 = time.time()
    samples, shapes, lines, layouts = [], [], [], []
    out = Path(args.out_dir) if args.out_dir else None
    if out:
        (out / "images").mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        s = words[i]
        shapes.append(s.image.shape)
        layouts.append(s.layout)
        if i < args.sheet_n:
            samples.append(s)
        if out:
            name = f"{i:06d}.png"
            Image.fromarray(s.image).save(out / "images" / name)
            lines.append(f"{name}\t{s.text}\n")
    seconds = time.time() - t0
    shapes = np.array(shapes)
    summary = {"n": args.n, "seed": args.seed, "glyphs": Path(args.glyphs).name, "lexicon": Path(args.lexicon).name,
               "sizes": args.sizes if args.sizes in ("measured", "font") else Path(args.sizes).name,
               "numbers": args.numbers, "stop": args.stop,
               "alpha": args.alpha, "rare_share": lexicon.rare_share, "rare_characters": lexicon.rare_chars,
               "built": args.built,
               "config": dataclasses.asdict(cfg),
               "height_px": {"mean": round(float(shapes[:, 0].mean()), 1), "max": int(shapes[:, 0].max())},
               "width_px": {"mean": round(float(shapes[:, 1].mean()), 1), "max": int(shapes[:, 1].max())},
               "ms_per_word": round(1000 * seconds / max(args.n, 1), 2),
               "ink_gaps_in_L": {k: {"n": len(v), "p10": round(float(np.percentile(v, 10)), 3),
                                     "median": round(float(np.median(v)), 3),
                                     "p90": round(float(np.percentile(v, 90)), 3),
                                     "touching": round(float(np.mean(np.array(v) <= 0)), 3),
                                     "under_0.03": round(float(np.mean(np.array(v) < 0.03)), 3)}
                                 for k, v in line_gaps(layouts, synth.prior).items() if v}}
    if out:
        (out / "labels.tsv").write_text("".join(lines), encoding="utf-8")
        (out / "config.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.sheet:
        contact_sheet(samples, args.sheet)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
