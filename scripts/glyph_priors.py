"""Size and position of every TUMMHCD character in print: the prior for composing words.

    python scripts/glyph_priors.py [font.ttf] \\
        --out mayek_words/assets/glyph_priors.json --glyphs mayek_words/assets/font_glyphs.npz

TUMMHCD images are 24 x 24 with the ink stretched to fill the frame (Phase 0), so a
character's size, proportions and place in the word are lost. This script takes them
from a font. Each character is shaped with HarfBuzz, alone, or (the vowel signs, nung and
apun) after each of the 27 letters, and its ink box is measured in units of L, the height
of a letter: y points up, the baseline is at 0 and the top of a letter at 1.

- Letters, lonsum letters, digits and cheikhei ("base"): width, bottom, top, and the side
  bearings, so that the pen moves by lsb + width + rsb.
- Signs ("mark"): width, bottom, top, the left edge relative to the pen position after the
  preceding character ("off"), and how far the sign moves the pen ("adv"; 0 for signs
  drawn above or below). Medians over the 27 letters. Apun is drawn under the whole
  letter before it ("span").

--glyphs also writes each character as a TUMMHCD-like image (the ink box of the printed
glyph stretched to 24 x 24), for tests and for previews without the dataset.

The font, by default the one in mayek_words/assets: Noto Sans Meetei Mayek Regular 2.002
(SIL Open Font Licence 1.1, assets/OFL.txt), built with fontmake from
github.com/notofonts/meetei-mayek at commit e562454, unmodified. Needs uharfbuzz and
fonttools (pip install uharfbuzz fonttools).
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_words.charset import APUN, LETTERS, MARKS, TUMMHCD  # noqa: E402

SIDE = 24


def shaper(font_path):
    import uharfbuzz as hb

    face = hb.Face(hb.Blob.from_file_path(str(font_path)))
    font = hb.Font(face)

    def shape(text):
        """-> list of glyph dicts (ink box, pen position before the glyph, advance), in font units."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf)
        pen, out = 0, []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            e = font.get_glyph_extents(info.codepoint)
            x0 = pen + pos.x_offset + e.x_bearing
            top = pos.y_offset + e.y_bearing
            out.append({"x0": x0, "x1": x0 + e.width, "top": top, "bottom": top + e.height,
                        "pen": pen, "adv": pos.x_advance})
            pen += pos.x_advance
        return out

    return shape


def measure(font_path):
    shape = shaper(font_path)
    letters = sorted(LETTERS)
    alone = {ch: shape(ch)[0] for ch in letters}
    L = float(np.median([g["top"] for g in alone.values()]) - np.median([g["bottom"] for g in alone.values()]))
    rows = []
    for index, ch in enumerate(TUMMHCD):
        row = {"class": index, "char": ch, "name": unicodedata.name(ch)}
        if ch not in MARKS:
            g = shape(ch)[0]
            row.update(kind="base", w=(g["x1"] - g["x0"]) / L, bottom=g["bottom"] / L, top=g["top"] / L,
                       lsb=g["x0"] / L, rsb=(g["adv"] - g["x1"]) / L)
        else:
            m = []
            for base in letters:
                glyphs = shape(base + ch)
                b = glyphs[0]
                # the sign's own glyph: for apun the font draws an underline in parts; take their union
                parts = glyphs[1:]
                x0, x1 = min(p["x0"] for p in parts), max(p["x1"] for p in parts)
                bottom, top = min(p["bottom"] for p in parts), max(p["top"] for p in parts)
                pen = b["pen"] + b["adv"]
                m.append([x1 - x0, bottom, top, x0 - pen, sum(p["adv"] for p in parts), x0 - b["x0"], x1 - b["x1"]])
            m = np.array(m) / L
            med = np.median(m, axis=0)
            row.update(kind="mark", w=med[0], bottom=med[1], top=med[2], off=med[3], adv=med[4],
                       off_spread=float(m[:, 3].std()))
            if ch == APUN:
                row.update(span=True, span_left=med[5], span_right=med[6])
            row["place"] = "above" if med[1] >= 0.9 else "below" if med[2] <= 0.1 else "beside"
        rows.append({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    return L, rows


def font_info(font_path):
    from fontTools.ttLib import TTFont

    names = TTFont(str(font_path))["name"]
    return {"family": names.getDebugName(4), "version": names.getDebugName(5),
            "licence": "SIL Open Font Licence 1.1"}


def font_glyph_images(font_path, rows, px_per_L=96):
    """Each character as TUMMHCD stores it: the ink box stretched to 24 x 24, dark ink on white."""
    size = int(round(px_per_L / 0.663))  # font size in px for a letter height of about px_per_L
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.BASIC)  # no dotted circles
    images = np.zeros((len(rows), SIDE, SIDE), np.uint8)
    for r in rows:
        canvas = Image.new("L", (6 * size, 4 * size), 0)
        ImageDraw.Draw(canvas).text((2 * size, 2 * size), r["char"], font=font, fill=255, anchor="ls")
        ink = np.asarray(canvas)
        ys, xs = np.nonzero(ink > 127)
        crop = Image.fromarray(ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        images[r["class"]] = 255 - np.asarray(crop.resize((SIDE, SIDE), Image.BOX))
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("font", nargs="?", default=str(Path(__file__).resolve().parents[1] / "mayek_words"
                                                    / "assets" / "NotoSansMeeteiMayek-Regular.ttf"))
    ap.add_argument("--out", default="mayek_words/assets/glyph_priors.json")
    ap.add_argument("--glyphs", help="also write the characters as 24 x 24 images (.npz)")
    ap.add_argument("--source", default="github.com/notofonts/meetei-mayek, commit e562454, built with fontmake",
                    help="where the font came from, recorded in the output")
    args = ap.parse_args()

    L, rows = measure(args.font)
    out = {"font": {**font_info(args.font), "source": args.source, "file": Path(args.font).name},
           "unit": f"L = letter height (median over the 27 letters) = {L:g} font units; y up from the baseline",
           "classes": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} classes written to {args.out}")
    if args.glyphs:
        images = font_glyph_images(args.font, rows)
        np.savez_compressed(args.glyphs, images=images, labels=np.arange(len(rows)),
                            note=np.array("characters of " + out["font"]["family"] + ", stretched to 24 x 24"))
        print(f"glyph images written to {args.glyphs}")


if __name__ == "__main__":
    main()
