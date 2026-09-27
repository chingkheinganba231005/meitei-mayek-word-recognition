"""Cut the handwritten words out of written pages into a word set, in the form of
the synthetic sets (images/NNNNNN.png and labels.tsv, read by mayek_htr.data.FixedSet),
with manifest.json in place of config.json, and contact sheets to check every word.

    python scripts/cut_writing_pages.py --template work/real/real_words_pages.pdf \\
        --written work/real/written.pdf --out-dir work/real/real_test --tar \\
        --sheets work/real/sheets --summary results/phase3_real_set.json

--written: the pages as exported from the iPad (PDFs, or page images), in any order; a
page given twice counts once (the later one). A word's image is named after its item
number. manifest.json lists every item with its text, kind and page, whether it was
written, and its flags: 'outside' (part of the word lies outside its box: check it is
whole), 'shared' (a stroke runs into another box). Handwriting away from every box is
listed per page as stray. The contact sheets (one per page) show every item's printed
word beside its cut-out handwriting: a green frame is a written word, orange a flagged
one, and an empty box is marked. --summary writes counts only (no words).
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mayek_htr.pages import Cutter, meitei_font, read_pages  # noqa: E402
from mayek_words.charset import ALPHABET  # noqa: E402

TILE_W, TILE_H, CROP_H = 800, 124, 84


def sheet(page_no, items, crops, path):
    """The contact sheet of one page: two columns of tiles, the printed word beside the
    handwriting cut out for it."""
    rows = -(-len(items) // 2)
    im = Image.new("RGB", (2 * TILE_W, 50 + rows * TILE_H), "white")
    d = ImageDraw.Draw(im)
    mm, small = meitei_font(34), ImageFont.load_default(20)
    d.text((10, 14), f"page {page_no}: printed word | handwriting cut out (green: written, orange: flagged)",
           font=small, fill=(40, 40, 40))
    for k, it in enumerate(items):
        x, y = (k % 2) * TILE_W, 50 + (k // 2) * TILE_H
        d.text((x + 10, y + 30), str(it["item"]), font=small, fill=(90, 90, 90))
        d.text((x + 60, y + 50), it["text"], font=mm, fill="black", anchor="ls")
        note = it["status"] if it["status"] != "written" else " ".join(it["flags"])
        if note:
            d.text((x + 10, y + 70), note, font=small, fill=(200, 90, 0) if it["flags"] else (120, 120, 120))
        crop = crops.get(it["item"])
        if crop is None:
            continue
        g = Image.fromarray(crop)
        scale = min(CROP_H / g.height, (TILE_W - 290) / g.width)
        g = g.resize((max(int(g.width * scale), 1), max(int(g.height * scale), 1)), Image.LANCZOS)
        cx, cy = x + 270, y + (TILE_H - g.height) // 2
        im.paste(g.convert("RGB"), (cx, cy))
        d.rectangle((cx - 3, cy - 3, cx + g.width + 2, cy + g.height + 2),
                    outline=(230, 140, 0) if it["flags"] else (60, 170, 60), width=2)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="the blank template .pdf the pages were written on")
    ap.add_argument("--written", required=True, nargs="+", help="written pages: .pdf files or page images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tar", action="store_true", help="also write <out-dir>.tar")
    ap.add_argument("--sheets", help="folder for the contact sheets")
    ap.add_argument("--summary", help="counts .json")
    args = ap.parse_args()

    cutter = Cutter(args.template)
    man = cutter.manifest
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    results, pages = {}, []
    for name, rgb in read_pages(args.written):
        r = cutter.cut(rgb)
        pages.append({"source": Path(name).name, "page": r["page"], "corner_squares": r["marks"],
                      "printed_ink_missing": r["lack"], "stray": [list(map(int, s)) for s in r["stray"]]})
        if r["page"] is None:
            print(f"{name}: not a page of this template (printed ink missing: {r['lack']:.1%})")
            continue
        if r["page"] in results:
            print(f"{name}: page {r['page']} given again; this one is used")
        results[r["page"]] = r
        print(f"{name}: page {r['page']}" + ("" if r["marks"] else ", corner squares not found (scaled only)") +
              (f", {len(r['stray'])} stray pieces of handwriting" if r["stray"] else ""))

    items, crops, labels = [], {}, []
    for it in man["items"]:
        row = {k: it[k] for k in ("item", "page", "text", "kind")}
        r = results.get(it["page"])
        if r is None:
            row.update(status="page not given", flags=[], file=None)
        else:
            crop, flags = r["items"][it["item"]]
            if crop is None:
                row.update(status="empty", flags=[], file=None)
            else:
                name = f"{it['item']:06d}.png"
                Image.fromarray(crop).save(out / "images" / name)
                crops[it["item"]] = crop
                labels.append(f"{name}\t{it['text']}\n")
                row.update(status="written", flags=flags, file=name, size=list(crop.shape))
        items.append(row)
    (out / "labels.tsv").write_text("".join(labels), encoding="utf-8")
    manifest = {"format": "mayek-real-words/1", "template": Path(args.template).name, "title": man["title"],
                "lexicon": man.get("lexicon"), "lexicon_sha1": man.get("lexicon_sha1"),
                "selection": man.get("selection"), "pages": pages, "items": items}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.tar:
        shutil.make_archive(str(out), "tar", out)
    if args.sheets:
        Path(args.sheets).mkdir(parents=True, exist_ok=True)
        for p in sorted(results):
            sheet(p, [it for it in items if it["page"] == p], crops, Path(args.sheets) / f"page_{p:02d}.png")

    status = Counter(it["status"] for it in items)
    written = [it for it in items if it["status"] == "written"]
    summary = {"template": manifest["template"], "title": man["title"], "pages": man["pages"],
               "pages_given": len(results), "pages_missing": [p for p in range(1, man["pages"] + 1) if p not in results],
               "items": len(items), "status": dict(status),
               "written_by_kind": dict(Counter(it["kind"] for it in written)),
               "characters_written": sum(len(it["text"]) for it in written),
               "character_counts": {ch: n for ch, n in sorted(Counter(ch for it in written for ch in it["text"]).items(),
                                                             key=lambda kv: ALPHABET.index(kv[0]))},
               "flagged": dict(Counter(f for it in written for f in it["flags"])),
               "stray_pieces": sum(len(p["stray"]) for p in pages),
               "image_px": {"median_height": int(np.median([it["size"][0] for it in written])) if written else None,
                            "median_width": int(np.median([it["size"][1] for it in written])) if written else None}}
    print(f"{out}: {status.get('written', 0)} words written, {status.get('empty', 0)} empty boxes, "
          f"{status.get('page not given', 0)} on pages not given; flagged {summary['flagged'] or 'none'}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
