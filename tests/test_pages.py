import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, features

pytest.importorskip("pymupdf")
pytestmark = pytest.mark.skipif(not features.check("raqm"), reason="printing Meitei Mayek needs Pillow with RAQM")

from mayek_htr import pages  # noqa: E402
from mayek_words.charset import CHEIKHEI, DIGITS, LETTERS, LONSUM, I_LONSUM, VOWEL_SIGNS, problem  # noqa: E402
from mayek_words.lexicon import split_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fake_counts(n=6000, seed=0):
    """Random words of letters, vowel signs and lonsum letters, with counts."""
    rng = np.random.default_rng(seed)
    letters, signs, finals = sorted(LETTERS), sorted(VOWEL_SIGNS), sorted(LONSUM - {I_LONSUM})
    counts = {}
    while len(counts) < n:
        w = ""
        for _ in range(int(rng.integers(1, 5))):
            w += letters[int(rng.integers(len(letters)))]
            if rng.random() < 0.5:
                w += signs[int(rng.integers(len(signs)))]
            if rng.random() < 0.2:
                w += finals[int(rng.integers(len(finals)))]
        if problem(w) is None:
            counts[w] = int(rng.integers(1, 500))
    return counts


def test_choose_items():
    counts = fake_counts()
    items = pages.choose_items(counts, 300, np.random.default_rng(1), cover=4)
    again = pages.choose_items(counts, 300, np.random.default_rng(1), cover=4)
    assert items == again and len(items) == 300
    kinds = Counter(it["kind"] for it in items)
    assert kinds["number"] == 15
    words = [it["text"].rstrip(CHEIKHEI) for it in items if it["kind"] == "lexicon"]
    assert len(set(words)) == len(words)                                   # distinct
    assert all(split_of(w) == "test" and w in counts and 2 <= len(w) <= 14 for w in words)
    assert all(set(it["text"]) <= DIGITS for it in items if it["kind"] == "number")
    have = Counter(ch for it in items for ch in it["text"])
    pool = Counter(ch for w in counts if split_of(w) == "test" for ch in w)
    assert all(have[ch] >= min(4, pool[ch]) for ch in pool)                # every letter and sign covered
    assert all(have[d] >= 2 for d in DIGITS)
    trial = pages.choose_items(counts, 20, np.random.default_rng(1), split="val")
    assert all(split_of(it["text"].rstrip(CHEIKHEI)) == "val" for it in trial if it["kind"] == "lexicon")


def test_layout():
    counts = fake_counts()
    items = pages.choose_items(counts, 120, np.random.default_rng(2))
    man = pages.make_manifest(items, "Test")
    assert [it["item"] for it in man["items"]] == list(range(1, 121))
    assert Counter(it["text"] for it in man["items"]) == Counter(it["text"] for it in items)
    order = [(it["page"], it["box"][1], it["box"][0]) for it in man["items"]]
    assert order == sorted(order)                                           # numbered in reading order
    for p in range(1, man["pages"] + 1):
        boxes = [it["box"] for it in man["items"] if it["page"] == p]
        for x0, y0, x1, y1 in boxes:
            assert pages.MARGIN <= x0 < x1 <= pages.PAGE[0] - pages.MARGIN and pages.TOP < y0 < y1 < pages.BOTTOM
        for a in boxes:                                                     # boxes and their prompts apart
            for b in boxes:
                if a is not b:
                    assert a[2] <= b[0] or b[2] <= a[0] or a[3] + pages.ROW_GAP <= b[1] - pages.PROMPT_H or \
                        b[3] + pages.ROW_GAP <= a[1] - pages.PROMPT_H


def test_homography_and_marks(tmp_path):
    src = [(0, 0), (10, 0), (0, 10), (10, 10)]
    dst = [(3, 4), (25, 5), (1, 30), (27, 33)]
    H = pages.homography(src, dst)
    for (x, y), (u, v) in zip(src, dst):
        q = H @ [x, y, 1]
        assert np.allclose(q[:2] / q[2], (u, v))
    man = pages.make_manifest([{"text": chr(0xABC0) + chr(0xABE5), "kind": "lexicon"}], "Test")
    g = np.asarray(pages.draw_page(man, 1).convert("L"), np.float32)
    assert np.allclose(pages.find_marks(g), pages.MARK_CENTRES, atol=0.5)
    dim = g * np.linspace(0.5, 0.9, g.shape[1])[None, :]                    # a photo in uneven light
    flat = pages.flatten(dim)
    assert np.percentile(flat, 50) > 240 and flat[g < 50].mean() < 60
    assert pages.flatten(g) is g                                            # a white page as it is


def write_on(page, it, text, rng, color=(0, 0, 0), dy=None):
    """Handwriting stand-in: the text in the Meitei font, letters 60-70 px high, in the box."""
    font = pages.meitei_font(int(rng.uniform(60, 70) / pages.L_PER_EM))
    x0, y0, x1, y1 = it["box"]
    layer = Image.new("L", pages.PAGE, 0)
    d = ImageDraw.Draw(layer)
    y = (y0 + y1) / 2 + 25 if dy is None else y1 + dy
    d.text((x0 + 30 + rng.uniform(0, 30), y), text, font=font, fill=255, anchor="ls", stroke_width=1)
    ink = np.asarray(layer, np.float32)[..., None] / 255
    page[:] = page * (1 - ink) + np.array(color, np.float32) * ink
    return float((ink[..., 0] > 0.5).sum())


def test_cut_written_pages(tmp_path):
    counts = fake_counts()
    items = pages.choose_items(counts, 40, np.random.default_rng(3))
    man = pages.make_manifest(items, "Test")
    assert man["pages"] >= 2
    template = tmp_path / "template.pdf"
    pages.write_pdf(template, man)
    assert pages.read_manifest(template) == man
    blank = pages.read_pages([template])
    assert len(blank) == man["pages"] and blank[0][1].size == pages.PAGE
    drawn = np.asarray(pages.draw_page(man, 1).convert("L"), np.float32)
    assert np.abs(np.asarray(blank[0][1].convert("L"), np.float32) - drawn).mean() < 1

    rng = np.random.default_rng(4)
    written = [np.asarray(im, np.float32).copy() for _, im in blank]
    first = [it for it in man["items"] if it["page"] == 1]
    empty, over, blue, apart = first[0]["item"], first[1]["item"], first[2]["item"], first[3]["item"]
    ink = {}
    for it in man["items"]:
        if it["item"] == empty:
            continue
        ink[it["item"]] = write_on(written[it["page"] - 1], it, it["text"], rng,
                                   color=(20, 60, 190) if it["item"] == blue else (0, 0, 0),
                                   dy=20 if it["item"] == over else None)       # over the bottom line
    x0, y0 = first[3]["box"][:2]
    written[0][y0 - 30:y0 - 24, x0 + 100:x0 + 140] = 0                           # a stroke above its box
    written[0][pages.BOTTOM + 40:pages.BOTTOM + 46, 700:760] = 0                 # a stray scribble
    import pymupdf

    doc = pymupdf.open()
    for page in written[::-1][:-1]:                                              # reversed; page 1 below
        buf = io.BytesIO()
        Image.fromarray(page.astype(np.uint8)).save(buf, "PNG")
        doc.new_page(width=pages.PAGE[0] * 72 / pages.DPI, height=pages.PAGE[1] * 72 / pages.DPI).insert_image(
            pymupdf.Rect(0, 0, pages.PAGE[0] * 72 / pages.DPI, pages.PAGE[1] * 72 / pages.DPI), stream=buf.getvalue())
    doc.save(tmp_path / "written.pdf")
    photo = Image.fromarray(written[0].astype(np.uint8)).rotate(0.7, expand=True, fillcolor=(225, 225, 220))
    photo = photo.resize((int(photo.width * 0.8), int(photo.height * 0.8)), Image.BILINEAR)
    canvas = Image.new("RGB", (photo.width + 50, photo.height + 70), (225, 225, 220))
    canvas.paste(photo, (20, 30))
    canvas.save(tmp_path / "page1.jpg", quality=90)
    other = pages.make_manifest(pages.choose_items(counts, 10, np.random.default_rng(9)), "Other")
    pages.write_pdf(tmp_path / "other.pdf", other)

    out = tmp_path / "set"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cut_writing_pages.py"), "--template", str(template),
                        "--written", str(tmp_path / "written.pdf"), str(tmp_path / "page1.jpg"),
                        str(tmp_path / "other.pdf"), "--out-dir", str(out), "--tar", "--sheets", str(tmp_path / "sheets"),
                        "--summary", str(tmp_path / "summary.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(p["page"] for p in got["pages"] if p["page"]) == list(range(1, man["pages"] + 1))
    assert sum(p["page"] is None for p in got["pages"]) == 1                     # the other template's page
    by_item = {it["item"]: it for it in got["items"]}
    assert by_item[empty]["status"] == "empty"
    assert all(it["status"] == "written" for k, it in by_item.items() if k != empty)
    assert "outside" in by_item[over]["flags"] and "outside" in by_item[apart]["flags"]
    stray = [s for p in got["pages"] if p["page"] == 1 for s in p["stray"]]
    assert len(stray) == 1 and stray[0][1] >= pages.BOTTOM
    for k, it in by_item.items():
        if it["status"] != "written":
            continue
        g = np.asarray(Image.open(out / "images" / it["file"]))
        kept = (g < (170 if k == blue else 128)).sum() / ink[k]
        assert 0.85 < kept < (1.3 if k == apart else 1.1), (k, kept)
    assert len((out / "labels.tsv").read_text(encoding="utf-8").splitlines()) == len(man["items"]) - 1
    assert (tmp_path / "set.tar").exists() and len(list((tmp_path / "sheets").glob("page_*.png"))) == man["pages"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == {"written": len(man["items"]) - 1, "empty": 1} and summary["stray_pieces"] == 1

    pytest.importorskip("torch")
    from mayek_htr.data import FixedSet

    data = FixedSet(tmp_path / "set.tar")
    assert len(data) == len(man["items"]) - 1 and data.info is None
    assert data.kinds == [by_item[int(Path(n).stem)]["kind"] for n in data.names]
