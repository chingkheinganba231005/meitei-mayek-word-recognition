"""Pages for writing words by hand, and cutting the written words out of them (Phase 3:
the real word set, written by the owner on an iPad).

Template (``choose_items``, ``layout``, ``write_pdf``): A4 pages at 200 dpi (1654 x 2339
px). Each word is printed above an empty box sized to it; the boxes are packed in rows and
a row's boxes are widened to fill it. Four black squares mark the corners. The PDF carries
its layout (every item's page, box and text) as an embedded file, manifest.json, so the
blank template PDF is all the cutter needs besides the written pages.

Cutting (``Cutter``), for pages exported from the iPad as a PDF or as images, in any
order, at any size:
1. the four corner squares give a projective map onto the template, so a page that was
   scaled, shifted or photographed still lines up;
2. the page is matched to its template page: the one whose printed ink it all shows;
3. handwriting is what is darker than the printed page nearby (so not the printed words,
   numbers and box lines), in any pen colour;
4. each connected piece of handwriting belongs to the box it lies in (the one it overlaps
   most; 'shared' if it overlaps two); a piece outside every box but near one goes to the
   nearest ('outside'), so a word written over its box line is kept whole;
5. the word is cut out of its own handwriting only (white elsewhere), with a margin of
   0.15 x its height, as ``images.crop_ink`` cuts a word out of a photo.
A box without handwriting is empty: the word was not written.
"""

import io
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from scipy import ndimage

from mayek_words.charset import ALPHABET, CHEIKHEI, DIGITS, problem
from mayek_words.glyphs import ASSETS
from mayek_words.lexicon import number, split_of

FORMAT = "mayek-writing-pages/1"
DPI = 200
PAGE = (1654, 2339)                       # A4 at 200 dpi
MARK, MARK_AT = 50, 45                    # corner squares: side, distance from the page edges
MARK_CENTRES = tuple((x + MARK / 2, y + MARK / 2) for y in (MARK_AT, PAGE[1] - MARK_AT - MARK)
                     for x in (MARK_AT, PAGE[0] - MARK_AT - MARK))   # top-left, top-right, bottom-left, -right
MARGIN, TOP, BOTTOM = 100, 190, PAGE[1] - 150
PROMPT_H, BOX_H, ROW_GAP, BOX_GAP = 58, 170, 24, 50                   # a box is 2.2 cm high
LINE, LINE_W = (150, 190, 240), 3         # box lines: light blue
PROMPT_SIZE = 46                          # printed words: letters 3.9 mm high
WRITE_L, MIN_BOX, PAD = 70, 330, 110      # a box fits the word at letters 8.9 mm high, x 1.25, + 1.4 cm
L_PER_EM = 0.663                          # letter height of the Meitei font, per em (glyph_priors.json)
MEITEI_FONT = ASSETS / "NotoSansMeeteiMayek-Regular.ttf"
INK = 60                                  # handwriting: this much darker than the printed page nearby
NEAR = 40                                 # a piece outside every box goes to one this near (5 mm)
MIN_PIECE, MIN_STRAY = 6, 30             # smaller pieces are specks; smaller stray pieces not reported
EDGE = 30                                 # the page's border (4 mm) holds no handwriting
INSTRUCTIONS = (100, 172)                 # rows of the instructions printed under the title


# ---------------------------------------------------------------- words to write

def choose_items(counts, n, rng, split="test", numbers=0.05, stop=0.02, min_count=2, min_len=2, max_len=14,
                 cover=8):
    """The items of a writing set: n words and numbers, in random order.

    Words are distinct words of the lexicon `counts` ({word: count}) whose hash split
    (``lexicon.split_of``) is `split`, so test words are never training words; with
    min_len..max_len characters, no digit or full stop, a count of at least min_count;
    drawn without replacement with probability proportional to count ** 0.5, as the
    synthetic sets draw words. First, every letter and sign gets `cover` occurrences if
    the lexicon has them (rare letters such as ꯘ, words with a count of 1 allowed). A
    share `numbers` of the items are numbers (``lexicon.number``, each digit at least
    cover // 2 times) and a share `stop` of the words end with a full stop (꯫), as in the
    synthetic sets. -> [{'text', 'kind'}] with kind 'lexicon' or 'number'."""
    def usable(w):
        return (split_of(w) == split and min_len <= len(w) <= max_len and problem(w) is None
                and not any(ch in DIGITS or ch == CHEIKHEI for ch in w))

    pool = sorted(w for w in counts if usable(w))
    weight = np.array([counts[w] for w in pool], np.float64) ** 0.5
    order = np.argsort(np.log(rng.random(len(pool))) / weight)[::-1]     # weighted, without replacement
    ranked = [pool[i] for i in order]
    n_num = int(round(numbers * n))
    n_words = n - n_num
    chosen, seen, have = [], set(), Counter()

    def take(w):
        chosen.append(w)
        seen.add(w)
        have.update(w)

    signs = [ch for ch in ALPHABET if ch not in DIGITS and ch != CHEIKHEI]
    in_pool = Counter(ch for w in ranked for ch in set(w))
    for ch in sorted(signs, key=lambda c: in_pool[c]):                   # rarest first
        for w in ranked:
            if have[ch] >= cover or len(chosen) >= n_words:
                break
            if ch in w and w not in seen:
                take(w)
    for w in ranked:
        if len(chosen) >= n_words:
            break
        if w not in seen and counts[w] >= min_count:
            take(w)
    words = [w + CHEIKHEI if rng.random() < stop else w for w in chosen]

    nums, digits = [], Counter()
    while len(nums) < n_num:
        x = number(rng)
        short = [d for d in sorted(DIGITS) if digits[d] < cover // 2]
        if short and not any(d in x for d in short) and len(nums) >= n_num // 2:
            continue                           # the second half fills the digits still short
        nums.append(x)
        digits.update(x)
    items = [{"text": w, "kind": "lexicon"} for w in words] + [{"text": x, "kind": "number"} for x in nums]
    return [items[i] for i in rng.permutation(len(items))]


# ---------------------------------------------------------------- template pages

def meitei_font(size):
    if not features.check("raqm"):
        raise RuntimeError("printing Meitei Mayek needs text shaping: Pillow with RAQM")
    return ImageFont.truetype(str(MEITEI_FONT), size, layout_engine=ImageFont.Layout.RAQM)


def box_width(text, font):
    """A box's width for the text: its printed width at letters WRITE_L high x 1.25, plus
    PAD, at least MIN_BOX, at most the page's width. `font`: the Meitei font at the size
    that gives letters WRITE_L high."""
    return int(min(max(1.25 * font.getlength(text) + PAD, MIN_BOX), PAGE[0] - 2 * MARGIN))


def layout(texts, lookahead=40):
    """Texts -> pages, each [(index into texts, box (x0, y0, x1, y1))] in reading order:
    rows filled left to right, each with the next text and then any of the following
    `lookahead` texts that still fit; a row's boxes widened to fill it; rows top to bottom."""
    font = meitei_font(round(WRITE_L / L_PER_EM))
    usable = PAGE[0] - 2 * MARGIN
    row_h = PROMPT_H + BOX_H + ROW_GAP
    per_page = (BOTTOM - TOP + ROW_GAP) // row_h
    widths = [box_width(t, font) for t in texts]
    left = list(range(len(texts)))
    rows = []
    while left:
        row = [left.pop(0)]
        used = widths[row[0]]
        for i in list(left[:lookahead]):
            if used + BOX_GAP + widths[i] <= usable:
                row.append(i)
                left.remove(i)
                used += BOX_GAP + widths[i]
        rows.append([(i, widths[i]) for i in row])
    pages = []
    for r, row in enumerate(rows):
        if r % per_page == 0:
            pages.append([])
        y0 = TOP + (r % per_page) * row_h + PROMPT_H
        extra = (usable - sum(w for _, w in row) - BOX_GAP * (len(row) - 1)) / len(row)
        x = float(MARGIN)
        for i, w in row:
            pages[-1].append((i, (round(x), y0, round(x + w + extra), y0 + BOX_H)))
            x += w + extra + BOX_GAP
    return pages


def make_manifest(items, title, **meta):
    """Items [{'text', 'kind'}] -> the manifest: the items laid out, numbered from 1 in
    reading order, each with its page and box."""
    pages = layout([it["text"] for it in items])
    out = []
    for p, page in enumerate(pages, 1):
        for i, box in page:
            out.append({"item": len(out) + 1, "page": p, "box": list(box), **items[i]})
    return {"format": FORMAT, "title": title, "dpi": DPI, "page_px": list(PAGE), "pages": len(pages),
            **meta, "items": out}


def draw_page(manifest, page_no):
    """Template page page_no (from 1) of a manifest -> RGB image."""
    im = Image.new("RGB", PAGE, "white")
    d = ImageDraw.Draw(im)
    mm, small, head = meitei_font(PROMPT_SIZE), ImageFont.load_default(24), ImageFont.load_default(32)
    for cx, cy in MARK_CENTRES:
        d.rectangle((cx - MARK / 2, cy - MARK / 2, cx + MARK / 2 - 1, cy + MARK / 2 - 1), fill="black")
    d.text((PAGE[0] / 2, 72), f"{manifest['title']}  -  page {page_no} of {manifest['pages']}", font=head,
           fill=(30, 30, 30), anchor="mm")
    d.text((PAGE[0] / 2, 122), "Copy each word into the box below it, in black, in your everyday handwriting.",
           font=small, fill=(90, 90, 90), anchor="mm")
    d.text((PAGE[0] / 2, 152), "To correct a word, erase it and write it again. Leave a box empty to skip its word.",
           font=small, fill=(90, 90, 90), anchor="mm")
    for it in manifest["items"]:
        if it["page"] != page_no:
            continue
        x0, y0, x1, y1 = it["box"]
        d.text((x0 + 4, y0 - 16), str(it["item"]), font=small, fill=(90, 90, 90), anchor="ls")
        d.text((x0 + 64, y0 - 16), it["text"], font=mm, fill="black", anchor="ls")
        d.rectangle((x0, y0, x1, y1), outline=LINE, width=LINE_W)
    return im


def write_pdf(path, manifest):
    """The template PDF: each page a lossless 200 dpi image on an A4 page, the manifest
    embedded as manifest.json."""
    import pymupdf

    doc = pymupdf.open()
    for p in range(1, manifest["pages"] + 1):
        page = doc.new_page(width=PAGE[0] * 72 / DPI, height=PAGE[1] * 72 / DPI)
        buf = io.BytesIO()
        draw_page(manifest, p).save(buf, "PNG", optimize=True)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.embfile_add("manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
    doc.set_metadata({"title": manifest["title"], "creator": "mayek_htr.pages"})
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()


def read_manifest(path):
    """The manifest embedded in a template PDF."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return json.loads(doc.embfile_get("manifest.json").decode("utf-8"))


def read_pages(paths, dpi=DPI):
    """PDFs (every page rendered at dpi, with its annotations) and image files -> [(name,
    RGB image)] in the order given."""
    out = []
    for path in paths:
        name = str(path)
        if name.lower().endswith(".pdf"):
            import pymupdf

            with pymupdf.open(name) as doc:
                for k, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=dpi, alpha=False)
                    out.append((f"{name}#{k + 1}", Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
        else:
            out.append((name, Image.open(name).convert("RGB")))
    return out


# ---------------------------------------------------------------- cutting

def find_marks(gray):
    """Centres of the four corner squares of a page (greyscale array), in the order of
    MARK_CENTRES, or None if one is missing."""
    H, W = gray.shape
    lab, _ = ndimage.label(gray < min(100.0, 0.5 * float(np.percentile(gray, 90))))
    found, best = [None] * 4, [np.inf] * 4
    corners = ((0, 0), (W, 0), (0, H), (W, H))
    for k, sl in enumerate(ndimage.find_objects(lab)):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (0.012 * W < min(h, w) and max(h, w) < 0.07 * W and 0.75 < w / h < 1.33):
            continue
        if (lab[sl] == k + 1).mean() < 0.85:                   # a solid square
            continue
        cx, cy = (sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2
        for q, (qx, qy) in enumerate(corners):
            dist = np.hypot(cx - qx, cy - qy)
            if dist < 0.2 * W and dist < best[q]:
                best[q], found[q] = dist, (cx, cy)
    return None if any(f is None for f in found) else found


def homography(src, dst):
    """The projective map (3 x 3) taking four points src to dst."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64)), 1.0).reshape(3, 3)


def flatten(g):
    """Paper made white: each pixel divided by the paper level around it (the brightest
    level within 6% of the page's width, smoothed), for photos in uneven light. A page
    exported from the iPad is white already and is left as it is."""
    if np.percentile(g, 90) >= 245:
        return g
    H, W = g.shape
    k = max(W // 200, 1)
    small = g[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).max(axis=(1, 3))
    bg = ndimage.uniform_filter(ndimage.maximum_filter(small, 12), 12)
    bg = np.asarray(Image.fromarray(bg.astype(np.float32), "F").resize((W, H), Image.BILINEAR))
    return np.clip(g * 255.0 / np.maximum(bg, 1.0), 0, 255).astype(np.float32)


def align(rgb):
    """A written page -> (greyscale float32 in the template's frame, corner squares found,
    resampled). A page exported as it was printed is used as it is; without the squares
    the page is taken to be the whole A4 page, scaled."""
    g = flatten(np.asarray(rgb.convert("L"), np.float32))
    gray = Image.fromarray(np.clip(g + 0.5, 0, 255).astype(np.uint8))
    marks = find_marks(g)
    if marks is None:
        return np.asarray(gray.resize(PAGE, Image.BILINEAR), np.float32), False, True
    if gray.size == PAGE and max(np.hypot(mx - cx, my - cy) for (mx, my), (cx, cy) in zip(marks, MARK_CENTRES)) < 1:
        return g, True, False
    Hm = homography(MARK_CENTRES, marks)
    out = gray.transform(PAGE, Image.Transform.PERSPECTIVE, tuple(Hm.flatten()[:8]), Image.BILINEAR, fillcolor=255)
    return np.asarray(out, np.float32), True, True


def block_min(g, k=4):
    H, W = g.shape[0] // k * k, g.shape[1] // k * k
    return g[:H, :W].reshape(H // k, k, W // k, k).min(axis=(1, 3))


class Cutter:
    """Cuts written pages of one template (its PDF: pages and manifest)."""

    def __init__(self, template_pdf):
        self.manifest = read_manifest(template_pdf)
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{template_pdf}: not a template of {FORMAT}")
        self.templates = [np.asarray(im.convert("L"), np.float32) for _, im in read_pages([template_pdf])]
        if any(t.shape != PAGE[::-1] for t in self.templates):
            raise ValueError("template pages are not A4 at 200 dpi")
        self.printed = [self.distinct(t) for t in self.templates]
        self.items = {p: [it for it in self.manifest["items"] if it["page"] == p]
                      for p in range(1, self.manifest["pages"] + 1)}

    @staticmethod
    def distinct(template):
        """The printed ink that tells a template page from the others, at half resolution:
        all but the corner squares and the instructions, which every page prints alike."""
        m = block_min(template, 2) < 128
        m[INSTRUCTIONS[0] // 2:INSTRUCTIONS[1] // 2] = False
        for cx, cy in MARK_CENTRES:
            r = MARK // 2 + 4
            m[int(cy - r) // 2:int(cy + r) // 2, int(cx - r) // 2:int(cx + r) // 2] = False
        return m

    def identify(self, page):
        """(template page number or None, share of each template page's distinct printed
        ink that the page lacks, within 3 px): a page lacks none of its own template's."""
        near = ndimage.minimum_filter(block_min(page, 2), 3) > 200
        lack = [float((near & printed).sum() / max(printed.sum(), 1)) for printed in self.printed]
        order = np.argsort(lack)
        ok = lack[order[0]] < 0.05 and (len(lack) == 1 or lack[order[1]] > 2 * lack[order[0]] + 0.05)
        return (int(order[0]) + 1 if ok else None), lack

    def cut(self, rgb):
        """A written page -> {'page', 'marks', 'lack', 'items': {item: (crop or None, flags)},
        'stray': [(x0, y0, x1, y1, pixels)]}."""
        page, marks, resampled = align(rgb)
        p, lack = self.identify(page)
        out = {"page": p, "marks": marks, "lack": round(min(lack), 4), "items": {}, "stray": []}
        if p is None:
            return out
        template = self.templates[p - 1]
        # darker than the printed page nearby (within 2 px; 4 px if the page was resampled,
        # as a photo is, whose printed edges are blurred)
        dark = ndimage.minimum_filter(template, 9 if resampled else 5) - page
        ink = dark > INK
        ink[:EDGE], ink[-EDGE:], ink[:, :EDGE], ink[:, -EDGE:] = False, False, False, False
        for cx, cy in MARK_CENTRES:                                   # the corner squares' blurred edges
            r = MARK // 2 + EDGE
            ink[int(cy) - r:int(cy) + r, int(cx) - r:int(cx) + r] = False
        lab, n = ndimage.label(ink, structure=np.ones((3, 3), bool))
        items = self.items[p]
        inner = [(it["box"][0] + LINE_W + 1, it["box"][1] + LINE_W + 1, it["box"][2] - LINE_W - 1,
                  it["box"][3] - LINE_W - 1) for it in items]
        pieces = {it["item"]: [] for it in items}
        flags = {it["item"]: set() for it in items}
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            piece = lab[sl] == j
            size = int(piece.sum())
            if size < MIN_PIECE:
                continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            inside = []
            for k, (bx0, by0, bx1, by1) in enumerate(inner):
                ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
                if ix0 < ix1 and iy0 < iy1:
                    m = int(piece[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].sum())
                    if m:
                        inside.append((m, k))
            if inside:
                inside.sort(reverse=True)
                k = inside[0][1]
                if inside[0][0] < size:
                    flags[items[k]["item"]].add("outside")
                if len(inside) > 1:
                    for _, k2 in inside:
                        flags[items[k2]["item"]].add("shared")
            else:
                gaps = [max(bx0 - x1, x0 - bx1, 0) + max(by0 - y1, y0 - by1, 0) for bx0, by0, bx1, by1 in inner]
                k = int(np.argmin(gaps))
                if gaps[k] > NEAR:
                    if size >= MIN_STRAY:
                        out["stray"].append((x0, y0, x1, y1, size))
                    continue
                flags[items[k]["item"]].add("outside")
            pieces[items[k]["item"]].append((j, sl))
        for it in items:
            ids = pieces[it["item"]]
            if not ids:
                out["items"][it["item"]] = (None, [])
                continue
            y0 = min(sl[0].start for _, sl in ids)
            y1 = max(sl[0].stop for _, sl in ids)
            x0 = min(sl[1].start for _, sl in ids)
            x1 = max(sl[1].stop for _, sl in ids)
            pad = int(0.15 * (y1 - y0)) + 1
            y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
            y1, x1 = min(y1 + pad, PAGE[1]), min(x1 + pad, PAGE[0])
            mine = np.isin(lab[y0:y1, x0:x1], [j for j, _ in ids])
            keep = ndimage.binary_dilation(mine, iterations=2) & (dark[y0:y1, x0:x1] > 16)
            crop = np.where(keep, page[y0:y1, x0:x1], 255.0)
            out["items"][it["item"]] = (np.clip(crop + 0.5, 0, 255).astype(np.uint8), sorted(flags[it["item"]]))
        return out
