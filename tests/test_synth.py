import json
from pathlib import Path

import numpy as np
import pytest

from mayek_words.charset import CHEIKHEI, DIGITS, I_LETTER, I_LONSUM
from mayek_words.glyphs import GlyphStore
from mayek_words.lexicon import Lexicon
from mayek_words.synth import (Config, Words, WordSynth, contact, ink_distance, line_gaps, load_priors,
                               reach_past, set_pen, sign_body, stroke_width)

K, LAI = chr(0xABC0), chr(0xABC2)
ANAP, UNAP, INAP, NUNG, APUN = chr(0xABE5), chr(0xABE8), chr(0xABE4), chr(0xABEA), chr(0xABED)
PLAIN = Config(width=(1, 1), glyph_width_jitter=0, glyph_height_jitter=0, gap=(0.1, 0.1), gap_jitter=0,
               baseline_jitter=0, mark_scale=(1, 1), mark_size_jitter=0, mark_jitter=0, slant=0, rotation=0,
               pen=None, blur=(0, 0), noise=0, letter_height_jitter=0, style_k=None, margin=(0.2, 0.2),
               touch=None, p_uneven=0.0)


@pytest.fixture(scope="module")
def store():
    return GlyphStore.from_font()


def test_same_seed_same_image(store):
    s = WordSynth(store)
    a = s.render(K + ANAP + I_LONSUM, np.random.default_rng(5))
    b = s.render(K + ANAP + I_LONSUM, np.random.default_rng(5))
    c = s.render(K + ANAP + I_LONSUM, np.random.default_rng(6))
    assert a.image.dtype == np.uint8 and a.image.ndim == 2
    assert np.array_equal(a.image, b.image) and a.image.shape != c.image.shape or not np.array_equal(a.image, c.image)
    assert a.text == K + ANAP + I_LETTER                      # everyday spelling
    with pytest.raises(ValueError):
        s.render("abc", np.random.default_rng(0))


def boxes(store, word):
    return {ch: b for ch, *b in WordSynth(store, config=PLAIN).render(word, np.random.default_rng(0)).boxes}


def test_signs_sit_where_they_belong(store):
    b = boxes(store, K + ANAP)                  # above the letter
    assert b[ANAP][3] <= b[K][1] + 2 and b[K][0] < b[ANAP][0] < b[K][2]
    b = boxes(store, K + UNAP)                  # below it
    assert b[UNAP][1] >= b[K][3] - 2
    b = boxes(store, K + INAP)                  # beside it, as tall as a letter or taller
    assert b[INAP][0] >= b[K][2] - 1 and b[INAP][3] - b[INAP][1] >= b[K][3] - b[K][1]
    b = boxes(store, K + APUN + LAI)            # under the letter before it
    assert b[APUN][1] >= b[K][3] - 2 and abs(b[APUN][0] - b[K][0]) < 3 and b[LAI][0] > b[K][2]
    b = boxes(store, K + chr(0xABE3) + NUNG)    # nung above the sign before it
    assert b[NUNG][3] <= b[chr(0xABE3)][3] and b[NUNG][2] > b[K][2]


def test_boxes_inside_image(store):
    s = WordSynth(store)
    for i in range(20):
        r = s.render(K + ANAP + LAI + UNAP + NUNG, np.random.default_rng(i))
        H, W = r.image.shape
        for _, x0, y0, x1, y1 in r.boxes:
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


def test_letters_closer_than_in_print(store):
    """Handwritten Meitei Mayek sets letters closer together than print (the owner's
    observation); the default spacing must stay tighter than the font's."""
    words = [K + LAI + chr(0xABC3) + chr(0xABC4), LAI + INAP + K + chr(0xABC5)] * 150
    def median_gap(cfg):
        s = WordSynth(store, config=cfg)
        layouts = [s.render(w, np.random.default_rng(i)).layout for i, w in enumerate(words)]
        return float(np.median(line_gaps(layouts, s.prior)["letter to letter"]))
    printed = median_gap(Config(gap=(0, 0), gap_jitter=0, width=(1, 1), glyph_width_jitter=0))
    assert median_gap(Config()) < printed


def test_contact():
    canvas = np.zeros((20, 40), np.float32)
    canvas[2:18, 10:13] = 1                      # a stroke ending at column 12
    ink = np.zeros((16, 6), np.float32)
    ink[:, 1:3] = 1                              # to be pasted at column 20: its ink starts at 21
    assert contact(canvas, ink, 20, 2, 20) == 21 - 12 - 1
    assert contact(canvas, ink, 20, 2, 5) == 0   # nothing within reach


def test_ink_distance_and_reach():
    canvas = np.zeros((20, 40), np.float32)
    canvas[2:18, 10:13] = 1                      # a stroke ending at column 12
    ink = np.zeros((16, 6), np.float32)
    ink[:, 1:3] = 1                              # pasted at column 20: its ink starts at 21
    assert ink_distance(canvas, ink, 20, 2, 20) == 21 - 12 - 1
    assert reach_past(canvas, ink, 20, 2, 20) == 12 - 21 + 1
    assert ink_distance(canvas, ink, 11, 2, 20) == -1                  # overlapping
    assert ink_distance(canvas, ink, 30, 2, 5) is None                 # nothing within reach
    lead = np.zeros((10, 10), np.float32)
    lead[1, 0:6] = 1                             # a thin lead-in stroke into ...
    lead[1:10, 6:8] = 1                          # ... a stem
    assert sign_body(lead)[:, :6].max() == 0 and sign_body(lead)[:, 6:].sum() == lead[:, 6:].sum()


def test_sign_nearer_its_own_letter(store):
    """A sign beside its letter (ꯤ, ꯦ, ꯣ, ꯧ) is never nearer the next letter than its own, on the
    ink as drawn, and the next letter never touches it: in evenly spaced words the gaps are about
    even, in unevenly spaced words the sign is closer to its letter (owner, 25 September 2026)."""
    from scipy import ndimage

    def dist(a, b):
        (_, ax, ay, ai), (_, bx, by, bi) = a, b
        H, W = max(ay + ai.shape[0], by + bi.shape[0]), max(ax + ai.shape[1], bx + bi.shape[1])
        A, B = np.zeros((H, W), bool), np.zeros((H, W), bool)
        A[ay:ay + ai.shape[0], ax:ax + ai.shape[1]] = ai > 0.5
        B[by:by + bi.shape[0], bx:bx + bi.shape[1]] = bi > 0.5
        return -1.0 if (A & B).any() else float(ndimage.distance_transform_edt(~A)[B].min()) - 1

    ratios = {}
    for uneven in (0.0, 1.0):
        s = WordSynth(store, config=Config(touch=(0.5, 0.5), p_uneven=uneven))
        ratios[uneven] = []
        for sign in (INAP, chr(0xABE6), chr(0xABE3), chr(0xABE7)):
            for i in range(15):
                trace = []
                s.render(K + sign + LAI + K, np.random.default_rng(i), trace)
                letter, (_, x, y, ink), nxt = trace[0], trace[1], trace[2]
                own, after = dist(letter, (None, x, y, sign_body(ink))), dist(trace[1], nxt)
                assert own < after and after >= 1
                ratios[uneven].append((max(own, 0) + 1) / (after + 1))
    assert np.median(ratios[1.0]) < np.median(ratios[0.0])      # closer to its letter when uneven


def test_joined_letters_share_ink(store):
    """With every letter joined, a word is one piece of ink more often than with none joined."""
    from scipy import ndimage
    word = K + LAI + chr(0xABC3) + chr(0xABC4) + chr(0xABC5)
    def pieces(touch):
        cfg = Config(touch=touch, gap=(0.1, 0.1), slant=0, rotation=0, blur=(0, 0), noise=0)
        s = WordSynth(store, config=cfg)
        return [ndimage.label(s.render(word, np.random.default_rng(i)).image < 128,
                              structure=np.ones((3, 3)))[1] for i in range(20)]
    assert np.mean(pieces((1.0, 1.0))) < np.mean(pieces((0.0, 0.0))) - 1.5


def test_pen_width():
    alpha = np.zeros((30, 30), np.float32)
    alpha[3:27, 13:16] = 1                      # a 3 px stroke
    grown, border = set_pen(alpha, 7)
    assert border > 0 and abs(stroke_width(grown > 0.5) - 7) <= 1.5
    thinned, border = set_pen(grown, 3)
    assert abs(stroke_width(thinned > 0.5) - 3) <= 1


def test_measured_sizes_replace_the_font(tmp_path):
    sizes = tmp_path / "sizes.json"
    sizes.write_text(json.dumps({"classes": [{"char": ANAP, "w": 0.9, "h": 0.5},
                                             {"char": UNAP, "w": None, "h": 10.0}]}), encoding="utf-8")
    font, measured = load_priors(sizes=None), load_priors(sizes=sizes)
    assert measured[ANAP]["w"] == 0.9 and measured[ANAP]["bottom"] == font[ANAP]["bottom"]
    assert abs(measured[ANAP]["top"] - measured[ANAP]["bottom"] - 0.5) < 1e-9
    h0 = font[UNAP]["top"] - font[UNAP]["bottom"]           # below the baseline: keeps its top, clipped at 2x
    assert measured[UNAP]["top"] == font[UNAP]["top"]
    assert abs(measured[UNAP]["top"] - measured[UNAP]["bottom"] - 2 * h0) < 1e-9


def test_measured_sizes_are_the_default():
    root = Path(__file__).resolve().parents[1]
    packaged = json.loads((root / "mayek_words" / "assets" / "glyph_sizes_tummhcd.json").read_text(encoding="utf-8"))
    result = json.loads((root / "results" / "glyph_sizes_tummhcd.json").read_text(encoding="utf-8"))
    assert packaged["classes"] == result["classes"]                 # the copy is the latest run's result
    default, measured = load_priors(), load_priors(sizes=root / "results" / "glyph_sizes_tummhcd.json")
    assert default == measured and default != load_priors(sizes=None)


def test_words_on_demand(store):
    lex = Lexicon({K + ANAP: 5, LAI: 1})
    w = Words(WordSynth(store), lex, seed=3)
    assert np.array_equal(w[7].image, w[7].image) and w[7].text in (K + ANAP, LAI)
    digits = Words(WordSynth(store), lex, seed=3, numbers=1.0, stop=1.0)[0].text
    built = Words(WordSynth(store), lex, seed=3, numbers=0.0, built=1.0)
    assert built.bank is not None and len(built[1].text) >= 1
    assert Words(WordSynth(store), lex, seed=3).bank is not None         # on by default (15%)
    assert Words(WordSynth(store), lex, seed=3, built=0.0).bank is None
    assert digits[-1] == CHEIKHEI and all(ch in DIGITS for ch in digits[:-1])
