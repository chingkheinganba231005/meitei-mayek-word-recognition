import sys
from pathlib import Path

import numpy as np

from mayek_words.glyphs import GlyphStore
from mayek_words.lexicon import Lexicon
from mayek_words.synth import Config, Words, WordSynth, line_gaps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import measure_spacing as ms  # noqa: E402

LETTERS = [chr(c) for c in range(0xABC0, 0xABDB)]


def page_and_truth(touch, seed=0):
    rng = np.random.default_rng(seed)
    lex = Lexicon({"".join(rng.choice(LETTERS, 5)): 1 for _ in range(200)})
    cfg = Config(touch=touch, slant=0, rotation=0, blur=(0, 0), noise=0, letter_height_jitter=0)
    words = Words(WordSynth(GlyphStore.from_font(), config=cfg), lex, numbers=0, stop=0)
    layouts = []
    render = words.synth.render

    def keep(text, r):  # remember each word's layout
        smp = render(text, r)
        layouts.append(smp.layout)
        return smp
    words.synth.render = keep
    page = ms.synthetic_page(words, rng)
    return page, np.array(line_gaps(layouts, words.synth.prior)["letter to letter"])


def test_spacing_comes_back_from_a_synthetic_page():
    page, truth = page_and_truth((0.0, 0.0))
    r = ms.measure(ms.pieces(page.astype(np.float32)))
    assert abs(r["letter_height_px"] - 32) <= 3
    assert abs(r["visible_gaps_in_L"]["median"] - np.median(truth[truth >= 0.03])) < 0.05


def test_touching_is_seen():
    joined = ms.measure(ms.pieces(page_and_truth((1.0, 1.0))[0].astype(np.float32)))["touching"]
    apart = ms.measure(ms.pieces(page_and_truth((0.0, 0.0))[0].astype(np.float32)))["touching"]
    assert joined > 0.6 and apart < 0.35
