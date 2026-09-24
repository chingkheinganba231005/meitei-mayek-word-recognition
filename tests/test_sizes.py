import numpy as np

from mayek_words.glyphs import GlyphStore
from mayek_words.sizes import chord_thickness, estimate_boxes
from mayek_words.synth import load_priors


def test_chords():
    m = np.zeros((24, 24), bool)
    m[2:22, 10:13] = True                     # vertical bar, 3 px thick
    assert chord_thickness(m) == (3.0, None)
    m = np.zeros((24, 24), bool)
    m[10:14, 2:22] = True                     # horizontal bar, 4 px thick
    assert chord_thickness(m) == (None, 4.0)
    m[2:22, 10:13] = True                     # a cross: both
    assert chord_thickness(m) == (3.0, 4.0)
    assert chord_thickness(np.ones((24, 24), bool)) == (None, None)   # a blob, not a stroke


def test_sizes_come_back_on_the_font_characters():
    store = GlyphStore.from_font()
    T, est = estimate_boxes(store.alpha > 0.5, store.labels)
    prior = load_priors()
    anap, yenap = est[45], est[46]            # small signs: stretched most
    assert abs(anap["w"] / prior[chr(0xABE5)]["w"] - 1) < 0.15
    assert abs(anap["h"] / (prior[chr(0xABE5)]["top"] - prior[chr(0xABE5)]["bottom"]) - 1) < 0.2
    assert abs(yenap["h"] / (prior[chr(0xABE6)]["top"] - prior[chr(0xABE6)]["bottom"]) - 1) < 0.2
    letters = [est[c]["h"] for c in range(10, 37)]
    assert np.median(letters) == 1.0
