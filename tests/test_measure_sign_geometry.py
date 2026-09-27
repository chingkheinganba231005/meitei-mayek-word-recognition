import dataclasses
import importlib.util
from pathlib import Path

import numpy as np

from mayek_words.glyphs import GlyphStore
from mayek_words.synth import Config, WordSynth

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_sign_geometry", ROOT / "scripts" / "measure_sign_geometry.py")
msg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msg)

K, LAI, ANAP = chr(0xABC0), chr(0xABC2), chr(0xABE5)
PLAIN = Config(width=(1, 1), glyph_width_jitter=0, glyph_height_jitter=0, gap=(0.1, 0.1), gap_jitter=0,
               baseline_jitter=0, mark_scale=(1, 1), mark_size_jitter=0, mark_jitter=0, slant=0, rotation=0,
               pen=(0.1, 0.1), blur=(0, 0), noise=0, letter_height=48, letter_height_jitter=0, style_k=None,
               margin=(0.2, 0.2), touch=None, p_uneven=0.0)


def test_axes():
    t = np.arange(40.0)
    assert abs(msg.axis_angle(t, t) - 45) < 1e-6              # falling to the right (image y down)
    assert abs(msg.axis_angle(t, -t) - 135) < 1e-6
    a, e = msg.axes(np.cos(t / 40 * 2 * np.pi), np.sin(t / 40 * 2 * np.pi))
    assert e < 1.1                                            # a ring


def test_print_and_beside():
    """Print puts ꯥ over its letter, round 3 after it; the measure tells them apart."""
    store = GlyphStore.from_font()
    for p, inside in ((0.0, True), (1.0, False)):
        synth = WordSynth(store, config=dataclasses.replace(PLAIN, p_marks_beside=p))
        centres = []
        for i in range(3):
            smp = synth.render(K + ANAP + LAI, np.random.default_rng(i))
            m, why = msg.measure(smp.image, smp.text)
            assert m is not None, why
            (sign,) = m
            assert sign["char"] == ANAP and 20 < sign["angle"] < 70 and sign["elongation"] > 2
            centres.append(sign["centre"])
            assert (sign["bottom"] > 0) == inside                 # print: wholly above the letter's top
        assert (max(centres) < 0) if inside else (min(centres) > 0)


def test_image_shapes():
    shapes = msg.image_shapes(GlyphStore.from_font())
    assert shapes[ANAP]["n"] == 1 and shapes[ANAP]["falling_right_20_to_70"] == 1.0
