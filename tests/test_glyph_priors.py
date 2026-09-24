import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_committed_priors_are_what_the_script_writes(tmp_path):
    pytest.importorskip("uharfbuzz")
    pytest.importorskip("fontTools")
    out, glyphs = tmp_path / "p.json", tmp_path / "g.npz"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glyph_priors.py"), "--out", str(out),
                    "--glyphs", str(glyphs)], check=True, capture_output=True)
    committed = json.loads((ROOT / "mayek_words" / "assets" / "glyph_priors.json").read_text(encoding="utf-8"))
    assert json.loads(out.read_text(encoding="utf-8"))["classes"] == committed["classes"]
    assert np.array_equal(np.load(glyphs)["images"], np.load(ROOT / "mayek_words" / "assets" / "font_glyphs.npz")["images"])
