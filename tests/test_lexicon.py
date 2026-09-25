import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mayek_words import lexicon as lx
from mayek_words.charset import DIGITS, I_LETTER, I_LONSUM

ROOT = Path(__file__).resolve().parents[1]
K, ANAP, APUN, LAI = chr(0xABC0), chr(0xABE5), chr(0xABED), chr(0xABC2)


def words(n, seed=0):
    rng = np.random.default_rng(seed)
    letters = [chr(c) for c in range(0xABC0, 0xABDB)]
    return ["".join(rng.choice(letters, rng.integers(2, 8))) for _ in range(n)]


def test_split_by_word_is_stable_and_proportional():
    ws = set(words(20000))
    parts = Counter(lx.split_of(w) for w in ws)
    assert abs(parts["train"] / len(ws) - 0.90) < 0.015 and abs(parts["val"] / len(ws) - 0.05) < 0.01
    assert all(lx.split_of(w) == lx.split_of(w) for w in list(ws)[:100])


def test_clean_combine_split():
    raw = Counter({K + ANAP + I_LONSUM: 7, K + ANAP + I_LETTER: 3, K + APUN: 2, "x": 1, K * 30: 1})
    kept, dropped = lx.clean(raw)
    assert kept == Counter({K + ANAP + I_LETTER: 10})       # two spellings of one word
    assert dropped == {"apun not between two consonants": [1, 2], "character outside the alphabet": [1, 1],
                       "longer than 24 characters": [1, 1]}
    assert lx.combine({"a": Counter({K: 3, LAI: 1}), "b": Counter({K: 2, LAI: 4})}) == Counter({K: 3, LAI: 4})
    counts = Counter({w: 1 for w in words(3000)})
    parts = lx.split(counts)
    assert sum(len(p) for p in parts.values()) == len(counts)
    assert not set(parts["train"]) & set(parts["test"])
    some = list(parts["train"])[:5]
    assert not set(some) & set(lx.split(counts, exclude=some)["train"])


def test_sampling_and_numbers():
    lex = lx.Lexicon(Counter({K: 100, LAI: 1}), alpha=1.0)
    rng = np.random.default_rng(0)
    draws = Counter(lex.sample(rng) for _ in range(5000))
    assert 0.97 < draws[K] / 5000 < 1.0
    flat = lx.Lexicon(Counter({K: 100, LAI: 1}), alpha=0.0)
    assert 0.45 < Counter(flat.sample(rng) for _ in range(5000))[K] / 5000 < 0.55
    for _ in range(200):
        n = lx.number(rng)
        assert all(ch in DIGITS for ch in n) and (len(n) == 1 or n[0] != chr(0xABF0))


def test_rare_characters_are_drawn_more_often():
    counts = Counter({K + ANAP: 10000, LAI + ANAP: 10000, K + LAI: 10000, chr(0xABD8) + ANAP: 1})  # ꯘ: 0.17%
    GHOU = chr(0xABD8)
    plain, boosted = lx.Lexicon(counts, rare_share=0.0), lx.Lexicon(counts)
    assert boosted.rare_chars == [GHOU] and plain.rare_chars == []
    a, b = lx.sampled_shares(plain, 20000), lx.sampled_shares(boosted, 20000)
    assert a[GHOU] < 0.003 and b[GHOU] > 0.03 and b[K] > 0.85 * a[K]   # the rest shrink by about rare_share


def test_words_built_from_syllables():
    AA, NG, M_L = chr(0xABE5), chr(0xABE1), chr(0xABDD)
    lex = lx.Lexicon(Counter({K + AA + NG + LAI: 50, K + M_L: 20, LAI + K + AA: 10}), rare_share=0.0)
    bank = lx.SyllableBank(lex)
    assert set(bank.kinds) == {"C", "CV", "CVC", "CC"}
    rng = np.random.default_rng(0)
    words = [bank.word(rng) for _ in range(4000)]
    s = lx.structure(words)
    assert all(v > 0.08 for v in s["syllables_per_word"].values())       # 1 to 8 syllables, all present
    assert all(abs(v - 0.25) < 0.03 for v in s["syllable_kinds"].values())  # each kind about a quarter
    assert lx.structure([K + AA, K])["syllables_per_word"]["1"] == 1.0


def test_scripts_end_to_end(tmp_path):
    src = tmp_path / "wiki.tsv"
    src.write_text("".join(f"{w}\t{i + 1}\n" for i, w in enumerate(words(400))), encoding="utf-8")
    run = [sys.executable, str(ROOT / "scripts" / "build_lexicon.py"), f"wiki={src}",
           "--out-dir", str(tmp_path / "lex"), "--stats", str(tmp_path / "stats.json")]
    subprocess.run(run, check=True, capture_output=True)
    assert (tmp_path / "lex" / "train.tsv").exists()
    run = [sys.executable, str(ROOT / "scripts" / "render_words.py"), "--glyphs", "font",
           "--lexicon", str(tmp_path / "lex" / "train.tsv"), "--n", "5", "--seed", "1",
           "--out-dir", str(tmp_path / "synth"), "--sheet", str(tmp_path / "sheet.png")]
    subprocess.run(run, check=True, capture_output=True)
    labels = (tmp_path / "synth" / "labels.tsv").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 5 and (tmp_path / "synth" / "images" / "000004.png").exists()
    assert (tmp_path / "sheet.png").exists()
