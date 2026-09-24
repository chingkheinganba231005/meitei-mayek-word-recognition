import bz2
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import corpus_stats as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
I, IL = cs.I_LETTER, cs.I_LONSUM


def test_rule_prediction():
    assert cs.rule_prediction(I + "ꯃ", 0) == I             # word start
    assert cs.rule_prediction("ꯀ" + IL, 1) == IL          # after a consonant letter (inherent a)
    assert cs.rule_prediction("ꯀꯥ" + IL, 2) == IL         # after a vowel sign
    assert cs.rule_prediction("ꯀꯛ" + I, 2) == I           # after a final consonant
    assert cs.rule_prediction("ꯀ꯭" + I, 2) == I           # after apun


def test_stats_words_and_rule():
    s = cs.Stats()
    s.add("ꯀ" + IL + " " + I + "ꯃ꯫ <b>markup</b> ১২৩ ꯱꯲")   # Meitei word, word with ꯏ, full stop, Bengali, digits
    s.add("ꯀ" + IL + " ꯃ" + I)                            # second ꯃꯏ breaks the rule
    r = s.summary()
    assert r["running_words"] == 4
    assert r["distinct_words"] == 3
    rule = r["rule_running_words"]
    assert rule["occurrences"] == 4
    assert rule["rule_accuracy"] == 0.75
    assert rule["actual_i_letter"] == {I: 1, IL: 1}
    assert rule["i_lonsum_per_i_letter"] == 1.0
    assert r["rule_distinct_words"]["occurrences"] == 3


def test_contexts():
    s = cs.Stats()
    s.add("ꯀꯥ" + I + " ꯀꯥ" + IL + " " + I + "ꯃ ꯀꯛ" + I)  # after a vowel sign twice, word start, after a lonsum
    kinds = {r["before"]: (r["i_lonsum"], r["i_letter"])
             for r in s.summary()["i_contexts_running_words"]["by_kind"]}
    assert kinds == {"vowel sign": (1, 1), "word start": (0, 1), "final consonant (lonsum)": (0, 1)}


def test_cli_reads_folders_and_bz2(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("ꯃꯤꯇꯩ ꯂꯣꯟ\n", encoding="utf-8")
    with bz2.open(tmp_path / "wiki.xml.bz2", "wt", encoding="utf-8") as f:
        f.write("<page><text>ꯃꯅꯤꯄꯨꯔ [[link]]</text></page>")
    out = tmp_path / "stats.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "corpus_stats.py"), f"a={tmp_path / 'a'}",
                    f"wiki={tmp_path / 'wiki.xml.bz2'}", f"missing={tmp_path / 'nope'}", "--out", str(out)],
                   check=True, capture_output=True)
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["a"]["running_words"] == 2
    assert r["wiki"]["top_words"] == [["ꯃꯅꯤꯄꯨꯔ", 1]]
    assert r["missing"]["error"] == "not found"
    assert r["all_sources_combined"]["running_words"] == 3
