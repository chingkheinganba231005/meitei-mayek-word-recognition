import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import corpus_stats as cs  # noqa: E402
import i_exceptions as ie  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
I, IL = cs.I_LETTER, cs.I_LONSUM


def test_candidates_and_twins(tmp_path):
    words = {
        "ꯂꯥ" + IL: 50,          # follows the convention
        "ꯂꯥ" + I: 5,            # variant: its twin is more frequent
        "ꯃꯆꯥꯈꯥ" + I + "ꯕ": 9,  # after ꯥ with ꯏ and no twin: possible exception
        I + "ꯃꯥ": 7,            # ꯏ at word start: follows the convention
        IL + "ꯃꯥ": 1,           # ꯢ at word start: variant of the above
        "ꯈꯨꯠ" + I: 3,          # after a final: follows the convention
    }
    path = tmp_path / "w.tsv"
    path.write_text("".join(f"{w}\t{n}\n" for w, n in words.items()), encoding="utf-8")
    rows, table = ie.candidates(ie.read_words(path))
    assert [(r["word"], r["hint"]) for r in rows] == [
        ("ꯃꯆꯥꯈꯥ" + I + "ꯕ", "only this spelling in the list"),
        ("ꯂꯥ" + I, "variant: the twin is more frequent"),
        (IL + "ꯃꯥ", "variant: the twin is more frequent")]
    assert table["after aa, o or u"] == {IL: 50, I: 14}

    out, summary = tmp_path / "c.csv", tmp_path / "s.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "i_exceptions.py"), str(path),
                    "--out", str(out), "--summary", str(summary)], check=True, capture_output=True)
    s = json.loads(summary.read_text(encoding="utf-8"))
    assert s["i_occurrences"] == 75 and s["disagreeing_occurrences"] == 15
    assert s["convention_holds"] == 0.8
    assert s["convention_holds_distinct_words"] == 0.5  # 3 of the 6 words' i's disagree
    top = s["convention_holds_without_top_candidate_word"]
    assert top["word"] == "ꯃꯆꯥꯈꯥ" + I + "ꯕ" and top["value"] == round(1 - 6 / 66, 4)
    sheet = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(sheet) == 3 and sheet[0]["expert_verdict"] == ""
