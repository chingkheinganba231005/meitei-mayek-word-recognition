"""Exception candidates for the ꯢ / ꯏ convention, from a word-frequency list.

    python scripts/i_exceptions.py wikipedia_consistent_i_lonsum.tsv \\
        --out results/i_exception_candidates.csv --summary results/i_exception_summary.json

The input is a word list as written by corpus_stats.py --words-out (word<TAB>count),
normally the one of the documents that always write ꯢ after ꯥ, ꯣ or ꯨ.

The convention tested is the standard spelling's: ꯢ for the i after ꯥ, ꯣ or ꯨ, ꯏ
everywhere else. (The project itself writes ꯏ throughout; the standard spelling is an
optional rendering of its output.) Every i that disagrees with it becomes a row of the review sheet,
with its "twin" (the same word with the other i) and how often the twin occurs in the
same list. A twin that is more frequent points to a variant or a slip; no twin points to
a possible genuine exception. The last column is left empty for the language expert.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from corpus_stats import AA_O_U, I_LETTER, I_LONSUM, LONSUM, NUNG_APUN, VOWEL_LETTERS, VOWEL_SIGNS

OTHER = {I_LETTER: I_LONSUM, I_LONSUM: I_LETTER}


def context(prev):
    """Where an i stands, in the terms of the convention."""
    if prev is None:
        return "word start"
    if prev in AA_O_U:
        return "after aa, o or u"
    if prev in VOWEL_SIGNS:
        return "after another vowel sign"
    if prev in VOWEL_LETTERS:
        return "after a vowel letter"
    if prev in LONSUM:
        return "after a final (lonsum)"
    if prev in NUNG_APUN:
        return "after nung or apun"
    if chr(0xABC0) <= prev <= chr(0xABDA):  # consonant letters (with the vowel letters handled above)
        return "after a consonant letter"
    return "other"


def convention(prev):
    return I_LONSUM if prev is not None and prev in AA_O_U else I_LETTER


def read_words(path):
    words = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, n = line.rstrip("\n").partition("\t")
            if word and n.isdigit():
                words[word] += int(n)
    return words


def candidates(words):
    """Rows for every i that disagrees with the convention, and counts per context."""
    table = defaultdict(Counter)
    rows = []
    for word, n in words.items():
        for i, ch in enumerate(word):
            if ch not in OTHER:
                continue
            prev = word[i - 1] if i else None
            where = context(prev)
            table[where][ch] += n
            expected = convention(prev)
            if ch != expected:
                twin = word[:i] + expected + word[i + 1:]
                twin_n = words.get(twin, 0)
                hint = ("variant: the twin is more frequent" if twin_n > n else
                        "only this spelling in the list" if twin_n == 0 else "both spellings")
                rows.append({"word": word, "count": n, "position": i + 1, "where": where,
                             "before": prev or "", "written": ch, "convention": expected,
                             "twin": twin, "twin_count": twin_n, "hint": hint, "expert_verdict": ""})
    rows.sort(key=lambda r: (-r["count"], r["word"], r["position"]))
    return rows, table


def summarise(words, rows, table):
    by_where = [{"where": w, "i_lonsum": c[I_LONSUM], "i_letter": c[I_LETTER],
                 "convention": I_LONSUM if w == "after aa, o or u" else I_LETTER}
                for w, c in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))]
    total = sum(sum(c.values()) for c in table.values())
    off = sum(r["count"] for r in rows)
    hints = Counter()
    for r in rows:
        hints[r["hint"]] += r["count"]
    types = sum(sum(1 for ch in w if ch in OTHER) for w in words)  # each word's i's counted once
    top = rows[0] if rows else None
    top_word_off = sum(r["count"] for r in rows if top and r["word"] == top["word"])
    top_word_all = top["count"] * sum(1 for ch in top["word"] if ch in OTHER) if top else 0
    return {"words_in_list": len(words), "running_words": sum(words.values()),
            "i_occurrences": total, "disagreeing_occurrences": off,
            "convention_holds": round(1 - off / total, 4) if total else None,
            "convention_holds_distinct_words": round(1 - len(rows) / types, 4) if types else None,
            "convention_holds_without_top_candidate_word": (
                {"word": top["word"], "value": round(1 - (off - top_word_off) / (total - top_word_all), 4)}
                if top and total > top_word_all else None),
            "disagreeing_words": len({r["word"] for r in rows}),
            "disagreeing_occurrences_by_hint": dict(hints),
            "by_where": by_where,
            "top_candidates": [[r["word"], r["count"], r["where"], r["twin_count"]] for r in rows[:20]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wordlist")
    ap.add_argument("--out", default="results/i_exception_candidates.csv")
    ap.add_argument("--summary", default="results/i_exception_summary.json")
    args = ap.parse_args()

    words = read_words(args.wordlist)
    rows, table = candidates(words)
    summary = {"wordlist": Path(args.wordlist).name, **summarise(words, rows, table)}
    for path in (args.out, args.summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["word"])
        w.writeheader()
        w.writerows(rows)
    Path(args.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{summary['i_occurrences']:,} i's, convention holds for {summary['convention_holds']:.2%} "
          f"({summary['convention_holds_distinct_words']:.2%} over distinct words); "
          f"{summary['disagreeing_occurrences']:,} disagree, in {summary['disagreeing_words']} words")
    for hint, n in summary["disagreeing_occurrences_by_hint"].items():
        print(f"  {hint}: {n}")
    print(f"written to {args.out} and {args.summary}")


if __name__ == "__main__":
    main()
