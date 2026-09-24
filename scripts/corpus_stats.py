"""Size, script purity and ꯢ / ꯏ statistics of Meitei Mayek text corpora.

    python scripts/corpus_stats.py wiki=corpora/mniwiki-latest-pages-articles.xml.bz2 \\
        fineweb2=corpora/fineweb2/data/mni_Mtei --out results/corpus_stats.json

Each source is name=path; a path can be a file or a folder (read recursively).
Readable files: .txt, .xml, .bz2 and .gz of those, .parquet and .jsonl(.gz)
(the "text" field, or every string field if there is none).

Words are maximal runs of Meitei Mayek letters and signs (U+ABC0-U+ABEA,
U+ABEC-U+ABED and the extensions U+AAE0-U+AAF6), so markup, other scripts,
digits and the full stop (cheikhei, U+ABEB) never count. This also means
Bengali-script text contributes nothing: the counts are native Meitei Mayek
only.

The orthographic rule (Hijam's thesis): ꯢ (i lonsum, U+ABE2) follows a vowel;
ꯏ (i, U+ABCF) begins a word or follows a non-vowel. Here "non-vowel" is a
final consonant (lonsum, ꯛ ... ꯡ), nung (ꯪ) or apun (꯭); a consonant letter
(inherent a), a vowel letter or a vowel sign counts as a vowel. For every ꯢ
or ꯏ the rule predicts one of the two from the preceding character, and the
script reports how often that matches the text, over running words and over
distinct words, and counts both letters by what precedes them.

Typed text uses ꯢ almost only after ꯥ, ꯣ or ꯨ, and there often writes ꯏ
instead. So the script also sorts documents (Wikipedia pages, parquet rows,
JSON lines, whole text files) by how they spell the i after ꯥ, ꯣ or ꯨ: always
ꯢ, always ꯏ, or mixed. The documents that always write ꯢ there are candidate
text in the thesis spelling; their statistics are reported separately.
"""

import argparse
import bz2
import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

I_LETTER = "\uABCF"  # ꯏ, TUMMHCD class 025
I_LONSUM = "\uABE2"  # ꯢ, TUMMHCD class 044
LONSUM = set("\uABDB\uABDC\uABDD\uABDE\uABDF\uABE0\uABE1\uABE2")  # ꯛ ꯜ ꯝ ꯞ ꯟ ꯠ ꯡ ꯢ
NUNG_APUN = {"\uABEA", "\uABED"}  # nung ꯪ, apun (the vowel killer)
NON_VOWEL = LONSUM | NUNG_APUN
VOWEL_SIGNS = set("\uABE3\uABE4\uABE5\uABE6\uABE7\uABE8\uABE9")  # ꯣ ꯤ ꯥ ꯦ ꯧ ꯨ ꯩ
VOWEL_LETTERS = set("\uABCE\uABCF\uABD1")  # ꯎ ꯏ ꯑ
AA_O_U = set("\uABE5\uABE3\uABE8")  # ꯥ ꯣ ꯨ: where typed text uses ꯢ at all
WORD = re.compile("[\uABC0-\uABEA\uABEC\uABED\uAAE0-\uAAF6]+")
MEITEI_ANY = re.compile("[\uABC0-\uABFF\uAAE0-\uAAFF]")
TEXT_SUFFIXES = {".txt", ".xml", ".text", ".csv", ".tsv"}


def open_text(path):
    name = path.name.lower()
    if name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_texts(path):
    """Yields (document, text chunk) from a file or every readable file in a folder.

    A document is a parquet row, a JSON line, a <page> of an XML dump, or a whole text file.
    """
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        name = f.name.lower()
        base = name[:-4] if name.endswith(".bz2") else name[:-3] if name.endswith(".gz") else name
        if base.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(f)
            cols = ["text"] if "text" in df.columns else [c for c in df.columns if df[c].dtype == object]
            for c in cols:
                for row, value in enumerate(df[c]):
                    if isinstance(value, str):
                        yield f"{f.name}#{row}", value
        elif base.endswith(".jsonl") or base.endswith(".json"):
            with open_text(f) as fh:
                for row, line in enumerate(fh):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        texts = [rec["text"]] if isinstance(rec.get("text"), str) else \
                            [v for v in rec.values() if isinstance(v, str)]
                        for text in texts:
                            yield f"{f.name}#{row}", text
        elif Path(base).suffix in TEXT_SUFFIXES or path.is_file():
            xml = Path(base).suffix == ".xml"
            page = 0
            with open_text(f) as fh:
                for line in fh:
                    if xml and "<page>" in line:
                        page += 1
                    yield (f"{f.name}#page{page}" if xml else f.name), line


def context(prev):
    """What kind of character precedes a ꯢ or ꯏ (None: the word starts there)."""
    if prev is None:
        return "word start"
    if prev in LONSUM:
        return "final consonant (lonsum)"
    if prev in NUNG_APUN:
        return "nung or apun"
    if prev in VOWEL_SIGNS:
        return "vowel sign"
    if prev in VOWEL_LETTERS:
        return "vowel letter"
    if "\uABC0" <= prev <= "\uABDA":
        return "consonant letter"
    return "other"


def i_after_aa_o_u(words):
    """(ꯢ, ꯏ) counts right after ꯥ, ꯣ or ꯨ in a word counter."""
    lonsum = letter = 0
    for word, n in words.items():
        for i in range(1, len(word)):
            if word[i - 1] in AA_O_U:
                if word[i] == I_LONSUM:
                    lonsum += n
                elif word[i] == I_LETTER:
                    letter += n
    return lonsum, letter


def rule_prediction(word, i):
    prev = word[i - 1] if i > 0 else None
    return I_LETTER if prev is None or prev in NON_VOWEL else I_LONSUM


class Stats:
    def __init__(self):
        self.words = Counter()
        self.nonspace_chars = 0
        self.meitei_chars = 0
        self.chunks = 0
        self.doc_words = defaultdict(Counter)
        self.consistent_i_lonsum = Counter()  # words of the documents that always write ꯢ after ꯥ/ꯣ/ꯨ

    def add(self, text, doc=None):
        text = unicodedata.normalize("NFC", text)
        self.chunks += 1
        self.nonspace_chars += sum(1 for ch in text if not ch.isspace())
        self.meitei_chars += len(MEITEI_ANY.findall(text))
        words = WORD.findall(text)
        self.words.update(words)
        if doc is not None:
            self.doc_words[doc].update(words)

    def rule(self, by_type=False):
        """Confusion counts {actual: {predicted: n}} for ꯢ / ꯏ, over running words or distinct words."""
        table = {I_LETTER: Counter(), I_LONSUM: Counter()}
        for word, n in self.words.items():
            weight = 1 if by_type else n
            for i, ch in enumerate(word):
                if ch in table:
                    table[ch][rule_prediction(word, i)] += weight
        total = sum(sum(c.values()) for c in table.values())
        correct = sum(table[ch][ch] for ch in table)
        return {"occurrences": total,
                "rule_accuracy": round(correct / total, 4) if total else None,
                "actual_i_letter": dict(table[I_LETTER]), "actual_i_lonsum": dict(table[I_LONSUM]),
                "i_lonsum_per_i_letter": (round(sum(table[I_LONSUM].values()) / sum(table[I_LETTER].values()), 3)
                                          if table[I_LETTER] else None)}

    def contexts(self, top=40):
        """Counts of ꯢ and ꯏ by what precedes them, over running words: by kind and by character."""
        kinds, chars = defaultdict(Counter), defaultdict(Counter)
        for word, n in self.words.items():
            for i, ch in enumerate(word):
                if ch in (I_LETTER, I_LONSUM):
                    prev = word[i - 1] if i > 0 else None
                    kinds[context(prev)][ch] += n
                    chars["word start" if prev is None else f"U+{ord(prev):04X} {prev}"][ch] += n

        def rows(table, limit=None):
            items = sorted(table.items(), key=lambda kv: -sum(kv[1].values()))[:limit]
            return [{"before": k, "i_lonsum": c[I_LONSUM], "i_letter": c[I_LETTER]} for k, c in items]
        return {"by_kind": rows(kinds), "by_character": rows(chars, top)}

    def documents(self, min_i=5, bins=10):
        """Documents sorted by how they spell the i after ꯥ, ꯣ or ꯨ: always ꯢ (90% or more), always ꯏ
        (10% or less), or mixed; those with fewer than min_i such i's cannot tell."""
        groups = {"consistent_i_lonsum": [], "consistent_i_letter": [], "mixed": [], "too_few_to_tell": []}
        hist = Counter()
        for doc, words in self.doc_words.items():
            lonsum, letter = i_after_aa_o_u(words)
            if lonsum + letter < min_i:
                groups["too_few_to_tell"].append(doc)
                continue
            share = lonsum / (lonsum + letter)
            hist[min(int(share * bins), bins - 1)] += 1
            groups["consistent_i_lonsum" if share >= 0.9 else "consistent_i_letter" if share <= 0.1 else "mixed"
                   ].append(doc)
        out = {"documents": len(self.doc_words), "min_i_after_aa_o_u": min_i,
               "share_of_i_lonsum_after_aa_o_u": {f"{k / bins:.1f}-{(k + 1) / bins:.1f}": hist[k]
                                                  for k in range(bins)}}
        for key, docs in groups.items():
            out[key] = {"documents": len(docs),
                        "running_words": sum(sum(self.doc_words[d].values()) for d in docs)}
        self.consistent_i_lonsum = Counter()
        for d in groups["consistent_i_lonsum"]:
            self.consistent_i_lonsum.update(self.doc_words[d])
        if self.consistent_i_lonsum:
            subset = Stats()
            subset.words = self.consistent_i_lonsum
            out["consistent_i_lonsum_subset"] = {
                "distinct_words": len(subset.words),
                "rule_running_words": subset.rule(), "rule_distinct_words": subset.rule(by_type=True),
                "i_contexts_running_words": subset.contexts(top=20)}
        return out

    def summary(self, top=30):
        chars = Counter()
        for word, n in self.words.items():
            for ch in word:
                chars[ch] += n
        running = sum(self.words.values())
        return {
            "chunks": self.chunks,
            "running_words": running,
            "distinct_words": len(self.words),
            "meitei_letters_in_words": sum(chars.values()),
            "share_of_nonspace_chars_in_meitei_block": (round(self.meitei_chars / self.nonspace_chars, 4)
                                                        if self.nonspace_chars else None),
            "mean_word_length": round(sum(chars.values()) / running, 3) if running else None,
            "char_frequencies": {f"U+{ord(c):04X} {c}": n for c, n in chars.most_common()},
            "top_words": [[w, n] for w, n in self.words.most_common(top)],
            "rule_running_words": self.rule(by_type=False),
            "rule_distinct_words": self.rule(by_type=True),
            "i_contexts_running_words": self.contexts(),
            **({"by_document": self.documents()} if self.doc_words else {}),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="name=path")
    ap.add_argument("--out", default="results/corpus_stats.json")
    ap.add_argument("--words-out", help="folder for one word-frequency file per source (word<TAB>count)")
    args = ap.parse_args()

    report, combined = {}, Stats()
    for spec in args.sources:
        name, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            report[name] = {"path": path, "error": "not found"}
            print(f"{name}: {path} not found, skipped")
            continue
        s = Stats()
        for doc, text in iter_texts(path):
            s.add(text, doc)
        combined.words.update(s.words)
        combined.chunks += s.chunks
        combined.nonspace_chars += s.nonspace_chars
        combined.meitei_chars += s.meitei_chars
        report[name] = {"path": path, **s.summary()}
        r = report[name]
        print(f"{name}: {r['running_words']:,} words, {r['distinct_words']:,} distinct, "
              f"ꯢ/ꯏ rule {r['rule_running_words']['rule_accuracy']} "
              f"(ꯢ per ꯏ {r['rule_running_words']['i_lonsum_per_i_letter']})")
        d = r.get("by_document")
        if d:
            print(f"  documents always writing ꯢ after ꯥ/ꯣ/ꯨ: {d['consistent_i_lonsum']['documents']:,} "
                  f"({d['consistent_i_lonsum']['running_words']:,} words); always ꯏ: "
                  f"{d['consistent_i_letter']['documents']:,}; mixed: {d['mixed']['documents']:,}; "
                  f"too few to tell: {d['too_few_to_tell']['documents']:,}")
        if args.words_out:
            out = Path(args.words_out)
            out.mkdir(parents=True, exist_ok=True)
            for suffix, words in (("", s.words), ("_consistent_i_lonsum", s.consistent_i_lonsum)):
                if words:
                    (out / f"{name}{suffix}.tsv").write_text(
                        "".join(f"{w}\t{n}\n" for w, n in words.most_common()), encoding="utf-8")
    report["all_sources_combined"] = {"note": "sources can overlap (FineWeb-2 contains Wikipedia pages)",
                                      **combined.summary()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
