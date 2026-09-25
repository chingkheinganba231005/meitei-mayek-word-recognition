"""Words to write: the Phase 0 word lists in everyday spelling, split by word.

Input: word-frequency lists (word<TAB>count), as written by ``scripts/corpus_stats.py
--words-out`` (one per source, on Drive after the Phase 0 run). Each word is normalised
(ꯢ written ꯏ) and kept if ``charset.problem`` finds nothing wrong with it and it has at
most MAX_LEN characters. Counts from several sources are combined by taking the largest:
FineWeb-2 contains Wikipedia pages, so adding would count those twice.

Split: by word, never by occurrence, so that no word is in two splits. A word's split
depends only on the word (a hash), so it stays the same when sources are added: 90%
train, 5% validation, 5% test. Real-test-set prompts (Phase 3) can be kept out of
training with ``exclude``.

Sampling: a word is drawn with probability proportional to count ** alpha (alpha = 0.5
by default), between running text (alpha = 1) and a plain word list (alpha = 0). Some
letters are rare in text (ꯘ is 0.009% of the characters, ꯓ and ꯙ about 0.02%), yet each
is a character the recogniser must read (ꯗ/ꯘ is a confusable pair). So a share of the
draws (rare_share, 10% by default) goes to the rare characters, those under rare_below
(0.5%) of all characters: one of them is picked at random, then a word containing it.
"""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from .charset import ALPHABET, CHEIKHEI, DIGITS, SYLLABLE_TYPES, normalise, problem, split_syllables, syllable_type

MAX_LEN = 24
SPLITS = (("train", 0.90), ("val", 0.95), ("test", 1.0))


def split_of(word):
    """'train', 'val' or 'test', from a hash of the (normalised) word."""
    h = int.from_bytes(hashlib.sha256(("mayek-words|" + word).encode("utf-8")).digest()[:8], "big") / 2 ** 64
    return next(name for name, upper in SPLITS if h < upper)


def read_counts(path):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, n = line.rstrip("\n").partition("\t")
            if word and n.strip().isdigit():
                counts[word] += int(n)
    return counts


def write_counts(counts, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{w}\t{n}\n" for w, n in counts.most_common()), encoding="utf-8")


def clean(counts, max_len=MAX_LEN):
    """Normalise and filter -> (kept counts, dropped {reason: [distinct words, running words]})."""
    kept, dropped = Counter(), {}
    for word, n in counts.items():
        w = normalise(word)
        why = problem(w) or (f"longer than {max_len} characters" if len(w) > max_len else None)
        if why:
            d = dropped.setdefault(why, [0, 0])
            d[0] += 1
            d[1] += n
        else:
            kept[w] += n  # two spellings of one word become one
    return kept, dropped


def combine(sources):
    """{name: Counter} -> Counter with the largest count of each word over the sources."""
    out = Counter()
    for counts in sources.values():
        for w, n in counts.items():
            if n > out[w]:
                out[w] = n
    return out


def split(counts, exclude=()):
    exclude = set(exclude)
    parts = {name: Counter() for name, _ in SPLITS}
    for w, n in counts.items():
        s = split_of(w)
        if s == "train" and w in exclude:
            continue
        parts[s][w] = n
    return parts


def describe(counts):
    running = sum(counts.values())
    lengths = np.repeat([len(w) for w in counts], list(counts.values())) if counts else np.zeros(1)
    chars = Counter()
    for w, n in counts.items():
        for ch in w:
            chars[ch] += n
    total = sum(chars.values()) or 1
    return {"distinct_words": len(counts), "running_words": running,
            "length_running": {"mean": round(float(lengths.mean()), 2),
                               "p50": int(np.percentile(lengths, 50)), "p95": int(np.percentile(lengths, 95)),
                               "max": int(lengths.max())},
            "characters_running": {ch: {"count": chars[ch], "share": round(chars[ch] / total, 5)}
                                   for ch in ALPHABET if ch not in DIGITS},
            "top_words": counts.most_common(20)}


class Lexicon:
    """Words with sampling weights count ** alpha; with rare_share, that share of the draws
    is a word containing one of the rare characters (see the module notes)."""

    def __init__(self, counts, alpha=0.5, rare_share=0.1, rare_below=0.005):
        self.words = list(counts)
        w = np.array([counts[x] for x in self.words], np.float64) ** alpha
        self.cum = np.cumsum(w / w.sum())
        self.rare = {}
        if rare_share > 0 and self.words:
            weight, total = Counter(), float((w * np.array([len(x) for x in self.words])).sum())
            for word, wt in zip(self.words, w):
                for ch in set(word):
                    weight[ch] += wt * word.count(ch)
            for ch in ALPHABET:
                if ch not in DIGITS and ch != CHEIKHEI and 0 < weight[ch] / total < rare_below:
                    idx = np.array([i for i, word in enumerate(self.words) if ch in word])
                    self.rare[ch] = (idx, np.cumsum(w[idx] / w[idx].sum()))
        self.rare_chars = sorted(self.rare)
        self.rare_share = rare_share if self.rare else 0.0

    @classmethod
    def load(cls, path, alpha=0.5, rare_share=0.1):
        return cls(read_counts(path), alpha, rare_share)

    def __len__(self):
        return len(self.words)

    def sample(self, rng):
        if self.rare_share and rng.random() < self.rare_share:
            idx, cum = self.rare[self.rare_chars[int(rng.integers(len(self.rare_chars)))]]
            return self.words[int(idx[min(int(np.searchsorted(cum, rng.random(), side="right")), len(idx) - 1)])]
        return self.words[min(int(np.searchsorted(self.cum, rng.random(), side="right")), len(self.words) - 1)]


class SyllableBank:
    """The syllables of the lexicon's words, by kind (C, CV, CVC, CC; ``charset.syllable_type``),
    weighted as the words are. ``word`` composes a word of real syllables: its number of
    syllables and the kind of each drawn evenly, so that short and long words and every
    kind of syllable are well represented, whatever the text's own mix. At most 6 syllables:
    longer words are rare in writing (the owner, 25 September 2026)."""

    def __init__(self, lexicon):
        weights = np.diff(np.concatenate([[0.0], lexicon.cum]))
        bank = {t: Counter() for t in SYLLABLE_TYPES}
        for word, wt in zip(lexicon.words, weights):
            for s in split_syllables(word):
                t = syllable_type(s)
                if t in bank:
                    bank[t][s] += wt
        self.kinds = [t for t in SYLLABLE_TYPES if bank[t]]
        self.bank = {t: (list(bank[t]), np.cumsum(np.array(list(bank[t].values())) / sum(bank[t].values())))
                     for t in self.kinds}

    def word(self, rng, syllables=(1, 6)):
        out = []
        for _ in range(int(rng.integers(syllables[0], syllables[1] + 1))):
            names, cum = self.bank[self.kinds[int(rng.integers(len(self.kinds)))]]
            out.append(names[min(int(np.searchsorted(cum, rng.random(), side="right")), len(names) - 1)])
        return "".join(out)


def structure(words):
    """Share of words by number of syllables, and of syllables by kind, in a list of words."""
    n_syl, kinds, has = Counter(), Counter(), Counter()
    for w in words:
        parts = split_syllables(w)
        n_syl["7+" if len(parts) >= 7 else str(len(parts))] += 1
        ks = [syllable_type(p) for p in parts]
        kinds.update(k for k in ks if k in SYLLABLE_TYPES)
        has.update(set(k for k in ks if k in SYLLABLE_TYPES))
    n, total = max(len(words), 1), max(sum(kinds.values()), 1)
    return {"syllables_per_word": {k: round(n_syl[k] / n, 4) for k in ["1", "2", "3", "4", "5", "6", "7+"]},
            "syllable_kinds": {k: round(kinds[k] / total, 4) for k in SYLLABLE_TYPES},
            "words_containing_kind": {k: round(has[k] / n, 4) for k in SYLLABLE_TYPES}}


def sampled_shares(lexicon, n=200000, seed=0):
    """Share of each character among the characters of n drawn words."""
    rng = np.random.default_rng(seed)
    chars = Counter()
    for _ in range(n):
        chars.update(lexicon.sample(rng))
    total = sum(chars.values())
    return {ch: chars[ch] / total for ch in ALPHABET if ch not in DIGITS and ch != CHEIKHEI}


def number(rng, max_digits=4):
    """A number in Meitei Mayek digits, 1 to max_digits long, not starting with zero (unless 0)."""
    digits = sorted(DIGITS)  # ꯰ ꯱ ... ꯹
    n = int(rng.integers(1, max_digits + 1))
    first = digits[int(rng.integers(1, 10))] if n > 1 else digits[int(rng.integers(0, 10))]
    return first + "".join(digits[int(rng.integers(0, 10))] for _ in range(n - 1))
