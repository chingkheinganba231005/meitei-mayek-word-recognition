"""A character n-gram language model of words, for decoding.

Interpolated Kneser-Ney with a fixed discount (0.75 by default, as counts may be weighted):

    P(c | h) = max(N(h c) - D, 0) / N(h) + D * T(h) / N(h) * P(c | h minus its first token)

N are the counts of the highest order and, below it, continuation counts (the number of
different characters seen before h c); T(h) is the number of different characters seen
after h. The unigram step is interpolated with the uniform distribution, so every
character keeps some probability. A word is a sequence of characters followed by the
end symbol, with the context padded by start symbols.

The test words are not in the training word list (the lexicon is split by word), so the
model is open-vocabulary: it helps through the syllables and endings it has seen.
``training_counts`` makes its training text like the words the synthesiser writes: the
training lexicon's words weighted by count ** power, plus numbers and words followed by a
full stop in the synthesiser's shares.
"""

import math
import pickle
from collections import defaultdict

import numpy as np

from mayek_words.charset import CHEIKHEI

BOS, EOS = "<s>", "</s>"


class CharLM:
    def __init__(self, order=6, discount=0.75):
        self.order, self.discount = order, discount
        self.counts, self.context, self.vocab = {}, {}, []
        self._cache = {}

    # ------------------------------------------------------------------ fitting

    def fit(self, weights):
        """weights: {word: weight > 0}."""
        N = self.order
        top = defaultdict(float)
        vocab = {EOS}
        for word, wt in weights.items():
            toks = [BOS] * (N - 1) + list(word) + [EOS]
            vocab.update(word)
            for k in range(N - 1, len(toks)):
                top[(tuple(toks[k - N + 1:k]), toks[k])] += wt
        self.counts = {N: dict(top)}
        for n in range(N - 1, 0, -1):
            cont = defaultdict(float)
            for h, w in self.counts[n + 1]:
                cont[(h[1:], w)] += 1.0
            self.counts[n] = dict(cont)
        self.context = {}
        for n, table in self.counts.items():
            ctx = defaultdict(lambda: [0.0, 0])
            for (h, _), c in table.items():
                ctx[h][0] += c
                ctx[h][1] += 1
            self.context[n] = {h: (tot, types) for h, (tot, types) in ctx.items()}
        self.vocab = sorted(vocab)
        self._cache = {}
        return self

    # ------------------------------------------------------------------ scoring

    def _p(self, w, h, n):
        D = self.discount
        if n == 1:
            tot, types = self.context[1][()]
            return max(self.counts[1].get(((), w), 0.0) - D, 0.0) / tot + D * types / tot / len(self.vocab)
        hh = h[len(h) - (n - 1):]
        lower = self._p(w, h, n - 1)
        stats = self.context[n].get(hh)
        if stats is None:
            return lower
        tot, types = stats
        return max(self.counts[n].get((hh, w), 0.0) - D, 0.0) / tot + D * types / tot * lower

    def logprob(self, w, history):
        """Natural log P(w | history); w is a character or EOS, history the characters before."""
        h = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        h = (BOS,) * (self.order - 1 - len(h)) + h
        key = (w, h)
        if key not in self._cache:
            if len(self._cache) > 2_000_000:
                self._cache.clear()
            self._cache[key] = math.log(self._p(w, h, self.order))
        return self._cache[key]

    def word_logprob(self, word):
        """Natural log P(word), end symbol included."""
        return sum(self.logprob(c, word[:i]) for i, c in enumerate(word)) + self.logprob(EOS, word)

    def perplexity(self, words, weights=None):
        """Per-character perplexity (end symbols counted) over a list of words."""
        weights = np.ones(len(words)) if weights is None else np.asarray(weights, float)
        lp = sum(w * self.word_logprob(x) for x, w in zip(words, weights))
        n = sum(w * (len(x) + 1) for x, w in zip(words, weights))
        return float(math.exp(-lp / n))

    # ------------------------------------------------------------------ files

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"order": self.order, "discount": self.discount, "counts": self.counts,
                         "context": self.context, "vocab": self.vocab}, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        lm = cls(d["order"], d["discount"])
        lm.counts, lm.context, lm.vocab = d["counts"], d["context"], d["vocab"]
        return lm


def training_counts(counts, power=0.5, numbers=0.03, stop=0.02, n_numbers=5000, seed=0):
    """Training weights for the LM: {word: count ** power}, plus random numbers (share
    `numbers` of the total weight) and words followed by a full stop (share `stop`), as
    in the synthetic words (``mayek_words.synth.Words``)."""
    from mayek_words.lexicon import number

    rng = np.random.default_rng(seed)
    out = {w: float(c) ** power for w, c in counts.items()}
    total = sum(out.values())
    for _ in range(n_numbers):
        x = number(rng)
        out[x] = out.get(x, 0.0) + numbers * total / n_numbers
    words = list(counts)
    p = np.array([out[w] for w in words])
    picks = rng.choice(len(words), size=min(n_numbers, len(words)), replace=False, p=p / p.sum())
    for k in picks:
        out[words[k] + CHEIKHEI] = out.get(words[k] + CHEIKHEI, 0.0) + stop * total / len(picks)
    return out
