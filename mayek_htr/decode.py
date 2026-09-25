"""Turning the recogniser's per-frame class probabilities into text.

``greedy``: the best class of every frame, repeats merged, blanks dropped.
``beam_search``: CTC prefix beam search (Hannun et al. 2014) with a character language
model: a prefix's score is its CTC log probability plus alpha * the language model's log
probability of its characters (and of the word ending there) plus beta per character.
alpha and beta are chosen on the validation set. Frames are pruned to the classes with
log probability over `prune` (at most `topk` of them), which keeps it fast.
``segments``: for a word already cut into characters (the isolated-character baseline):
one class per character, the sum of the classifier's log probabilities plus alpha * the
language model's.
"""

import math

import numpy as np

from .labels import BLANK, decode
from .lm import EOS

NEG = -math.inf


def _lse(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def greedy(logp):
    """logp: (T, C) log probabilities -> text."""
    best = np.asarray(logp).argmax(1)
    ids, prev = [], BLANK
    for k in best:
        if k != prev and k != BLANK:
            ids.append(int(k))
        prev = k
    return decode(ids)


def beam_search(logp, lm=None, alpha=0.5, beta=0.0, beam=16, prune=-10.0, topk=8):
    """logp: (T, C) natural log probabilities of one word -> text."""
    logp = np.asarray(logp, np.float64)
    beams = {(): (0.0, NEG)}                      # prefix -> (log P ending in blank, ... in a character)
    lm_cache = {}

    def lm_score(c, prefix):
        if lm is None or alpha == 0:
            return beta
        key = (c, prefix)
        if key not in lm_cache:
            text = decode(prefix)
            lm_cache[key] = alpha * lm.logprob(decode([c]), text) + beta
        return lm_cache[key]

    for t in range(len(logp)):
        row = logp[t]
        cand = np.flatnonzero(row > prune)
        cand = cand[cand != BLANK]
        if len(cand) > topk:
            cand = cand[np.argsort(-row[cand])[:topk]]
        new = {}

        def add(prefix, pb, pnb):
            ob, onb = new.get(prefix, (NEG, NEG))
            new[prefix] = (_lse(ob, pb), _lse(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = _lse(pb, pnb)
            add(prefix, total + row[BLANK], NEG)                     # a blank: the prefix stays
            if prefix:
                add(prefix, NEG, pnb + row[prefix[-1]])               # the last character again
            for c in cand:
                c = int(c)
                ext = prefix + (c,)
                via = pb if prefix and c == prefix[-1] else total      # a repeat needs a blank between
                add(ext, NEG, via + row[c] + lm_score(c, prefix))
        beams = dict(sorted(new.items(), key=lambda kv: -_lse(*kv[1]))[:beam])

    def final(item):
        prefix, (pb, pnb) = item
        s = _lse(pb, pnb)
        if lm is not None and alpha:
            s += alpha * lm.logprob(EOS, decode(prefix))
        return s

    best = max(beams.items(), key=final)[0]
    return decode(best)


def segments(logp, lm=None, alpha=0.5, beam=16, topk=5):
    """logp: (n, len(ALPHABET)) natural log probabilities of the n characters of a segmented
    word, columns in alphabet order -> text."""
    from .labels import ALPHABET

    logp = np.asarray(logp, np.float64)
    if lm is None or alpha == 0:
        return "".join(ALPHABET[k] for k in logp.argmax(1))
    beams = [(0.0, "")]
    for row in logp:
        cand = np.argsort(-row)[:topk]
        beams = sorted(((s + row[k] + alpha * lm.logprob(ALPHABET[k], text), text + ALPHABET[k])
                        for s, text in beams for k in cand), key=lambda b: -b[0])[:beam]
    return max(beams, key=lambda b: b[0] + alpha * lm.logprob(EOS, b[1]))[1]
