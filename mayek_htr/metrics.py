"""Scoring recognised words against the truth.

- CER: edit distance (substitutions + deletions + insertions) summed over all words,
  divided by the number of reference characters.
- WER: the share of words not recognised exactly (the sets hold single words); word
  accuracy is 1 - WER.
- Confusable pairs: the pairs isolated-character recognition confuses most (the first
  paper: 046/009 ꯦ/꯰, 047/011 ꯨ/ꯁ, 033/034 ꯗ/ꯘ; ꯢ/ꯏ is one letter now). For each pair,
  of the reference characters of the pair, the share read correctly and how many were
  read as the other member, from an edit-distance alignment of each word.
"""

from collections import Counter

import numpy as np

CONFUSABLE = (("ꯦ", "꯰"), ("ꯨ", "ꯁ"), ("ꯗ", "ꯘ"))


def align(ref, hyp):
    """Levenshtein alignment -> [(reference character or '', recognised character or '')]."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out.append((ref[i - 1], ""))
            i -= 1
        else:
            out.append(("", hyp[j - 1]))
            j -= 1
    return out[::-1]


def distance(ref, hyp):
    return sum(r != h for r, h in align(ref, hyp))


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 5), round(float(c + h), 5))


def score(refs, hyps, pairs=CONFUSABLE, top=20):
    """Lists of reference and recognised texts -> a JSON-friendly summary."""
    assert len(refs) == len(hyps)
    chars = exact = subs = dels = ins = 0
    ref_count, correct, swaps = Counter(), Counter(), Counter()
    for ref, hyp in zip(refs, hyps):
        chars += len(ref)
        exact += ref == hyp
        for r, h in align(ref, hyp):
            if r:
                ref_count[r] += 1
            if r and h:
                if r == h:
                    correct[r] += 1
                else:
                    subs += 1
                    swaps[(r, h)] += 1
            elif r:
                dels += 1
                swaps[(r, "")] += 1
            else:
                ins += 1
                swaps[("", h)] += 1
    n = len(refs)
    edits = subs + dels + ins
    out = {"words": n, "characters": chars,
           "cer": round(edits / max(chars, 1), 5),
           "wer": round(1 - exact / max(n, 1), 5),
           "word_accuracy": round(exact / max(n, 1), 5), "word_accuracy_95ci": wilson(exact, n),
           "substitutions": subs, "deletions": dels, "insertions": ins,
           "pairs": {}}
    for a, b in pairs:
        k = ref_count[a] + ref_count[b]
        ok = correct[a] + correct[b]
        out["pairs"][f"{a}/{b}"] = {"reference": k, "correct": ok,
                                    "accuracy": round(ok / k, 5) if k else None,
                                    "read_as_the_other": swaps[(a, b)] + swaps[(b, a)]}
    out["top_errors"] = [[r or "(inserted)", h or "(deleted)", c] for (r, h), c in swaps.most_common(top)]
    per = {ch: round(1 - correct[ch] / ref_count[ch], 4) for ch in ref_count if ref_count[ch] >= 20}
    out["worst_characters"] = sorted(([ch, e, ref_count[ch]] for ch, e in per.items()), key=lambda r: -r[1])[:top]
    return out
