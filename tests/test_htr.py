import math

import numpy as np
import pytest

from mayek_htr import decode as dec
from mayek_htr import images, labels, metrics
from mayek_htr.lm import EOS, CharLM, training_counts

K, SAM, LAI, MIT = chr(0xABC0), chr(0xABC1), chr(0xABC2), chr(0xABC3)
AA, INAP, YENAP, UNAP = chr(0xABE5), chr(0xABE4), chr(0xABE6), chr(0xABE8)
ZERO, DIL, GHOU = chr(0xABF0), chr(0xABD7), chr(0xABD8)


def test_labels_round_trip():
    text = K + AA + LAI + INAP + chr(0xABE2)          # ends in ꯢ: written ꯏ
    ids = labels.encode(text)
    assert labels.BLANK not in ids and len(ids) == 5 and labels.NUM_CLASSES == 55
    assert labels.decode(ids) == K + AA + LAI + INAP + chr(0xABCF)
    with pytest.raises(ValueError):
        labels.encode("a")
    assert labels.clean(K + chr(0xABEC)) == (K, chr(0xABEC))   # lum iyek cannot be written


def test_image_normalisation():
    gray = np.full((80, 200), 235, np.uint8)
    gray[20:60, 30:40] = 150                           # pale ink on grey paper
    x = images.normalise(gray)
    assert x.dtype == np.uint8 and x.shape == (64, 160)
    assert x.max() == 255 and x[0].max() == 0          # ink full strength, paper 0
    wide = images.normalise(np.full((10, 5000), 255, np.uint8))
    assert wide.shape == (64, images.MAX_WIDTH)
    big = np.full((640, 2000), 240, np.uint8)
    big[100:500, 300:340] = 20
    assert images.normalise(big).shape == (64, 200)    # a photo, shrunk
    batch, widths = images.pad_batch([x, x[:, :50]])
    assert batch.shape == (2, 64, 160) and list(widths) == [160, 50] and batch[1, :, 50:].max() == 0
    crop = images.crop_ink(big)
    assert crop.shape[0] < 640 and crop.shape[1] < 2000 and crop.min() == 20


def test_alignment_and_scores():
    assert metrics.distance("abcd", "abcd") == 0
    assert metrics.distance("abcd", "abd") == 1 and metrics.distance("abc", "xabc") == 1
    assert metrics.align("ab", "b") == [("a", ""), ("b", "b")]
    refs = [K + YENAP, SAM + UNAP, DIL, K]
    hyps = [K + ZERO, SAM + UNAP, GHOU, K + K]
    s = metrics.score(refs, hyps)
    assert s["words"] == 4 and s["characters"] == 6
    assert s["word_accuracy"] == 0.25 and s["wer"] == 0.75
    assert s["cer"] == round(3 / 6, 5) and (s["substitutions"], s["deletions"], s["insertions"]) == (2, 0, 1)
    assert s["pairs"][f"{YENAP}/{ZERO}"] == {"reference": 1, "correct": 0, "accuracy": 0.0, "read_as_the_other": 1}
    assert s["pairs"][f"{UNAP}/{SAM}"]["accuracy"] == 1.0 and s["pairs"][f"{UNAP}/{SAM}"]["reference"] == 2
    assert s["pairs"][f"{DIL}/{GHOU}"]["read_as_the_other"] == 1
    lo, hi = s["word_accuracy_95ci"]
    assert lo < 0.25 < hi


def test_char_lm():
    lm = CharLM(order=4).fit({K + AA: 10, K + AA + LAI: 5, LAI + INAP: 3, MIT: 1})
    # probabilities of every next token sum to one, in seen and unseen contexts
    for hist in ("", K, K + AA, MIT + MIT + MIT):
        total = sum(math.exp(lm.logprob(c, hist)) for c in lm.vocab)
        assert abs(total - 1) < 1e-9
    assert lm.logprob(AA, K) > lm.logprob(INAP, K)                     # seen after ꯀ
    assert lm.word_logprob(K + AA) > lm.word_logprob(AA + K)
    assert lm.logprob(EOS, K + AA) > lm.logprob(EOS, K)
    assert 1 < lm.perplexity([K + AA, LAI + INAP]) < lm.perplexity([MIT + AA + INAP])


def test_lm_training_counts():
    w = training_counts({K + AA: 100, LAI: 4}, power=0.5, n_numbers=50, seed=1)
    assert w[K + AA] == pytest.approx(10.0) and w[LAI] == pytest.approx(2.0)
    base = 12.0
    nums = sum(v for k, v in w.items() if all(0xABF0 <= ord(c) <= 0xABF9 for c in k))
    stops = sum(v for k, v in w.items() if k.endswith(chr(0xABEB)))
    assert nums == pytest.approx(0.03 * base) and stops == pytest.approx(0.02 * base)


def ctc_logp(path, classes=labels.NUM_CLASSES, sharp=6.0, noise=None):
    """Log probabilities of a CTC path given as class ids per frame."""
    x = np.full((len(path), classes), -sharp)
    for t, k in enumerate(path):
        x[t, k] = 0.0
    if noise is not None:
        x = x + noise
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def test_greedy_and_beam_without_lm():
    k, aa = labels.INDEX[K], labels.INDEX[AA]
    lp = ctc_logp([0, k, k, 0, aa, 0, 0, k, 0, k, 0])
    assert dec.greedy(lp) == K + AA + K + K                 # a blank separates the repeated ꯀ
    assert dec.beam_search(lp, beam=8) == K + AA + K + K


def test_beam_search_with_lm_fixes_an_ambiguous_frame():
    lm = CharLM(order=3).fit({K + AA: 50, K + INAP: 1})
    k, aa, inap = labels.INDEX[K], labels.INDEX[AA], labels.INDEX[INAP]
    lp = ctc_logp([0, k, 0, inap, 0])
    lp[3, aa] = lp[3, inap] - 0.3                            # the sign is nearly a tie
    lp[3] -= np.log(np.exp(lp[3]).sum())
    assert dec.greedy(lp) == K + INAP
    assert dec.beam_search(lp, lm=None) == K + INAP
    assert dec.beam_search(lp, lm=lm, alpha=1.0) == K + AA
