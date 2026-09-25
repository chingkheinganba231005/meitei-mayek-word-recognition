"""The recogniser's output alphabet: the 54 characters of ``mayek_words.charset.ALPHABET``
(everyday spelling: ꯢ is written ꯏ) and the CTC blank, class 0. Character i of the
alphabet is class i + 1.

Text is in Unicode order, which for Meitei Mayek follows the writing: a sign above, below
or beside a letter comes after it, and apun comes between the two letters it joins.
"""

from mayek_words.charset import ALPHABET, normalise

BLANK = 0
NUM_CLASSES = len(ALPHABET) + 1          # 55
INDEX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}


def encode(text):
    """Text -> class ids (after ``normalise``). Raises ValueError for a character outside
    the alphabet."""
    ids = []
    for ch in normalise(text):
        if ch not in INDEX:
            raise ValueError(f"character outside the alphabet: {ch!r} (U+{ord(ch):04X}) in {text!r}")
        ids.append(INDEX[ch])
    return ids


def decode(ids):
    """Class ids (no blanks) -> text."""
    return "".join(ALPHABET[i - 1] for i in ids if i != BLANK)


def clean(text):
    """Reference text for scoring: normalised, with the characters the alphabet cannot write
    dropped. Returns (text, dropped characters)."""
    t = normalise(text)
    kept = "".join(ch for ch in t if ch in INDEX)
    return kept, "".join(ch for ch in t if ch not in INDEX)
