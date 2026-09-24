"""Characters of handwritten Meitei Mayek words and the TUMMHCD classes that draw them.

TUMMHCD has 55 classes, numbered as its folders train_000 ... train_054 (the first
project's ``mayek.charset``). Words are written in everyday spelling (Phase 0 decision):
ꯢ (i lonsum, class 44) is not used and every i is ꯏ (class 25). Images of both classes
draw ꯏ, since they cannot be told apart. The recogniser's alphabet therefore has 54
characters: every TUMMHCD class except 44.

The table is built from code points, not typed characters, so that it cannot silently
drift; ``tests/test_charset.py`` checks it against the Unicode names.
"""

import unicodedata

_DIGITS = [0xABF1 + i for i in range(9)] + [0xABF0]      # the dataset orders them 1..9, then 0
_LETTERS = list(range(0xABC0, 0xABDB))                   # 27 letters, kok ... bham
_LONSUM = list(range(0xABDB, 0xABE3))                    # 8 final consonants, kok lonsum ... i lonsum
_SIGNS = [0xABE5, 0xABE6, 0xABE8, 0xABE4, 0xABE9, 0xABE3, 0xABE7, 0xABEA]  # anap yenap unap inap
#                                                        # cheinap onap sounap nung, as in TUMMHCD
_OTHER = [0xABEB, 0xABED]                                # cheikhei (full stop), apun iyek

TUMMHCD = tuple(chr(c) for c in _DIGITS + _LETTERS + _LONSUM + _SIGNS + _OTHER)  # class index -> character
NUM_TUMMHCD = len(TUMMHCD)                                                     # 55

I_LETTER = chr(0xABCF)      # ꯏ, class 25
I_LONSUM = chr(0xABE2)      # ꯢ, class 44: written ꯏ in everyday spelling
APUN = chr(0xABED)
CHEIKHEI = chr(0xABEB)

DIGITS = frozenset(chr(c) for c in _DIGITS)
LETTERS = frozenset(chr(c) for c in _LETTERS)
LONSUM = frozenset(chr(c) for c in _LONSUM)
SIGNS = frozenset(chr(c) for c in _SIGNS)                # vowel signs and nung
MARKS = SIGNS | {APUN}                                   # written above, below or beside a letter

ALPHABET = tuple(ch for ch in TUMMHCD if ch != I_LONSUM)  # the recogniser's 54 characters
CLASS_OF = {ch: i for i, ch in enumerate(TUMMHCD)}
# TUMMHCD classes whose images can draw a character of the alphabet
IMAGE_CLASSES = {ch: (CLASS_OF[ch], CLASS_OF[I_LONSUM]) if ch == I_LETTER else (CLASS_OF[ch],)
                 for ch in ALPHABET}


def normalise(text):
    """NFC, and ꯢ written as ꯏ (everyday spelling)."""
    return unicodedata.normalize("NFC", text).replace(I_LONSUM, I_LETTER)


def renderable(word):
    """True if every character of the word is in the alphabet (after normalise)."""
    return bool(word) and all(ch in CLASS_OF and ch != I_LONSUM for ch in word)


def problem(word):
    """Why a normalised word cannot be a written word, or None.

    Only clear cases: a character outside the alphabet, a sign with no letter to sit on,
    apun that does not join two consonants. A consonant here is a letter or a lonsum
    letter: typed loanwords join a final consonant to the next letter (ꯑꯦꯟ꯭ꯗ "and",
    ꯂꯤꯁ꯭ꯠ "list").
    """
    if not renderable(word):
        return "character outside the alphabet"
    if word[0] in MARKS:
        return "starts with a sign"
    consonants = LETTERS | LONSUM
    for i, ch in enumerate(word):
        if ch == APUN and (i + 1 == len(word) or word[i - 1] not in consonants or word[i + 1] not in consonants):
            return "apun not between two consonants"
    return None
