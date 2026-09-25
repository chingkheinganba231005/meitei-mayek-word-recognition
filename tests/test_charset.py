import unicodedata

from mayek_words import charset as cs

K, LAI, NA_LONSUM, DIL = chr(0xABC0), chr(0xABC2), chr(0xABDF), chr(0xABD7)
ATIYA, YENAP, ANAP = chr(0xABD1), chr(0xABE6), chr(0xABE5)

# class names of the first project's mayek.charset, as Unicode names
EXPECTED = (["DIGIT " + d for d in "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ZERO".split()]
            + ["LETTER " + n for n in ("KOK SAM LAI MIT PA NA CHIL TIL KHOU NGOU THOU WAI YANG HUK UN I PHAM "
                                       "ATIYA GOK JHAM RAI BA JIL DIL GHOU DHOU BHAM").split()]
            + [f"LETTER {n} LONSUM" for n in "KOK LAI MIT PA NA TIL NGOU I".split()]
            + ["VOWEL SIGN " + n for n in "ANAP YENAP UNAP INAP CHEINAP ONAP SOUNAP NUNG".split()]
            + ["CHEIKHEI", "APUN IYEK"])


def test_table_follows_tummhcd_classes():
    assert [unicodedata.name(ch).replace("MEETEI MAYEK ", "") for ch in cs.TUMMHCD] == EXPECTED
    assert cs.TUMMHCD[25] == cs.I_LETTER and cs.TUMMHCD[44] == cs.I_LONSUM


def test_alphabet_merges_the_two_i():
    assert len(cs.ALPHABET) == 54 and cs.I_LONSUM not in cs.ALPHABET
    assert cs.IMAGE_CLASSES[cs.I_LETTER] == (25, 44)
    assert cs.normalise(K + ANAP + cs.I_LONSUM) == K + ANAP + cs.I_LETTER
    assert not cs.renderable(K + cs.I_LONSUM) and cs.renderable(K + cs.I_LETTER)


def test_syllables_follow_the_owners_rules():
    AA, NG, NUNG, I, HUK, INAP, RAI, BA = (chr(0xABE5), chr(0xABE1), chr(0xABEA), cs.I_LETTER, chr(0xABCD),
                                          chr(0xABE4), chr(0xABD4), chr(0xABD5))
    assert cs.syllables(K) == [0]                                     # onset, inherent vowel
    assert cs.syllables(K + AA) == [0, 0]                             # onset, long a
    assert cs.syllables(K + NUNG) == [0, 0]                           # nung closes it
    assert cs.syllables(K + AA + NG) == [0, 0, 0]                     # lonsum closes it
    assert cs.syllables(HUK + INAP + NG + HUK + NUNG) == [0, 0, 0, 1, 1]  # two syllables
    assert cs.syllables(DIL + cs.APUN + RAI) == [0, 0, 0]             # apun: one onset cluster
    assert cs.syllables(K + AA + I + BA) == [0, 0, 0, 1]              # an i after aa ends the vowel
    assert cs.syllables(K + LAI) == [0, 1]                            # a bare letter is a syllable
    assert cs.syllables(chr(0xABF1) + chr(0xABF2)) == [0, 1]          # digits stand alone


def test_problems():
    assert cs.problem(ATIYA + YENAP + NA_LONSUM + cs.APUN + DIL) is None  # "and": lonsum + apun + letter
    assert cs.problem(K + cs.APUN + LAI) is None
    assert cs.problem(YENAP + K) == "starts with a sign"
    assert cs.problem(K + cs.APUN) == "apun not between two consonants"
    assert cs.problem(K + cs.APUN + cs.APUN + LAI) == "apun not between two consonants"
    assert cs.problem(K + "a") == "character outside the alphabet"
    assert cs.problem(K + chr(0xABEC)) == "character outside the alphabet"  # lum iyek: not in TUMMHCD
