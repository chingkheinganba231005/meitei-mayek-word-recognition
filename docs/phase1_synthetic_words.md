# Phase 1: synthetic words

Started 24 September 2026. Status: the code, tests and Colab notebook are ready and have
been checked on the font's own characters and on a fake archive; the first run on TUMMHCD
(`notebooks/phase1_synthetic_words.ipynb`, run by the owner) is pending. No number in this
note comes from TUMMHCD yet, except those taken from Phase 0.

## 1. What the synthesiser does

Package `mayek_words`; one word image takes 5–7 ms on one CPU core.

1. **Text.** A word in everyday spelling (ꯢ written ꯏ), drawn from the lexicon (section 3),
   or a number in Meitei Mayek digits (3%); a full stop (cheikhei) follows 2% of words.
2. **Characters.** For each character, a TUMMHCD image of its class, chosen to match the
   word's style (section 2.5).
3. **Layout**, in units of L, the height of a letter. Letters, lonsum letters and digits
   stand on the baseline one after another, with a gap; a sign is placed relative to the
   pen position after the character before it, as the font places it (section 2.3). Sizes,
   gaps, positions and the baseline are jittered.
4. **Drawing.** Each 24 x 24 image is resized into its box, which gives back the
   proportions TUMMHCD lost, and its strokes are thickened or thinned to the word's pen
   width (resizing scales strokes with the box). The word is slanted, rotated a little,
   blurred a little, and drawn in ink on paper.

A seed fixes an image: item i of a set is always the same word and the same image.

## 2. Decisions

**2.1 Alphabet.** 54 characters: the 55 TUMMHCD classes without ꯢ (Phase 0 decision:
everyday spelling). ꯏ is drawn with images of both 025 (ꯏ) and 044 (ꯢ), which cannot be
told apart. Characters outside TUMMHCD (lum iyek ꯬, the Meetei Mayek Extensions) cannot be
drawn; words containing them are left out.

**2.2 Which character images.** The first paper's split, made by its own code
(`mayek.split`, pinned to commit 0d2c6e5): 61,504 train, 10,826 validation, 12,794 test.

| Synthetic set | Character images | Words |
|---|---|---|
| training | TUMMHCD train without our validation part (what the pretrained networks trained on) | training lexicon |
| validation | our validation part | validation lexicon |
| test | TUMMHCD test without the 469 images that have a train twin | test lexicon |

The 48 images in groups of identical pixels with conflicting labels are used nowhere. The
synthetic test set is a check, not the evaluation: that is the Phase 3 real set.

**2.3 Proportions and placement come from a font.** TUMMHCD lost every character's size,
proportions and place in the word (all images are the ink box stretched to 24 x 24). They
are taken from Noto Sans Meetei Mayek Regular 2.002 (SIL Open Font Licence 1.1; built from
its source, github.com/notofonts/meetei-mayek at commit e562454, and bundled in
`mayek_words/assets` with its licence). `scripts/glyph_priors.py` shapes every character
with HarfBuzz, alone or after each of the 27 letters, and measures its ink box
(`assets/glyph_priors.json`). In units of L:

| Character | Where | Box (w x h) |
|---|---|---|
| letters, lonsum letters, digits | on the baseline, 0 to 1 | 0.7–1.4 x 1 |
| ꯥ anap | above the right half of the character before it (1.08–1.42) | 0.45 x 0.34 |
| ꯩ cheinap | above (1.08–1.53) | 0.58 x 0.45 |
| ꯪ nung | above (1.07–1.57); above a sign if one precedes it | 0.36 x 0.50 |
| ꯦ yenap | beside, at the top (0.62–1.00) | 0.41 x 0.38 |
| ꯣ onap | beside, at the top (0.62–1.28) | 0.44 x 0.65 |
| ꯧ sounap | beside, at the top (0.74–1.27) | 0.58 x 0.53 |
| ꯤ inap | beside, from the baseline (0–1.27) | 0.47 x 1.27 |
| ꯨ unap | below the right half (−0.30 to −0.09) | 0.52 x 0.21 |
| ꯭ apun | under the whole letter before it (−0.27 to −0.16) | the letter's width x 0.11 |

Positions are relative to the pen, as the font does it, which also covers two signs on
one letter (ꯀꯥꯪ, ꯀꯨꯪ, ꯀꯣꯪ). With all jitter off, the synthesiser reproduces the font's
own rendering of every sign and of these combinations (`tests/test_synth.py` checks the
positions).

**2.4 Handwritten sizes, measured on TUMMHCD.** Print is only a prior: people may write
signs larger or smaller. The stretch to 24 x 24 leaves a trace: a pen of width p becomes
p x 24 / w pixels across a vertical stroke and p x 24 / h across a horizontal one. With the
same pen for every character, the thicknesses give back each class's width and height
relative to a letter (`mayek_words/sizes.py`; thickness from horizontal and vertical ink
chords). On the font's own characters, stretched the same way, it recovers the heights
within 4% (median ratio 1.00, 10th–90th percentile 0.96–1.01) and the widths within about
15% (median 0.95, 0.86–1.07); it gives no estimate where a character has no strokes in a
direction (apun, cheikhei). `scripts/glyph_sizes.py` runs it on TUMMHCD
(`results/glyph_sizes_tummhcd.json`); `--sizes` uses the measured sizes instead of the
font's (clipped to 0.5–2 times the font's), keeping the font's positions. Which to use is
decided on the contact sheets.

**2.5 Style matching instead of writers.** TUMMHCD has no writer information (Phase 0).
Each image gets four style features, standardised within its class: slant, stroke width in
the frame, ink fraction, ink darkness. A word takes the style of a random image as its
anchor, and each character is one of the 16 images of its class closest to the anchor.
The pen width is then made the same for the whole word (each character re-drawn from the
signed distance to its stroke edge), and the ink darkness is the mean of the chosen images.

**2.6 Word style.** Per word, drawn uniformly unless stated (`synth.Config`):

| Parameter | Range |
|---|---|
| letter height L | 32 px, log-normal jitter sd 0.1 |
| letter width factor | 0.8–1.25 (log-uniform) |
| gap between characters, added to the font's side bearings | −0.12 to +0.04 L, sd 0.03 L per gap |
| size of the signs relative to print | 0.85–1.3 |
| pen width | 0.07–0.12 L (TUMMHCD letters: 2.2 px in 24, about 0.09) |
| slant | normal, sd 0.12 (tan of the angle), clipped at 2.5 sd |
| rotation | normal, sd 1.5 degrees, clipped at 2.5 sd |
| baseline drift | 0.04 L per character, smoothed |
| blur | Gaussian sigma 0–0.8 px |
| paper, ink | paper 225–255; ink from the chosen images; noise sd 3 grey levels |

These are guesses to be checked on the contact sheets and, later, against the real set.
Stronger augmentation (backgrounds, lighting, perspective) belongs in Phase 2, on the GPU.

**Spacing.** The owner, a native writer, observed that handwritten Meitei Mayek sets its
letters closer together than the first previews did. Measured on the layout (the ink gap
between neighbours on the line, in units of L; `synth.line_gaps`): the font spaces letters
0.12 L apart (median; 10–90%: 0.08–0.16), the first default 0.28 L (0.15–0.41), more than
twice as far. The default is now tighter than print: 0.08 L (0.01–0.16), with 5% of
neighbouring letters touching. Every rendered set records these gaps in its `config.json`
(`ink_gaps_in_L`), and a test keeps the default spacing tighter than the font's. The
ranges should be set from real handwriting once some is measured.

## 3. Lexicon

`scripts/build_lexicon.py` reads the Phase 0 word-frequency lists (one per source, on
Drive), writes every word in everyday spelling (ꯢ as ꯏ, so two spellings of a word become
one), and drops what cannot be a written word: characters outside the alphabet, a sign at
the start, apun that does not join two consonants, more than 24 characters. A consonant
next to apun may be a lonsum letter: typed loanwords join a final consonant to the next
letter (ꯑꯦꯟ꯭ꯗ "and", ꯕꯨꯛ꯭ꯁ "books", ꯂꯤꯁ꯭ꯠ "list"). On the word list of the always-ꯢ
Wikipedia pages (Phase 0; 7,810 words) this drops 2 words, both typing debris (eight apuns in
a row; a final apun). Counts from several sources are combined by the largest, not the sum
(FineWeb-2 contains Wikipedia pages).

Split by word, never by occurrence: a hash of the word puts 90% of words in training, 5% in
validation and 5% in test, the same whatever the sources. Words are drawn with probability
proportional to count^0.5. The word lists stay on Drive (they derive from CC BY-SA and
ODC-By text); `results/lexicon_stats.json` has their sizes and character frequencies.
Real-test-set prompts (Phase 3) can be kept out of training with `--exclude`.

## 4. Checks done so far

- Layout against the font's own shaping of every sign and of two-sign combinations
  (section 2.3), and the size method on the font's characters (section 2.4).
- The whole notebook on a fake archive (font characters, distorted, in TUMMHCD's layout)
  with Colab stubbed out: the split rule (15% of every class, seed 42), the duplicate
  exclusion, the size check, lexicon, contact sheets, fixed sets.
- 33 tests (`pytest -q`), among them positions of every kind of sign, spacing, pen width, the split
  and its stability, duplicate exclusion, and that the committed priors are exactly what
  `scripts/glyph_priors.py` writes.

## 5. After the Colab run

1. The two contact sheets, read by a native writer: signs in the right place and of the
   right size, spacing, anything no writer would do. Then choose font or measured sizes.
2. `results/glyph_sizes_tummhcd.json`: are signs written larger or smaller than printed?
   Is the method's check on the font as good as here?
3. `results/lexicon_stats.json`: sizes, what was dropped, character frequencies. Rare
   letters (the borrowed ꯓ, ꯙ, ꯚ ...) may need words that contain them drawn more often.
4. `results/glyph_store_stats.json`: images per class after the exclusions.

Later: a check of the style matching (can a classifier tell matched words from randomly
mixed ones?), and the real-set prompts kept out of training.
