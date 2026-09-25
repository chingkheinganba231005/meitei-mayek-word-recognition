# Phase 1: synthetic words

Started 24 September 2026. Status (25 September): the code, tests and Colab notebook are
ready; the owner ran the notebook on TUMMHCD (section 5), and the syllable rule (section 2.7)
and the draws for rare letters (section 3) followed from it, and the owner chose the sizes
measured on TUMMHCD and a wider pen range. Next: a check that the words look handwritten,
with real characters (section 5), then the notebook is run again to regenerate the fixed
synthetic sets.

## 1. What the synthesiser does

Package `mayek_words`; one word image takes 5–7 ms on one CPU core.

1. **Text.** A word in everyday spelling (ꯢ written ꯏ), drawn from the lexicon (section 3),
   or a number in Meitei Mayek digits (3%); a full stop (cheikhei) follows 2% of words.
2. **Characters.** For each character, a TUMMHCD image of its class, chosen to match the
   word's style (section 2.5).
3. **Layout**, in units of L, the height of a letter. Letters, lonsum letters and digits
   stand on the baseline one after another; a sign is placed relative to the pen position
   after the character before it, as the font places it (section 2.3). Then the gaps on the
   line are set together, keeping every syllable together (section 2.7). Sizes, gaps,
   positions and the baseline are jittered.
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
(`results/glyph_sizes_tummhcd.json`). The measured sizes are used (the owner's choice
after a side-by-side comparison, 25 September 2026; a copy is packaged as
`assets/glyph_sizes_tummhcd.json`), clipped to 0.5–2 times the font's, with the font's
positions; `--sizes font` gives the printed sizes.

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
| gap between letters that do not join, added to the font's side bearings | −0.10 to +0.05 L, sd 0.03 L per gap |
| chance that a character joins the one before it (ink touching), inside a syllable | 2–50%; between syllables half that |
| words spaced unevenly by syllable (inside narrower, between wider) | 1 in 5, by 0–0.12 L |
| size of the signs relative to print | 0.85–1.3 |
| pen width | 0.06–0.14 L (TUMMHCD's scanned strokes about 0.07; photos look thicker; the owner's choice) |
| slant | normal, sd 0.12 (tan of the angle), clipped at 2.5 sd |
| rotation | normal, sd 1.5 degrees, clipped at 2.5 sd |
| baseline drift | 0.04 L per character, smoothed |
| blur | Gaussian sigma 0–0.8 px |
| paper, ink | paper 225–255; ink from the chosen images; noise sd 3 grey levels |

These are guesses to be checked on the contact sheets and, later, against the real set.
Stronger augmentation (backgrounds, lighting, perspective) belongs in Phase 2, on the GPU.

**Spacing.** The owner, a native writer, observed that handwritten Meitei Mayek sets its
letters closer together than the first previews did, and supplied screenshots of published
handwriting found by web image search (used only for this measurement; the images are not
kept). `scripts/measure_spacing.py` finds the pieces of ink in an image, the letter height
L, and the gaps between neighbouring letters; letters that touch form one piece, and their
number is estimated from the width of the pieces. On six samples (one page twice, as
photographed and as binarised, and four pages on ruled paper;
`results/spacing_web_samples.json`), a median of 33.5% of neighbouring letters touch
(14–73% per sample; the 73% is inflated by fragments of the ruled lines), and the letters
that do not touch are 0.097 L apart (median of the samples' medians; 0.05–0.18).

So letters either join or keep a small gap. The synthesiser now does the same: each word
has a chance of 5–65% that a letter joins the letter before it, and a joined letter is slid
left until its ink meets the ink before it (touching, not overlapping). The other letters
keep the font's side bearings plus −0.10 to +0.05 L. Measured the same way on 30 pages of
synthetic words written with the font's characters (`results/spacing_synthetic_font.json`):
33% touching, visible gaps 0.086 L (at these resolutions one pixel is about 0.03 L). The
first default (gaps of 0.02–0.3 L on top of the side bearings) spaced letters 0.28 L apart
by the layout, more than twice the font's 0.12 L. The notebook repeats the check with
TUMMHCD characters (`results/spacing_synthetic_tummhcd.json`): how readily two letters
join depends on their shapes.

**2.7 Syllables.** On the contact sheets of the first run, the owner saw ꯤ drawn closer to
the letter after it than to its own consonant. Measured with the font's characters, for ꯤ
followed by another syllable, the first layout put it nearer the next letter in 52% of cases
and joined it to the next letter alone in 35%: the word's gap and the joins applied to the
gap after a sign but not to the gap before it. In the owner's rules a syllable starts at a
letter (a consonant, or a cluster joined by apun), takes a vowel sign as its nucleus, and may
close with nung or a lonsum letter (`charset.syllables`; an i right after ꯥ, ꯣ or ꯨ also
closes it, like the ꯢ of the standard spelling). Handwriting is usually spaced evenly; when
it is not, the gaps follow the syllables, and nobody writes a vowel sign closer to the next
letter than to its own.

So the layout now sets all the gaps on a line together. Spacing is even by default: ꯤ, when
touching neither neighbour, sits halfway (median ratio of its two gaps 1.00). A gap inside a
syllable is never wider than the gaps around it. In 1 word in 5 the syllables stand apart.
A character joins the one before it with the word's chance inside a syllable and half that
between syllables, and a join between syllables joins their insides too. The ꯤ cases above
drop to 0%. The rule covers every sign beside a letter (ꯤ, ꯦ, ꯣ, ꯧ) and lonsum codas
(confirmed by the owner, 25 September 2026). Letters stay as close as before. The wider pen
made them touch more often (41%), so the chance of joining is now 2–50% per word; measured
like for like on 30 synthetic pages with the final settings
(`results/spacing_synthetic_font.json`), 33% of neighbouring letters touch and the others
are 0.086 L apart (the owner's screenshots: 33.5% and 0.097 L).

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

**Rare letters.** Some letters are rare in text: in the training words of the first run,
ꯘ is 478 of about 5.4 million characters (0.009%), ꯓ 934 and ꯙ 1,243
(`results/lexicon_stats.json`). Each is still a character the recogniser must read, and
ꯗ/ꯘ is one of the confusable pairs. So 10% of the draws go to the rare characters (under
0.5% of all characters): one of them is picked at random, then a word containing it.
On the Phase 0 word list available in development (7,810 words) this raises ꯘ about
17-fold, to 0.3% of characters, and barely changes the common ones (ꯤ 9.6% to 9.5%); the
next lexicon run records the shares with and without these draws (`sampling_train`).
Confirmed by the owner, 25 September 2026.

## 4. Checks done so far

- Layout against the font's own shaping of every sign and of two-sign combinations
  (section 2.3), and the size method on the font's characters (section 2.4).
- The whole notebook on a fake archive (font characters, distorted, in TUMMHCD's layout)
  with Colab stubbed out: the split rule (15% of every class, seed 42), the duplicate
  exclusion, the size check, lexicon, contact sheets, fixed sets.
- 37 tests (`pytest -q`), among them positions of every kind of sign, spacing and joining, the
  spacing tool on synthetic pages, pen width, the split
  and its stability, duplicate exclusion, and that the committed priors are exactly what
  `scripts/glyph_priors.py` writes.

## 5. First run on TUMMHCD (owner, 24–25 September 2026)

- **Character stores** (`results/glyph_store_stats.json`): 61,475 training, 10,820
  validation and 12,325 test images (29, 6 and 469 left out); the fewest per class are
  506 training images of cheikhei.
- **Sizes** (`results/glyph_sizes_tummhcd.json`): the method's check on the font is as in
  development. Letters and lonsum letters are written about as printed (median width 0.98
  and height 0.99 times the font's). Signs are written larger: height 2.4 times the font's for ꯨ, 1.5 for
  ꯩ, 1.4 for ꯧ, 1.3 for ꯪ, 1.2 for ꯦ and ꯥ; ꯤ is 1.4 times as wide. Strokes are about
  0.07 L across.
- **First-paper note:** ꯦ is written 0.47 L tall and ꯰ 0.87 L, so in the 24 x 24 images
  the strokes of ꯦ are about twice as thick: that is what the size features see.
- **Spacing** with TUMMHCD characters (first layout): 35.4% of neighbouring letters touch,
  visible gaps 0.094 L, as in real handwriting (33.5%, 0.097 L). To be repeated with the
  syllable layout.
- **Lexicon** (`results/lexicon_stats.json`): three sources (FLORES+ was not downloaded),
  77,580 distinct words, 76,066 kept (272 dropped for characters outside the alphabet, 98
  for apun, 30 for starting with a sign, 4 for length); 68,450 training, 3,855 validation
  and 3,761 test words. ꯤ is the most frequent character (10.9%).
- **Fixed sets**: 5,000 words each, 6.2 ms per word, made with the first layout: to be
  regenerated (their files, and the spacing check, are not committed).

Decided afterwards (owner, 25 September 2026): sizes measured on TUMMHCD, after a
side-by-side comparison with the font's; pen width 0.06–0.14 L, after a demonstration from
0.05 to 0.17 L (TUMMHCD strokes about 0.07 L; the owner's screenshots about 0.17 L, inflated
by blur). Next: the owner noted that some characters in the development previews look
typed. Those previews use the font's characters, since TUMMHCD is on the owner's Drive; the
words must be checked with real characters, including whether redrawing every stroke at an
even width makes them look machine-made.

Later: a check of the style matching (can a classifier tell matched words from randomly
mixed ones?), and the real-set prompts kept out of training.
