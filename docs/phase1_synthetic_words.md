# Phase 1: synthetic words

Started 24 September 2026. **Status (25 September): done.** The owner ran the notebook on
TUMMHCD four times (section 5) and approved the contact sheet of the fourth run. The first run
led to the syllable rule (section 2.7), the draws for rare letters and the words built from
syllables (section 3), the sizes measured on TUMMHCD, a wider pen range and a fix for pale
writing (section 2.8); the second to signs placed on the ink; the third to a minimum stroke
contrast. The fixed synthetic validation and test sets of the fourth run are on Drive
(`WORK/synth/{val,test}.tar`, settings in `results/synth_{val,test}.json`). Next: Phase 2.

## 1. What the synthesiser does

Package `mayek_words`; one word image takes 4–11 ms on one CPU core.

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
   width (resizing scales strokes with the box), faint strokes included. A sign beside its
   letter (ꯤ, ꯦ, ꯣ, ꯧ) is never nearer the next letter nor touched by it, and never goes into
   its letter; ꯤ is placed on the ink, close to its letter in unevenly spaced words; signs
   above and below are placed on the ink at their print distance (section 2.7). The
   word is slanted, rotated a little, blurred a little, and drawn in ink on paper.

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
| chance that a character joins the one before it (ink touching), inside a syllable | 2–60% (2–50% until the signs' rule, section 2.7); between syllables half that |
| words spaced unevenly by syllable (inside narrower, between wider) | 1 in 5, by 0–0.12 L |
| ꯤ beside its letter, on the ink | even words: the layout's gap; uneven words: 0.02–0.06 L, touching in 1 in 10 |
| ꯦ, ꯣ, ꯧ beside their letter | where the layout puts them |
| any of these signs | never into its letter; the next letter never touching it, never nearer it than the sign is to its letter (after ꯤ in uneven words: 0.04–0.08 L further, plus the syllable spacing) |
| ꯤ's height | from its letter's foot to about 0.1 L above its letter's top (sized like a letter) |
| sign above or below its letter (ꯥ, ꯩ, ꯪ, ꯨ, apun), on the ink | as far from the letter's ink as in print (about 0.08, 0.09, 0.16 L), spread 0.05 L, at least 0.02 L; never into a hollow of the letter |
| ink | from the chosen images, but a stroke's centre (after blur) at least 130 grey levels darker than the paper |
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
synthetic words written with the font's characters: 33% touching, visible gaps 0.086 L (at these resolutions one pixel is about 0.03 L). The
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
like for like on 30 synthetic pages with these settings (built words included), 35% of
neighbouring letters touched and the others were 0.084 L apart (the owner's screenshots: 33.5% and 0.097 L).

*Signs on the ink (after the second run, 25 September 2026).* The owner still saw ꯤ closer
to the letter after it. The rule above works on boxes: it kept ꯤ from being nearer the next
letter than its own, so with even spacing ꯤ sat halfway; and a handwritten ꯤ is not its box.
TUMMHCD's ꯤ starts with a long lead-in stroke from the left, and its stem, which the eye
reads as the sign, stands in the middle of the image (ꯧ has a lead-in too). Measured on the
ink (`scripts/check_signs.py`; the owner's validation characters and the development word
list, words where the sign is followed by another syllable; `results/sign_placement_dev_val.json`),
the body of ꯤ (the sign without its lead-in, `synth.sign_body`) was nearer the next letter
than its own in 73% of cases, and its centre of ink in 58%; for ꯦ, ꯣ and ꯧ the body was
nearer the next letter in 42–47%.

A first fix drew every sign against its letter. The owner confirmed the direction and
corrected the rule (25 September 2026): writing close to the left is what happens when
spacing is uneven, not in most writing, which is even; touching or very close signs are rare;
a sign stuck to the letter on its right never happens: it is always closer to its own
letter, or even; signs must not merge with their letter. Looking at a sheet of even against
uneven words, the owner found ꯤ right in uneven words except one where it nearly merged with
ꯅ, and asked not to apply the close placement to ꯦ, ꯣ and ꯧ, whose even spacing looked good.

The rule now, on the ink (distances are the shortest between pieces of ink, from the sign's
body):

- *ꯤ, evenly spaced words (4 in 5):* its body keeps the layout's gap from its letter (at
  least a pixel of paper).
- *ꯤ, unevenly spaced words (1 in 5):* its body is 0.02–0.06 L from its letter (almost
  stuck; touching in 1 of these words in 10), and the next letter is 0.04–0.08 L further
  from it than it is from its letter, plus the word's syllable spacing.
- *ꯦ, ꯣ, ꯧ:* where the layout puts them, in every word; moved only to keep the rules below.
  Slid on the ink like ꯤ, they tucked over the shoulder of their letter.
- *Every sign:* the next letter keeps the layout's gap but is never nearer the sign than the
  sign is to its own letter, and never touches it. A sign never goes into its letter: no ink
  of the sign may have the letter's ink above and below it in the same column (ꯤ inside the
  open side of ꯅ was the near-merge; passing over the letter is allowed), a lead-in may meet
  the letter's ink in the same row but not cross it (a touching ꯤ may reach 0.1 L past it),
  and no part of a sign starts more than 0.15 L left of the letter's rightmost ink (0.3 L for
  ꯤ in uneven words).

Result on the same words (`results/sign_placement_dev_val.json`), before and after:

| | before | after, even words | after, uneven words |
|---|---|---|---|
| sign's body nearer the next letter (ꯤ; ꯦ, ꯣ, ꯧ) | 73%; 42–47% | 0% | 0% |
| next letter touching the sign | 16–19% | 0% | 0% |
| sign touching its letter | | 0% | ꯤ 1%; others 0% |
| sign crossing into its letter's columns | | 0% | ꯤ 8%; others 0% |
| ꯤ: body to its letter, sign to the next letter (medians) | 0.28 L, 0.16 L | 0.19 L, 0.23 L | 0.18 L, 0.30 L |

By the cruder centre of ink, ꯤ is nearer the next letter in 16% of even words (from 75%) and
4% of uneven ones: with even spacing the centre sits near the middle, and a letter's far edge
can reach past it. Signs no longer join the letter after them, so the chance that a letter
joins the one before it is raised from 2–50% to 2–60% per word, which keeps a third of
neighbouring letters touching: 34% with the validation characters, visible gaps 0.059 L; 32%
and 0.061 L with the font's characters (`results/spacing_synthetic_font.json`; real samples
33.5% and 0.097 L). A word's even and uneven versions now draw the same characters (the
word's style draws the uneven spacing whether it is used or not), so the two can be compared.
Tests check the rule in even and uneven words (and fail on the previous synthesiser) and the
test for going into a letter; the notebook writes the check on the full lexicon
(`results/sign_placement_tummhcd.json`).

*Heights (owner, 25 September 2026).* On the same sheet the owner saw ꯤ placed too low in
one word and ꯩ too high above ꯇ in another: the height of a sign matters as much as its
spacing. Two causes. ꯤ was sized like the small signs (the word's sign scale, 0.85–1.3), so
with a small scale it ended below the top of its letter (0.71 L against 0.90 L); it is now
sized like a letter and runs from its letter's foot to a little above its letter's top (as
in TUMMHCD, 0.1 L). Signs above a letter stood at a fixed height (1.08 L for ꯩ), whatever the
letter's own height; ꯇ is short (0.92 L in print, 0.82 L in that word), so ꯩ floated 0.28 L
above it. Signs above and below a letter (ꯥ, ꯩ, ꯪ, ꯨ, apun) are now placed on the ink: moved
up or down until their ink is as far from the letter's ink as in print (about 0.08 L above,
0.09 L below, 0.16 L for apun; per sign a spread of 0.05 L, at least 0.02 L), never into a
hollow or cup of the letter (``synth.nests``: no letter ink between the sign and its far
side in a column, nor on both sides of it in a row). On the development words, before and
after (`results/sign_placement_dev_val.json`):

| sign | gap to its letter before: median, p10–p90 | touching | over 0.2 L | after: median, p10–p90 | touching | over 0.2 L |
|---|---|---|---|---|---|---|
| ꯥ | 0.14 L, 0.00–0.27 | 10% | 28% | 0.09 L, 0.03–0.15 | 1% | 1% |
| ꯩ | 0.16 L, 0.03–0.33 | 7% | 40% | 0.10 L, 0.04–0.16 | 0% | 1% |
| ꯪ | 0.12 L, −0.03–0.28 | 25% | 26% | 0.07 L, 0.03–0.14 | 4% | 3% |
| ꯨ | 0.14 L, 0.03–0.29 | 5% | 26% | 0.10 L, 0.04–0.17 | 1% | 4% |
| apun | 0.20 L, 0.11–0.29 | 0% | 50% | 0.18 L, 0.11–0.26 | 0% | 35% |

*Faint strokes (owner, 25 September 2026).* In one word ꯡ was partly missing. That scan has
dark vertical strokes and pale top and bottom bars (grey 184–220, lighter than its ink
threshold of 169): the pen step redraws a character from its pixels over 0.5, and the bars
were under it. The pen step now keeps faint strokes (over 0.2) that reach the dark ones
across their soft rim, and still drops faint specks on their own and the rim itself
(``synth.strokes``). On 4,000 validation characters, those with more than 10% of their strokes
missing after the pen step drop from 1.5% to 0.5%, more than 5% from 4.6% to 2.6%
(`scripts/check_strokes.py`, `results/pen_strokes_dev_val.json`).

*Pale words (owner, 25 September 2026).* On the contact sheet of the third run the owner
found everything placed right, but some characters looked as if they were disappearing. They
were words drawn in pale ink: a word takes the ink of its chosen images, the palest TUMMHCD
scans have ink at grey 167 (5% of images), the paper is 225–255, and blur of up to 0.8 px on
a thin pen lowers a stroke's centre further. On 1,000 words with the validation characters,
10.2% had their darkest strokes less than 100 grey levels below the paper, 2.3% less than 80.
The centre of a stroke, after blur (estimated from the pen width and the blur), is now kept at
least 130 grey levels darker than the paper (`Config.min_contrast`): it darkens 20% of words and
leaves the others as they were; under 100 grey levels 0.5%, under 80 none
(`scripts/check_strokes.py --lexicon`, `results/pen_strokes_dev_val.json`). A floor of 110
changed too little (9% of words), 150 flattened the ink of a third of them.

**2.8 Pale writing.** On previews with real characters the owner saw a few characters
disappearing. The cause was the pen step: it redraws each character from the pixels of its
ink map over 0.5, and the map was scaled so that the darkest 5% of the ink was 1. On a pale
scan with a few dark specks, most of the stroke fell under 0.5 and was erased: on the
10,820 validation characters, 11.4% lost more than 30% of their ink and 1.3% more than half
(worst: pale ꯄ, ꯥ, ꯌ, ꯆ, ꯭, ꯪ, ꯲, ꯹). The map is now 0 on paper, 0.5 half-way between the
last grey level the image's Otsu threshold counts as ink and the first it counts as paper,
and 1 from a typical ink pixel on, so what the scan shows as ink stays ink: no character
loses more than 30%.
TUMMHCD's own pale scans are drawn in the word's ink like any other. The size measurement
uses the same maps, so the sizes move a little (median width 1.03 times the old
measurement, height 1.01; single classes up to 13%): the next run re-measures them, the
contact sheet and fixed sets use the new measurement, and the packaged copy is updated.

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

**Variety of words (the owner, 25 September 2026).** The owner asked that training cover
short and long words and every kind of syllable: a letter alone with its inherent vowel
(C, ꯀ), letter and vowel sign (CV, ꯀꯥ), letter, vowel sign and final (CVC, ꯀꯥꯡ), and
letter and final (CC, ꯀꯝ) (`charset.split_syllables`, `charset.syllable_type`). Words
drawn from text have all of them, unevenly: on the development word list, 7.7% of drawn
words have one syllable, 1.4% six and 0.4% seven or more; 7.7% of syllables are CC. The
proposal: 15% of training words are composed of real syllables from the lexicon
(`lexicon.SyllableBank`), with 1 to 6 syllables and the kind of each syllable drawn
evenly (at most 6: longer words are rare in writing, the owner's judgement; in text, the
words of 7 or more syllables are mostly loanwords with endings, such as ꯑꯥꯔꯀꯦꯌꯣꯂꯣꯖꯤꯀꯦꯜ,
and long verb forms). On the development list this gives 9.1% one-syllable words and 3.6%
six-syllable words; words of 7 or more syllables stay at their share in text (0.4%);
CC syllables rise to 10.8%. The owner confirmed it, and to keep the long words of text; it
is the default (`Words(built=0.15)`, `render_words.py --built`), and the next lexicon run
records the mix with and without it (`structure_train`).

## 4. Checks done so far

- Layout against the font's own shaping of every sign and of two-sign combinations
  (section 2.3), and the size method on the font's characters (section 2.4).
- The whole notebook on a fake archive (font characters, distorted, in TUMMHCD's layout)
  with Colab stubbed out: the split rule (15% of every class, seed 42), the duplicate
  exclusion, the size check, lexicon, contact sheets, fixed sets.
- 49 tests (`pytest -q`), among them positions of every kind of sign, spacing and joining,
  signs nearer their own letter on the ink, heights of signs above and below, hollows of a
  letter, faint strokes, layout boxes in letter heights, the spacing tool on synthetic pages,
  pen width, the split
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
by blur). The owner then noted that some characters in the development previews looked
typed: those previews used the font's characters. With the owner's real validation
characters (`val.npz`) the words look handwritten, as the owner's contact sheets did. Next:
run the notebook again (sizes re-measured with the fixed ink maps, fixed sets regenerated).

**Second run (owner, 25 September 2026)**, with the fixed ink maps, measured sizes, wider pen,
syllable rule, rare-letter draws and built words:

- **Character stores**: as in the first run.
- **Sizes** (`results/glyph_sizes_tummhcd.json`, now also the packaged default): over all
  classes, median width 0.95 and height 1.02 times the font's; signs a little larger than in
  the first run (height over the font's: ꯨ 2.45, ꯩ 1.59, ꯧ 1.49, ꯦ 1.35, ꯪ 1.34, ꯥ 1.25;
  ꯤ 1.50 times as wide); strokes 0.084 L across (0.07 before: pale strokes are no longer
  thinned). The check on the font's characters: heights within 3%, median width 0.94.
- **Spacing** with TUMMHCD characters and the syllable layout
  (`results/spacing_synthetic_tummhcd.json`): 32.9% of neighbouring letters touch, visible
  gaps 0.061 L (real handwriting 33.5%, 0.097 L).
- **Lexicon** (`results/lexicon_stats.json`): as in the first run. Rare characters (under
  0.5%): ꯉ ꯓ ꯘ ꯙ ꯚ ꯞ ꯪ; with the rare draws ꯘ goes from 0.017% to 0.23% of drawn
  characters, ꯓ from 0.055% to 0.28%, ꯙ from 0.05% to 0.26%. Word structure of drawn
  training words, text alone and with 15% built words: one syllable 3.9% and 5.8%, six
  4.4% and 6.2%, seven or more 2.0% and 1.7%; CC syllables 7.8% and 10.4%. The syllable
  bank holds 178 C, 604 CV, 2,209 CVC and 452 CC syllables.
- **Fixed sets**: 5,000 words each, about 10 ms per word; made before signs were placed on
  the ink, so they are regenerated in the next run.
- **Contact sheet**: the owner found it good except that ꯤ still sat nearer the next letter
  (section 2.7, signs on the ink).

**Third run (owner, 25 September 2026)**, with signs on the ink and heights: the owner found all
placements right; some characters on the contact sheet looked as if they were disappearing (pale
words; section 2.7, pale words). Spacing with TUMMHCD characters: 32.7% of neighbouring letters
touch, visible gaps 0.06 L (`results/spacing_synthetic_tummhcd.json`). Sizes, lexicon and stores
as before. The gap statistics of the fixed sets (`ink_gaps_in_L` in `synth_*.json`) were wrong for
signs beside a letter: a variable in the new placement code overwrote the sign's box in the
recorded layout with pixels (the images were right); fixed, with a test, and the sets are
regenerated in the next run.

**Fourth run (owner, 25 September 2026): approved.** "I am pretty satisfied with the contact
sheet." Compared with the third run's sheet, only the eleven palest words changed (darker; for
example the darkest 2% of one went from grey 167 to 121); every other pixel is the same. On the
full lexicon and the training characters (`results/sign_placement_tummhcd.json`; 300 words per
sign, seed 1000):

- Signs beside a letter: the body nearer the next letter 0% and the next letter touching the
  sign 0%, for ꯤ, ꯦ, ꯣ and ꯧ, in even and uneven words. ꯤ touches its own letter in 0.3% of
  cases (1.4% of uneven words) and crosses into it in 0.6%; its body is a median 0.19 L from
  its letter and 0.24 L from the next (uneven words: 0.17 and 0.29 L). ꯦ, ꯣ, ꯧ never touch or
  cross their letter. By the cruder centre of ink, ꯤ is nearer the next letter in 7% of cases.
- Signs above and below: median gap to the letter 0.085–0.10 L (ꯥ, ꯩ, ꯪ, ꯨ), touching 0–1.3%,
  over 0.2 L 1–3%; apun 0.18 L (in print 0.16 L).
- Contrast (`results/pen_strokes_tummhcd_val.json`, validation characters and lexicon, 1,000
  words): strokes less than 100 grey levels below the paper 0.6% of words (9.7% without the
  minimum contrast), less than 80 none (2.6%); characters missing over 10% of their strokes after
  the pen step 0.55% (1.45% with the old cut at 0.5).
- Spacing (`results/spacing_synthetic_tummhcd.json`): 32.4% of neighbouring letters touch,
  visible gaps 0.059 L (real handwriting 33.5%, 0.097 L).
- Fixed sets (`results/synth_{val,test}.json`): 5,000 words each, about 10 ms per word.
  Their `ink_gaps_in_L` are gaps between the characters' boxes (the image frames as placed), not
  between their ink: signs beside a letter are placed on the ink, and ꯤ's lead-in makes its box
  overlap its neighbours where the ink does not touch, so their "touching" there (36% for the
  sign and the next letter) counts overlapping boxes, not ink (0%, above). From now on
  `render_words.py` calls them `box_gaps_in_L`, with "overlapping", and adds a note.

Later: a check of the style matching (can a classifier tell matched words from randomly
mixed ones?), and the real-set prompts kept out of training.
