# Handwritten Meitei Mayek word recognition

Owner: Chingkheinganba Rajkumar (GitHub chingkheinganba231005, ORCID 0009-0002-2620-0281),
School of Computing and Data Science, The University of Hong Kong.

This file is the project brief. Claude Code reads it at the start of every session, so keep it
up to date: when a decision is made or a number is final, write it here.

## Goal

1. **Word-level recognition (main project).** Read whole handwritten Meitei Mayek words instead of
   isolated characters, using context (a character language model and the script's orthographic
   rules) to fix errors that no isolated-character model can fix.
2. **Handwriting generation (second stage).** A diffusion model that writes Meitei Mayek words in
   different writers' styles. Test whether its output improves the recogniser on real handwriting.

## Where this comes from

The first paper, "Pretrained eyes on an old script: new state of the art in handwritten Meitei
Mayek recognition", is submitted to The Visual Computer (Springer), September 2026, under review.

- Code: https://github.com/chingkheinganba231005/Handwritten-Meitei-Mayek-Recognition (MIT,
  Python package `mayek`, one notebook that runs every experiment, pytest tests)
- Weights and demo: https://huggingface.co/Chingkheinganba/handwritten-meitei-mayek-recognition
- Archive: Zenodo DOI 10.5281/zenodo.22932285
- The LaTeX source of the submitted paper is on the owner's computer
  (Rajkumar_TVC_LaTeX_source_v6.zip). If reviewers ask for a revision, work from that zip.

Key results (TUMMHCD, 55 classes, official test set of 12,794 images):
- Six networks (ConvNeXt-T, EfficientNetV2-S, ResNet-50-D, each also with 5 size features),
  plain average: 98.12% test accuracy, 241 errors. Previous best: 97.08% (multilevel fusion).
- Four pairs cause 148 of the 241 errors: 044/025 (78), 046/009 (33), 011/047 (20), 033/034 (17).
- Size features separate 046/009 and 011/047 (vowel signs sit in the upper/lower zone of a word,
  digits and consonants in the middle zone). But see "Checks for the first paper" below: every
  TUMMHCD image is 24 × 24 with the ink filling the frame.
- 044 (ꯢ, i lonsum, U+ABE2) versus 025 (ꯏ, i, U+ABCF) cannot be separated from isolated images:
  a two-class specialist reaches 68.2% against a 67.1% majority baseline. Reading this pair
  perfectly would lift the ensemble to 98.73%. This pair was the original reason for the word-level
  project; Phase 0 (below) found it is a spelling convention (everyday writing uses ꯏ throughout),
  and the project now reads it as one letter.
- Rule from Hijam's thesis: ꯢ follows a vowel; ꯏ begins a word or does not follow a vowel.
  Checked by language experts on about 26,000 words (from a corpus of about 190,000): 94.7%.
  ꯢ is about 3.6 times as frequent as ꯏ.

## Prior work to beat and cite (check each against the source before citing)

- Hijam & Saharia, TUMMHCD, The Visual Computer 38:525-539 (2022), doi 10.1007/s00371-020-02032-y.
  Dataset: http://agnigarh.tezu.ernet.in/~sarat/resources.html (archive TUMMHCD-TEST-TRAIN.zip,
  train and test folders; our validation split is 15% per class, seed 42, via `mayek.split`).
- Hijam & Saharia, zone and rule assisted recognition, Evolutionary Intelligence 17:2963-2980
  (2024), doi 10.1007/s12065-024-00920-z. Second stage on segmented words using zones and the
  orthographic rule; 100 handwritten words (565 characters), 88.50% to 91.86%.
- Hijam, PhD thesis, Tezpur University (2024),
  http://agnee.tezu.ernet.in:8082/jspui/handle/1994/1707. Includes a CNN combined with a
  character-level LSTM language model on word images. **Our word-level work must be clearly
  different from and stronger than this** (public benchmark, modern sequence model, synthetic data
  at scale, proper real-handwriting test set).
- Naosekpam et al., EMBiL, CAIP 2023, doi 10.1007/978-3-031-44237-7_7 (scene text, English-Manipuri).

## Plan

**Phase 0: literature and data audit (do this first).**
- Search for any word- or line-level handwritten Meitei Mayek recognition since 2020, and for
  synthetic-word and diffusion handwriting work on Indic scripts. Write the novelty statement here
  before building anything.
- Find a Meitei Mayek text corpus we are allowed to use (check licences): the TDIL corpus used in
  the thesis, Meitei-script Wikipedia, AI4Bharat IndicCorp, FLORES-200, Meitei Mayek newspapers.
  Record the source, size and licence of each.
- Check whether TUMMHCD file names or folders carry writer IDs (needed for writer-consistent words).

**Phase 1: synthetic words.** Compose word images from TUMMHCD characters with zone-aware placement
(vowel signs above or below, lonsum finals, realistic spacing and baseline jitter). Where possible
take all characters of one word from the same writer. Split writers so no writer appears in both
training and test data.

**Phase 2: recogniser.** Reuse our pretrained backbones as the visual encoder, add a BiLSTM or
Transformer sequence head with CTC (and try an attention decoder). Decode with a character n-gram or
small language model. Baselines: our isolated-character ensemble plus the orthographic rule, and a
re-implementation of the zone-and-rule second stage. Metrics: CER, WER, and accuracy on the
confusable pairs (ꯢ/ꯏ turned out to be a spelling convention: see Phase 0 findings).

**Phase 3: real test set.** Collect real handwritten words from volunteers with written consent
(what is collected, how it is used and licensed). Target roughly 50+ writers. This set is the main
evaluation; synthetic data is only for training. Releasing it publicly would be a contribution in
itself. **Changed by the owner (26 September 2026): no volunteers; the owner writes the set
alone on an iPad** (see the Phase 3 section below).

**Phase 4: generation (project 2).** Diffusion model conditioned on text and writer style. Evaluate
by training the recogniser on generated data and testing on the real set, plus standard image
metrics.

## Phase 0 findings (first pass, 24 September 2026; details in `docs/phase0_audit.md`)

Found by web search only (no full-text access in that session): every paper still has to be
checked against its full text before citing.

**Novelty statement (draft; (3) and (4) revised and confirmed by the owner, 24 September 2026;
(3) reworded for one writer and then for 100 words, confirmed by the owner, 27 September 2026).**
No published work recognises handwritten Meitei Mayek words or lines end to end. Earlier work
classifies isolated characters, segments handwritten pages into lines and words without
recognising them (Inunganbi, Choudhary, Manglem, The Visual Computer 2020,
doi 10.1007/s00371-020-01799-4: 189 pages, word segmentation 88.96%), or corrects a character
classifier on segmented words with zones and the orthographic rule (Hijam and Saharia 2024;
Hijam's thesis). The only large handwritten Manipuri word dataset, IIIT-Indic-HW-UC (Mondal and
Jawahar, ICPR 2024), is in Bengali script. Meitei Mayek text recognition exists only for print
(NE-OCR 2026 preprint); for scene text there is character recognition, and EMBiL only detects
text and identifies its language. No handwriting generation model exists for the script. We
contribute: (1) the first segmentation-free handwritten Meitei Mayek word recogniser; (2) zone-aware
synthetic words from TUMMHCD at scale; (3) a first real handwritten word set, 100 words by a
native writer on a tablet, text-disjoint from training, with a fixed protocol and the tools to
extend it; (4) evidence that
ꯢ versus ꯏ, the largest error source of isolated-character recognition (78 of 241 errors), is a
spelling convention, not a visual distinction (everyday writing uses ꯏ throughout; the standard
spelling's ꯢ after ꯥ, ꯣ, ꯨ follows a rule for 95–99% of words), so the recogniser reads one letter
and renders either spelling; (5) later, the first diffusion model for Meitei Mayek handwriting.
Context still matters for ꯦ/꯰, ꯨ/ꯁ and ꯗ/ꯘ (70 of the 241 errors).

**IIIT Hyderabad risk: resolved (24 September 2026, from the author PDF).** IIIT-Indic-HW-UC
(doi 10.1007/978-3-031-78495-8_21) writes Manipuri in Bengali script (its Table 1; the Manipuri
word samples in Fig. 3 are Bengali-script handwriting): 101 writers, 200K words, 75,531 distinct,
CRNN baseline 90.99% CRR, 83.38% WRR. No Meitei Mayek, so the statement holds. Take from it:
its split (75/10/15% of word images) is not stated to be writer- or text-disjoint, so ours must be
both and say so; its phone-capture protocol is a template for Phase 3; its CRNN + CTC baseline
(Gongidi et al.) is a Phase 2 baseline. The paper states no licence.

**Corpora (measured, `results/corpus_stats.json`).** Native Unicode Meitei Mayek text is scarce:
Wikipedia 1,036,283 running words (72,437 distinct), FineWeb-2 `mni_Mtei` 52,395 (11,141), its
filtered-out part 365,832 (33,425); together 1.45M words and 77,580 distinct, overlapping.
Bengali-script Manipuri: 2.69M words in FineWeb-2 alone. Licences: Wikipedia CC BY-SA 4.0,
FineWeb-2 ODC-By, FLORES+ `mni_Mtei` dev CC BY-SA 4.0 (gated), IN22 CC BY 4.0 (script to check),
printed-dataset transcriptions CC BY; ILCI-II (TDIL-DC, about 22,000 Meitei Mayek sentences) needs
registration; newspapers need written permission. Transliterated Bengali-script text is for
training only, flagged (its ꯢ/ꯏ choices no longer matter; see below). Real-test-set prompts should
come from a CC BY source, in everyday spelling (ꯏ only).

**ꯢ/ꯏ in typed text (measured, `results/corpus_stats.json`).** Typed text uses ꯢ only after ꯥ,
ꯣ or ꯨ (97–99% of all ꯢ); after a consonant letter (inherent a), the other vowel signs, vowel
letters and at word start it writes ꯏ (98% or more in Wikipedia). For the i after ꯥ, ꯣ or ꯨ there
are two spelling practices, page by page: of 4,523 Wikipedia pages with at least five such i's,
1,615 write ꯢ for fewer than 30% of them and 2,463 for 70% or more (445 in between); web text
outside Wikipedia leans to ꯏ (FineWeb-2: 122 pages against 24). Pages writing ꯢ there 90% of the
time or more (Wikipedia 605 pages, 55,589 words; about 82,000 words over three overlapping
sources) match the thesis: the rule holds for 95–97% of their i's (92–95% over distinct words;
thesis 94.7%) and they have 4–6 ꯢ per ꯏ (thesis 3.6); part of this is built in by the selection,
but ꯏ at word start (1,002 against 14 ꯢ) is not. Over all pages the rule as coded holds for only
28–45%. The ꯢ-writing pages follow a formal standard, the thesis
spelling: under the rule "ꯢ after ꯥ, ꯣ or ꯨ, ꯏ elsewhere" the always-ꯢ Wikipedia pages agree for
95.7% of i's and 94.8% of distinct words (thesis 94.7%), 99.0% without one probably templated
word (ꯃꯆꯥꯈꯥꯏꯕ), with no clear lexical exception (`results/i_exception_summary.json`).
**Owner's note (native writer, 24 September 2026):** "we barely use the lonsum version of i. Its
always ꯑꯥꯏ." **Decision:** transcribe in everyday spelling, one letter ꯏ for every i. The
language-model text is all typed text with ꯢ mapped to ꯏ; the recogniser treats TUMMHCD 044 and
025 as one class; real-test-set prompts and ground truth use ꯏ only. The standard spelling is an
optional rendering of the output by the rule, evaluated on text (the expert review sheet
`results/i_exception_candidates.csv` matters only for that). The word-level motivation now rests
on reading whole words, the other confusable pairs and the real test set, not on ꯢ/ꯏ.
The owner confirmed this reframing and contribution (4) on 24 September 2026.

**TUMMHCD (measured, `results/tummhcd_audit.json`).** No writer information: no sub-folders or
side files; names are `mmhc<class+1>_<running index>`; neighbouring and same-numbered files are
unrelated in style; file times mark scan batches, not writers. All 85,124 images are 24 × 24 px
with the ink filling the frame, so size, aspect ratio and zone are lost. 1,741 groups of
pixel-identical images, almost all pairs (duplicated files, not look-alike glyphs); about a third
of the images of ꯩ, ꯥ, ꯤ, ꯭ and ꯧ sit in such pairs. 469 test images (3.7%) have an identical
train image, 456 with the same label. 24 groups carry conflicting labels, mostly ꯲/꯳ (16) and
꯲/꯹ (4); none ꯢ/ꯏ, so label noise does not explain that pair. Phase 1 therefore uses
style-matched characters instead of one writer per word, a per-class size and zone model (stroke
width in the frame may estimate relative size), and TUMMHCD's own train/test split for the
characters, without the test images that have a train twin; writer-disjoint evaluation comes only
from the Phase 3 real set.

**Checks for the first paper (under review).** (1) Image and ink-box size are (nearly) constant,
so the size probe's separation of 046/009 (86.5%) and 011/047 (84.1%) must come mostly from ink
fraction: a small glyph scaled up to 24 px gets thicker strokes. If the paper says the features
see a character's zone or size directly, correct that at revision; the Phase 1 size
measurement (`results/glyph_sizes_tummhcd.json`) shows it: ꯦ is written 0.47 L tall, ꯰ 0.87 L, so
in the 24 px images the strokes of ꯦ are about twice as thick. (2) Score the final system
without the 469 test images that have a train twin (`results/tummhcd_audit_duplicates.csv`); if
none of the 241 errors is among them, accuracy would be 98.04% instead of 98.12%. Earlier TUMMHCD
results share the test set, so the comparison stands; reporting both pre-empts a reviewer.
(3) Everyday writing does not distinguish ꯢ from ꯏ: the paper could say that 044/025 is a
spelling convention, under which its ensemble's accuracy is its own figure of 98.73%
(163 errors).

## Phase 1: synthetic words (details in `docs/phase1_synthetic_words.md`)

Package `mayek_words`, notebook `notebooks/phase1_synthetic_words.ipynb`. **Done (25 September
2026):** four runs on TUMMHCD by the owner, who approved the contact sheet of the fourth
("pretty satisfied"). Results of that run: `results/glyph_*.json`, `results/lexicon_stats.json`,
`results/spacing_synthetic_tummhcd.json`, `results/sign_placement_tummhcd.json`,
`results/pen_strokes_tummhcd_val.json`, `results/synth_{val,test}.json`; the fixed sets are on
Drive (`WORK/synth/{val,test}.tar`). Final checks on the full lexicon: signs beside a letter
never nearer the next letter nor touched by it (0%); signs above and below a median
0.085–0.10 L from their letter; 0.6% of words with strokes under 100 grey levels below the
paper; 32.4% of neighbouring letters touching (real 33.5%). Next: Phase 2 (recogniser).

- **Alphabet:** 54 characters, TUMMHCD without ꯢ; ꯏ is drawn with images of 025 and 044.
  Characters outside TUMMHCD (lum iyek ꯬, the Extensions) cannot be drawn.
- **Character images:** the paper's split, made by `mayek.split` (first project, pinned to
  commit 0d2c6e5). Synthetic training words use train images, validation words our
  validation part, test words TUMMHCD test without the 469 train twins; the 48 images with
  conflicting labels are used nowhere.
- **Proportions and placement** from Noto Sans Meetei Mayek (OFL 1.1, bundled in
  `mayek_words/assets`, measured with HarfBuzz): ꯥ, ꯩ, ꯪ above the character before them,
  ꯦ, ꯣ, ꯧ, ꯤ beside it, ꯨ below it, apun under the letter before it; positions relative to
  the pen, as in the font. With jitter off, the synthesiser reproduces the font's rendering.
- **Handwritten sizes from TUMMHCD itself:** the stretch to 24 x 24 thickens strokes in
  proportion, so the thickness of vertical and horizontal strokes gives back each class's
  lost width and height (checked on the font's characters: heights within 4%, widths
  within about 15%). Measured (`results/glyph_sizes_tummhcd.json`): letters as printed;
  signs written larger (height: ꯨ 2.4 times the font's, ꯩ 1.5, ꯧ 1.4, ꯪ 1.3, ꯦ and ꯥ 1.2;
  ꯤ 1.4 times as wide); strokes 0.07 L. The measured sizes are the default (owner,
  25 September 2026; packaged as `mayek_words/assets/glyph_sizes_tummhcd.json`); pen width
  0.06–0.14 L (owner, same day).
- **No writer IDs, so style matching:** each character is one of the 16 images of its class
  closest to a random anchor (within-class z-scores of slant, stroke width, ink fraction,
  ink darkness); the pen width is made equal across the word.
- **Spacing, from real handwriting:** in six samples of published handwriting (screenshots
  from the owner, measured with `scripts/measure_spacing.py`, images not kept) a third of
  neighbouring letters touch (median 33.5%) and the others are about 0.1 L apart. The
  synthesiser joins a letter to the one before it (ink touching) with a per-word chance of
  2–60% (2–50% before the signs' rule, below) and otherwise leaves −0.10 to +0.05 L on top of
  the font's side bearings; measured the same way, 34% touching with TUMMHCD characters, gaps
  0.059 L (`results/spacing_*.json`).
- **Syllables kept together (owner, 25 September 2026):** a syllable is a letter or an
  apun cluster, its vowel sign, and a nung or lonsum coda (`charset.syllables`). Spacing is
  usually even; a gap inside a syllable is never wider than the gaps around it, so ꯤ, ꯦ,
  ꯣ, ꯧ and a lonsum are never closer to the next letter than to their own; in 1 word in 5
  the syllables stand apart. Before this, ꯤ looked attached to the next letter in 87% of
  cases; now in none.
- **Signs on the ink (owner, 25 September 2026):** a sign beside its letter (ꯤ, ꯦ, ꯣ, ꯧ) is
  always closer to its own letter or even, never stuck to the letter on its right, never
  merged with its letter. TUMMHCD's ꯤ starts with a long lead-in and its stem stands
  mid-image, so the box rule had left the stem nearer the next letter in 73% of cases, and
  the next letter touched a sign in 16–19%. Now ꯤ is placed on the ink by its body: the
  layout's gap in even words (4 in 5); 0.02–0.06 L from its letter in uneven words (touching
  in 1 in 10), the next syllable further. ꯦ, ꯣ, ꯧ stay where the layout puts them (the owner:
  the close placement made them worse). For all: the next letter never touches the sign or
  sits nearer it; no sign ink goes into its letter (letter ink above and below it in a
  column; a lead-in may pass over). Result: 0% nearer the next letter, 0% next letter
  touching (`scripts/check_signs.py`, `results/sign_placement_dev_val.json`). Letters now
  join with a chance of 2–60% per word (was 2–50%), keeping 34% of letters touching.
- **Heights and faint strokes (owner, 25 September 2026):** ꯤ is sized like a letter, from its
  letter's foot to about 0.1 L above its top (it was scaled like the small signs and could end
  below its letter's top). Signs above and below (ꯥ, ꯩ, ꯪ, ꯨ, apun) are placed on the ink at
  their print distance from the letter (about 0.08–0.1 L; was a fixed height, so ꯩ floated
  over short letters such as ꯇ: 40% of ꯩ more than 0.2 L away, now 1%), never into a hollow
  of the letter. The pen step keeps faint strokes joined to the dark ones (pale bars of some
  ꯡ were erased): characters missing over 10% of their strokes 1.5% to 0.5%
  (`results/sign_placement_dev_val.json`, `results/pen_strokes_dev_val.json`).
- **Pale words (owner, third run, 25 September 2026):** placements approved; some characters
  looked as if disappearing: words in pale ink (palest scans on grey paper, blurred; 10% of
  words had strokes under 100 grey levels below the paper). A stroke's centre, after blur, is
  now at least 130 grey levels darker than the paper (`Config.min_contrast`; darkens 20% of
  words): under 100 grey levels 0.5%. The fixed sets' gap statistics of the third run were
  wrong for signs (a layout-recording bug, images unaffected); fixed, sets to be regenerated.
- **Lexicon:** the Phase 0 word lists, ꯢ written ꯏ, split by word with a hash (90% train,
  5% validation, 5% test), drawn with probability proportional to count^0.5; 3% numbers,
  2% full stops; 10% of draws go to rare letters (under 0.5% of characters: ꯘ, ꯓ, ꯙ ...;
  owner, 25 September 2026). Apun must join two consonants, lonsum letters included (typed
  loanwords: ꯑꯦꯟ꯭ꯗ "and"). First run: 76,066 words kept; 68,450 / 3,855 / 3,761 distinct
  words in training / validation / test.
- **Variety of words (owner, 25 September 2026):** training covers short and long words and
  the four kinds of syllable (C ꯀ, CV ꯀꯥ, CVC ꯀꯥꯡ, CC ꯀꯝ; `charset.syllable_type`): 15% of
  training words are built from real syllables (`lexicon.SyllableBank`; 1-6 syllables, the
  owner's maximum; kinds even). On the development list: 3.6% six-syllable words (text alone
  1.4%) and 10.8% CC syllables (7.7%). The rare long words of text (7+ syllables, 0.4%, mostly
  loanwords with endings) are kept.
- **Pale writing kept whole (25 September 2026):** the owner saw characters disappearing. The
  pen step keeps the ink map's pixels over 0.5, and the map was scaled to the darkest 5% of
  the ink, so pale scans lost their strokes: 11.4% of characters lost more than 30% of their
  ink (1.3% more than half). The map is now anchored to the Otsu threshold (what the scan shows
  as ink stays ink): none loses more than 30%. The measured sizes move a little with it
  (median width 1.03 times, height 1.01; single classes up to 13%); re-measured in the second
  run (strokes 0.084 L; signs a little larger).
- **Sets:** training words are rendered on the fly (4–11 ms per word per core); fixed
  synthetic validation and test sets of 5,000 words each, for model selection and a check.
  The main evaluation stays the Phase 3 real set. The gap figures in `synth_*.json` are
  between character boxes, not ink (`ink_gaps_in_L` in the fourth run's files, renamed
  `box_gaps_in_L`); for signs, the ink is measured by `scripts/check_signs.py`.

## Phase 2: recogniser (details in `docs/phase2_recogniser.md`)

Package `mayek_htr`, notebook `notebooks/phase2_recogniser.ipynb`. **Done on synthetic data
(26 September 2026):** two rounds of training by the owner, validation, the baseline, and
the synthetic test set used once (`docs/phase2_recogniser.md`, sections 6 and 7). The main
evaluation, on real handwriting, waits for the Phase 3 set.

- **Synthetic test, used once (`results/phase2_test_*.json`, `results/phase2_baseline_test.json`;
  5,000 words, TUMMHCD test characters):** final recipe (ConvNeXt-T from TUMMHCD, second
  round), mean of two seeds: CER 0.35%, WER 2.22% greedy; **CER 0.24%, WER 1.57% with the
  language model** (seeds 0.236% / 0.251%, 1.52% / 1.62%). Baseline with the same
  characters cut from the words, perfect cuts, perfect zones and the language model: CER
  4.26%, WER 21.0% (17 and 13 times as many errors); original isolated images with perfect
  zones and the language model (not attainable): 0.17%, 1.1%; the ensemble alone on
  isolated images: 1.20%, 7.6%. Swaps ꯦ/꯰ and ꯨ/ꯁ: ensemble on isolated images 94 and 115,
  recogniser 0 and 0 (every run); ꯗ/ꯘ 6-17 (ensemble 51). Rounds, seeds and encoders do not
  differ measurably (paired tests p >= 0.16). The figures are in-distribution (same
  synthesiser, same TUMMHCD writers): real handwriting will be harder.

- **Owner's decisions (26 September 2026):** (1) 10% of the training words are scrambled
  (`lexicon.scramble`: a lexicon word with each letter, lonsum letter and vowel sign replaced
  by a random one of its kind; `TrainConfig.scrambled`, default 0.1) to break the context
  prior behind the ꯘ errors; with scrambling off every word renders exactly as before, so
  the fixed sets stay reproducible. (2) Second round with this recipe: ConvNeXt-T from
  TUMMHCD with two seeds (0 and 1, training words 1000 and 1001), from ImageNet and the
  small CNN once each (runs `round2_*`, about 10 hours); the first round's runs are kept.
  (3) No real development set: to the owner the synthetic words look as realistic as
  actual writing, so the first check on real handwriting is the Phase 3 set.

- **Second round, synthetic validation (`results/phase2_val_round2_*.json`):** with the
  language model CER 0.253% / 0.253% / 0.268% / 0.259% and WER 1.68% / 1.64% / 1.78% / 1.70%
  (ConvNeXt-T from TUMMHCD seeds 0 and 1, from ImageNet, small CNN): slightly better than the
  first round but within the difference between two seeds; the seeds agree closely. The
  network alone reads the failing ꯘ words better (ꯃꯘ꯭ꯔꯦꯕꯤ misread 4-5 of 13 instead of 12,
  ꯇꯃꯟꯘꯁꯦꯠ 10-15 of 19 instead of 19, greedy), the language model pulls most back, and ꯗ is
  now read as ꯘ more often, so ꯗ/ꯘ keeps 35-36 errors: the hardest pair, to be measured on
  the real set. Second round = final recipe.
- **First round, synthetic validation (5,000 words; `results/phase2_val_*.json`):** CER
  0.30% / 0.29% / 0.32% and WER 2.0% / 1.9% / 2.1% greedy for ConvNeXt-T from TUMMHCD, from
  ImageNet and the small CNN from scratch; with the language model (order 6, every
  distinct word once, perplexity 7.61) CER 0.28% / 0.27% / 0.27%, WER 1.8% / 1.8% / 1.8%.
  Intervals overlap: synthetic words do not separate the encoders. ꯦ/꯰ and ꯨ/ꯁ read
  (almost) perfectly; ꯘ read as ꯗ half the time, but the validation set's 64 ꯘ come from
  at most four words (ꯘ is 0.009% of text; the draws for rare letters repeat them).
  Training was bound by CPU rendering (GPU waiting 49-65%; 2.4-2.6 h per run).
- **Baseline caveat:** the released first-paper networks were trained on TUMMHCD train
  *including* our validation part (`full` = train + val), so the baseline cannot be
  validated with them (isolated: 1 error in 33,175 validation characters,
  `results/phase2_baseline_val_released.json`); the synthetic test set is clean. Cut from
  the words, the same characters give CER 13.6%, and 4.2% with perfect zones and the
  language model (ꯁ read as ꯨ, ꯤ taking in its letter). Fix: the baseline is validated
  with the first paper's development networks (the first project's
  `runs/<network>/dev/final.pt`, trained without the validation part; the owner's Drive:
  `MyDrive/tummhcd98/runs`), assembled by `scripts/dev_ensemble.py`; the test uses the
  released networks with the weights chosen that way.
- **Clean baseline on validation (development networks, `results/phase2_baseline_val.json`):**
  isolated they miss 0.97% of the characters (so they had not seen them). Characters cut
  from the words, perfect zones and the language model: CER 4.0%, WER 20.2% (recogniser
  0.27%, 1.8%). Each character's original TUMMHCD image with perfect zones and the language
  model (not attainable by any segmenter): CER 0.17%, WER 1.1%.
- **ꯘ diagnosed (26 September 2026):** of the five validation words with ꯘ, two are misread
  every time by all three runs (ꯇꯃꯟꯘꯁꯦꯠ as ꯇꯃꯟꯗꯁꯦꯠ, ꯃꯘ꯭ꯔꯦꯕꯤ as ꯃꯗ꯭ꯔꯦꯕꯤ) and three
  never; the development networks read all 64 ꯘ images right. The recogniser has learned
  from its training words a context prior (ꯟꯗ, ꯗ꯭ꯔ are common) that overrides the image.
  Without those two words: CER 0.17-0.18%, WER 1.1-1.2% with the language model, the level
  of the isolated-image baseline. Proposed remedy (owner to decide; needs retraining):
  training words with letters replaced by random letters of the same kind.

- **Model:** the word image contrast-normalised (paper = its 90th percentile, ink = its 1st),
  64 px high, proportions kept (at most 1,024 px wide); ConvNeXt-T stem and stages 1-3, the
  third stage's downsampling halving the height only (4 rows x 384 channels, a column per
  8 px: on 3,000 synthetic words no word has too few columns for CTC, 1.8 times the need at
  the 1st percentile); each column projected, 2-layer BiLSTM (256 per direction), CTC over
  the 54 characters and the blank. Unicode order is the reading order (a sign after its letter).
- **Runs:** ConvNeXt-T from the first paper's TUMMHCD weights (main; the grey member without
  size features, from the Hugging Face release), from ImageNet, and a small CNN from scratch
  (the CRNN baseline).
- **Training:** 60,000 steps of 64 synthetic words rendered on the fly (own seed, 1,000),
  AdamW 3e-4, 2,000 warm-up steps, cosine to 1%, EMA 0.999, bfloat16 encoder; augmentation on
  the GPU towards phone photos (rotation, shear, scale, warp, stroke width, blur, ink,
  shading, ruled lines, noise; no flips). Greedy CER on the synthetic validation set every
  2,000 steps picks `best.pt`. Resumable after a disconnection.
- **Language model:** character n-grams (Kneser-Ney) of the training lexicon, order (3-7)
  and word weighting (count ** 0, 0.5 or 1) chosen by validation perplexity; CTC beam search
  with its weight alpha and a bonus beta per character chosen on the synthetic validation set.
- **Metrics:** CER, WER (word accuracy with a 95% Wilson interval), the pairs ꯦ/꯰, ꯨ/ꯁ,
  ꯗ/ꯘ, and, on synthetic sets, by kind of word (lexicon, composed of syllables, numbers).
- **Baseline:** perfect segmentation (the synthesiser's boxes), each character classified
  by the first paper's ensemble with ꯢ and ꯏ merged: as its original TUMMHCD image (upper
  bound) and cut from the word image; with perfect zones (the idea of Hijam and Saharia's
  second stage; the orthographic rule no longer matters in everyday spelling) and with the
  same language model. The CRNN from scratch is the segmentation-free baseline.
- **Protocol:** choices on the synthetic validation set; the synthetic test set once (the
  notebook's `RUN_TEST`); two or more seeds for final numbers; the Phase 3 real set is the
  main evaluation (no real development set: owner, 26 September 2026).

## Phase 3: the real word set (details in `docs/phase3_real_words.md`)

Module `mayek_htr/pages.py`, scripts `make_writing_pages.py`, `cut_writing_pages.py` and
`measure_strokes.py`; the demo: `mayek_htr/web.py`, `mayek_htr/reader.py`, `web/`, `space/`,
`scripts/build_demo.py`, `scripts/compare_runs.py`; notebook
`notebooks/phase3_real_words_and_demo.ipynb`. **Status (27 September 2026): the real test done,
once, by the owner (100 words, all seven runs); the demo built for Hugging Face. Results below.**

- **Real test, used once (`results/phase3_real_*.json`, `results/phase3_real_comparison.json`;
  100 words, one writer, iPad):** final recipe (ConvNeXt-T from TUMMHCD, second round), mean of
  two seeds: CER 3.79%, WER 21.0% greedy; **CER 3.12%, WER 17.0% with the language model**
  (seeds 2.36% / 3.88%, 13% / 21%: 87 and 79 words right). The same runs on the synthetic test:
  0.24%, 1.57%, so real handwriting brings about 13 times the character errors. Other runs with
  the language model: ImageNet 2.53% / 14%, small CNN 3.03% / 17% (first round 2.87% / 17%,
  2.53% / 15%, 4.05% / 21%). Paired tests (21 pairs, so p near 0.03 is weak evidence): with
  the language model the two seeds differ (9 words against 1, p = 0.02; greedy p = 0.18), seed
  0 reads more than the first round's small CNN (10 against 2, p = 0.04) and seed 1 fewer than
  ImageNet (1 against 8, p = 0.04); every other pair p >= 0.07. Greedy, the second round's
  small CNN is below both of its ConvNeXt runs (4 against 14, p = 0.03). Seeds differ far more
  on real words than on synthetic ones: report their mean. The language model puts 3-11
  words right per run and wrong only once in all seven (p = 0.001 for the second round's small
  CNN). 69 words read right by every run, 7 by none. Numbers 10 of 10 everywhere.
- **Where the errors are:** the pairs that trouble isolated characters are read without a
  swap (ꯦ/꯰ 32 of 32, ꯨ/ꯁ 55 of 55, ꯗ/ꯘ 20 of 20; but ꯰ occurs 3 times, ꯘ once). Nearly every
  error is a sign written above the letter: of the 56 words without ꯥ or ꯩ, 0-2 are misread
  (seed 0: 1); of the 44 with one, 12-20. ꯥ is misread 17-36% of its 47 times (dropped, or
  read as ꯣ or ꯦ); ꯩ is read as ꯧ in 3-5 of its 5 words. This writer draws ꯥ as a long
  slanted stroke at the upper right of the letter, unlike the training words' ꯥ (TUMMHCD
  shapes at the font's place): the gap is in the synthetic signs, not in the letters. A fix
  (signs from more writers, their shapes and places; Phase 4) must be judged on new real
  words (the 400 unwritten items), not on these 100.
- **Demo (`results/phase3_demo.json`):** built from `round2_convnext_tummhcd_seed1` (chosen on
  validation before the test; on the real words it is the weaker seed). ONNX with float16
  weights: the same greedy reading as PyTorch for 300 of 300 validation words (largest log
  probability difference 0.053), 32.7 MB; the language model for the browser 669,149
  entries, 3.19 MB compressed. The cards quote its own real-word scores (CER 3.88%, WER 21%).

- **Owner's decision (26 September 2026):** "I dont have any people to write actual words for
  me. I want to do it myself. I can write on my ipad many words in one page for many pages and
  you can crop them into the required real word test dataset." One writer (the owner, a
  native writer), Apple Pencil on an iPad. The set stays text-disjoint (test-split words) and
  its writer is not one of TUMMHCD's, but one writer cannot show the spread across writers,
  and digital ink is a second change of domain (even strokes on white, no paper). Contribution
  (3) of the novelty statement no longer holds as written.
- **Pages:** A4 at 200 dpi; each word printed (shaped, Noto Sans Meetei Mayek) above an empty
  box 2.2 cm high and as wide as the word needs; about 18 words a page; four corner squares;
  the layout embedded in the PDF (`manifest.json`).
- **Words:** distinct test-split words (hash, so never training words), 2-14 characters,
  count >= 2, drawn by count ** 0.5 without replacement; every letter and sign 8 times where
  the test words allow; 5% numbers, 2% full stops; random order. Only counts are committed.
- **Cutting:** corner squares -> projective alignment (photos too, paper flattened); page
  identified by its printed ink; handwriting = darker than the printed page nearby, any pen
  colour; each piece of ink to its box (`outside`, `shared` flags; words over a box line kept
  whole); word cut from its own ink with margin 0.15 x height; empty boxes skipped. Output in
  the fixed-set form (`images/`, `labels.tsv`) with `manifest.json` (kinds, flags), and a
  contact sheet per page. On simulated pages every page found, foreign pages refused, words
  keep 1.00-1.01 of their ink on exported pages (0.89-1.04 on a phone photo).
- **Confirmed by the owner (27 September 2026):** (1) 500 items; (2) the trial page first (17
  validation words, to check the iPad export and the cutting; never scored); (3) contribution
  (3) reworded (novelty statement above); (4) evaluation: all seven Phase 2 runs, greedy and
  with the language model at the alpha and beta chosen on synthetic validation, the set used
  once (the oracle baseline cannot run on real words: it needs character boxes).
- **Trial page (27 September):** the iPad's Markup export (iOS 18.5) works as it is: the page
  keeps its size and embedded layout, the drawing is one annotation rendered with the page;
  17 of 17 words cut whole, 3 flagged `outside` (a stroke over a box line), all complete.
  Letters about 12 mm high, strokes about 0.5 mm (0.043 of the letter height): at the
  recogniser's input the strokes are half as thick as the training words' (2.0 px against
  4.0 px; ink 6.7% of the image against 13.5%; `results/phase3_trial_strokes.json`), below
  the synthesiser's pens (0.06-0.14 L). Proposed: a pen about twice as wide.
- **Pen (27 September):** the owner tried two widths on page 1 and asked which. The second
  matches the training words (strokes 1.2 mm, 0.10 of the letter height; at the recogniser's
  input 4.0 px, 0.129 of the ink band, ink 14.0%, against 4.0 px, 0.121, 13.5%); the first is
  too thick (2.2 mm, 0.15 L; 6.0 px, 0.194, 23.1%) (`results/phase3_pen_choice.json`).
  Recommended: the second pen for every page, items 1-2 rewritten with it.
- **Test pages (27 September):** 500 items on 29 pages (15-20 a page) from the test lexicon
  (3,761 words, all test split by hash; 3,725 usable, 1,834 seen at least twice): 475 words
  and 25 numbers, 7 ending with a full stop, 3,110 characters (6.2 an item); every letter
  and sign at least 8 times except ꯘ (5: every test word that has it), digits 4-10 times
  (`results/phase3_pages.json`). The PDF (its words embedded) stays off the repository:
  Drive `WORK/real/real_words_pages.pdf`.

- **Owner's decision (27 September 2026):** "I have done 100 words . So i wanna only test using
  100 words and launch a similar hugging face space demo for my final model as the character
  model which hit 98% accuracy . And in the future , i can collect more real data and improve
  it. For now this is okay." The real test set is items 1-100 (pages 1-6; items 101-500 not
  written, kept for later). 100 words give wide intervals (at 80% of words right, the 95%
  interval is about 71-87%): a first check on real handwriting, not yet a benchmark.
  Contribution (3) reworded to match, confirmed by the owner the same day: "a first real
  handwritten word set, 100 words by a native writer on a tablet, text-disjoint from
  training, with a fixed protocol and the tools to extend it" (novelty statement above).
- **The real set (`results/phase3_real_set.json`, `results/phase3_real_strokes.json`):** 100
  words (90 from the lexicon, 10 numbers), 593 characters, 52 of the 54 (not ꯳, ꯓ; ꯘ, ꯙ, ꯚ, ꯴ once);
  for the pairs ꯦ 29, ꯰ 3, ꯨ 16, ꯁ 39, ꯗ 19, ꯘ 1. All cut whole (20 flagged `outside`, all
  complete on the contact sheets). One pen throughout, as the training words at the
  recogniser's input: strokes 4.0 px, 0.125 of the ink band, ink 14.7% (synthetic 4.0 px,
  0.121, 13.5%). The set (`real_test.tar`, with its manifest) goes on Drive in `WORK/real/`;
  not in the repository.
- **Test protocol:** the notebook's section 1 (`RUN_REAL_TEST`): all seven Phase 2 runs read
  the 100 words once (`eval_recogniser.py --tuned`, the settings of synthetic validation),
  greedy and with the language model: `results/phase3_real_<run>.json`.
- **Demo (as the first project's):** a static Hugging Face Space where everything runs in the
  visitor's browser (ONNX Runtime Web 1.30.0), and a model repository
  (`Chingkheinganba/handwritten-meitei-mayek-word-recognition`, both). The network exported to
  ONNX (float16 weights; one word of any width; checked against PyTorch on synthetic
  validation words), the language model written in back-off form (`web/lm.bin.gz`, the exact
  probabilities of `CharLM`), preprocessing and CTC beam search ported to JavaScript: on
  test inputs the page's images and readings are identical to Python's (`tests/test_web.py`,
  Node; checked end to end in headless Chromium). The page shows the reading with and
  without the language model, the alternatives, and the standard spelling on request
  (`charset.standard_spelling`: ꯢ after ꯥ, ꯣ, ꯨ). The demo's network: of the final
  recipe's two seeds, the better on synthetic validation (CER tied at 0.253%, WER 1.64%
  against 1.68%): `round2_convnext_tummhcd_seed1`, alpha 0.5, beta 0. Python use:
  `mayek_htr.reader.Reader`. The language model is built from CC BY-SA 4.0 (Wikipedia) and
  ODC-By (FineWeb-2) text: released under CC BY-SA 4.0 with attribution (model card).

## Working rules

- Environment: Google Colab, one NVIDIA A100 (80 GB). Keep notebooks runnable top to bottom.
- Every number in a paper comes from a results file written by the code (as `results/results.json`
  in the first project). Fix seeds; report more than one seed for final numbers.
- Choose everything on a validation split; touch the test set once.
- Commits: author `chingkheinganba231005 <chingkheinganbaofficial@gmail.com>`. Owner's preference:
  no Co-Authored-By or session trailers in commit messages.
- Never ask for tokens or passwords in chat. The Hugging Face token goes in Colab secrets or getpass.
- The owner's older TACL-Net project is unpublished and private. Do not mention it anywhere.
- Keep `docs/ai_use_log.md` up to date (date, what Claude Code did: code, data scripts, literature
  search, any text editing). Journals ask how AI tools were used; this log makes that statement
  quick and accurate.
- Writing: plain and precise, British spelling (recognise, optimise), no filler phrases.
