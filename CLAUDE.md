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
itself.

**Phase 4: generation (project 2).** Diffusion model conditioned on text and writer style. Evaluate
by training the recogniser on generated data and testing on the real set, plus standard image
metrics.

## Phase 0 findings (first pass, 24 September 2026; details in `docs/phase0_audit.md`)

Found by web search only (no full-text access in that session): every paper still has to be
checked against its full text before citing.

**Novelty statement (draft; (3) and (4) revised and confirmed by the owner, 24 September 2026).**
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
synthetic words from TUMMHCD at scale; (3) a public, consented, writer-disjoint real word set
(50+ writers) with a fixed protocol (CER, WER, accuracy on the confusable pairs); (4) evidence that
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

Package `mayek_htr`, notebook `notebooks/phase2_recogniser.ipynb`. **First round run by the
owner (26 September 2026): three runs, validation, baseline on validation. Second round
(scrambled words) run and validated the same day; the synthetic test set untouched**
(`docs/phase2_recogniser.md`, sections 6 and 7).

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
  the real set. Second round = final recipe. Next: the test, once, over all seven runs and
  the baseline (notebook section 7).
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
