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
  perfectly would lift the ensemble to 98.73%. **This pair is the reason for the word-level project.**
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
re-implementation of the zone-and-rule second stage. Metrics: CER, WER, and accuracy on the ꯢ/ꯏ pair.

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

**Novelty statement (draft).** No published work recognises handwritten Meitei Mayek words or
lines end to end. Earlier work classifies isolated characters, segments handwritten pages into
lines and words without recognising them (Inunganbi, Choudhary, Manglem, The Visual Computer 2020,
doi 10.1007/s00371-020-01799-4: 189 pages, word segmentation 88.96%), or corrects a character
classifier on segmented words with zones and the orthographic rule (Hijam and Saharia 2024;
Hijam's thesis). The only large handwritten Manipuri word dataset, IIIT-Indic-HW-UC (Mondal and
Jawahar, ICPR 2024), is in Bengali script. Meitei Mayek text recognition exists only for print
(NE-OCR 2026 preprint); for scene text there is character recognition, and EMBiL only detects
text and identifies its language. No handwriting generation model exists for the script. We
contribute: (1) the first segmentation-free handwritten Meitei Mayek word recogniser; (2) zone-aware
synthetic words from TUMMHCD at scale; (3) a public, consented, writer-disjoint real word set
(50+ writers) with a fixed protocol (CER, WER, ꯢ/ꯏ accuracy); (4) a controlled measurement of how
much context resolves ꯢ/ꯏ; (5) later, the first diffusion model for Meitei Mayek handwriting.

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
training only, flagged. Real-test-set prompts should come from a CC BY source.

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
28–45%. **Recommended convention (owner to confirm with a language expert):** the thesis spelling,
ꯢ for the i after ꯥ, ꯣ or ꯨ and ꯏ elsewhere, with an expert-checked exception list taken from the
ꯢ-writing pages. Text for the language model: ꯢ-writing pages (70% cut) as they are, ꯏ-writing
pages normalised and flagged; ꯢ/ꯏ accuracy measured only on held-out ꯢ-writing pages and on the
real test set. Also check the thesis's exact wording of the rule and how TUMMHCD labelled 044/025.

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
see a character's zone or size directly, correct that at revision. (2) Score the final system
without the 469 test images that have a train twin (`results/tummhcd_audit_duplicates.csv`); if
none of the 241 errors is among them, accuracy would be 98.04% instead of 98.12%. Earlier TUMMHCD
results share the test set, so the comparison stands; reporting both pre-empts a reviewer.

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
