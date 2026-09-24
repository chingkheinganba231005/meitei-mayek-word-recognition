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
  digits and consonants in the middle zone).
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
