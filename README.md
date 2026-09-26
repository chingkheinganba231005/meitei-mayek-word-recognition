# Handwritten Meitei Mayek word recognition

Reading whole handwritten Meitei Mayek words instead of isolated characters, using a character
language model and the script's orthographic rules to fix errors that isolated-character models
make (such as the vowel sign ꯦ against the digit ꯰, or ꯨ against ꯁ). A second stage will generate
handwriting in different writers' styles with a diffusion model. The project brief and every decision so far are
in [`CLAUDE.md`](CLAUDE.md).

This follows the character-level work in
[Handwritten-Meitei-Mayek-Recognition](https://github.com/chingkheinganba231005/Handwritten-Meitei-Mayek-Recognition)
(98.12% on TUMMHCD).

## Status

- Phase 0, the literature and data audit: done, [`docs/phase0_audit.md`](docs/phase0_audit.md).
- Phase 1, synthetic words from TUMMHCD characters: done,
  [`docs/phase1_synthetic_words.md`](docs/phase1_synthetic_words.md).
- Phase 2, the word recogniser: done on synthetic data (synthetic test set: CER 0.24%, WER
  1.57% with the language model, against 4.3% and 21% for the first paper's ensemble on the
  same characters cut from the words); the real-handwriting evaluation waits for Phase 3,
  [`docs/phase2_recogniser.md`](docs/phase2_recogniser.md).

## Layout

```
mayek_words/         synthetic words (Phase 1)
  charset.py         the 54-character alphabet and the TUMMHCD classes that draw it
  glyphs.py          character images by split, ink maps, style matching
  synth.py           layout from the font priors, drawing, word images on demand
  lexicon.py         words in everyday spelling, split by word, sampling
  sizes.py           characters' lost sizes recovered from stroke thickness
  assets/            font priors, font characters as 24 x 24 images, Noto Sans Meetei Mayek (OFL)
mayek_htr/           the word recogniser (Phase 2)
  labels.py          the output alphabet (54 characters and the CTC blank)
  images.py          word images: contrast, height, padding
  data.py            training batches rendered on the fly, fixed sets
  augment.py         augmentation on the GPU
  model.py           ConvNeXt-T or small CNN encoder, BiLSTM, CTC; the first paper's weights
  train.py           training loop (EMA, checkpoints, resuming)
  lm.py              character n-gram language model
  decode.py          greedy and beam search decoding
  metrics.py         CER, WER, confusable pairs
scripts/
  audit_tummhcd.py   writer information in TUMMHCD: folders, file names, side files, hidden grouping
  corpus_stats.py    size of Meitei Mayek text corpora and how often the ꯢ / ꯏ rule holds
  i_exceptions.py    words that break the ꯢ / ꯏ convention, as a review sheet for a language expert
  glyph_priors.py    size and position of every character in the font (-> mayek_words/assets)
  build_glyphs.py    character stores per split from the paper's split
  glyph_sizes.py     how large people write each character, measured on TUMMHCD
  build_lexicon.py   the lexicon from the Phase 0 word lists
  render_words.py    fixed synthetic word sets and contact sheets
  measure_spacing.py how close together handwritten letters are, in real images and synthetic pages
  check_signs.py     where the signs sit on the ink: beside (never nearer the next letter), above, below
  check_strokes.py   whether the pen step keeps every stroke, faint ones included
  build_char_lm.py   the character language model, chosen on validation
  train_recogniser.py  a training run of the recogniser
  eval_recogniser.py   scores, with and without the language model
  oracle_baseline.py   the baseline: perfect segmentation and the first paper's ensemble
notebooks/
  phase0_data_audit.ipynb       Phase 0 on Colab (TUMMHCD from Google Drive, corpora downloaded)
  phase1_synthetic_words.ipynb  Phase 1 on Colab
  phase2_recogniser.ipynb       Phase 2 on Colab (A100)
  collect_results.ipynb         the small result files from Drive in one zip, to send (CPU runtime)
results/             JSON files written by the code; every reported number comes from here
docs/                Phase 0 report, Phase 1 and Phase 2 design, AI use log
tests/               pytest
```

## Running

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/audit_tummhcd.py --zip TUMMHCD-TEST-TRAIN.zip          # -> results/tummhcd_audit.json
python scripts/corpus_stats.py wiki=mniwiki-latest-pages-articles.xml.bz2   # -> results/corpus_stats.json
python scripts/render_words.py --glyphs font --lexicon words.tsv --n 48 --sheet sheet.png   # a quick look
```

Phase 1 on TUMMHCD: `notebooks/phase1_synthetic_words.ipynb` (first project's split, character
stores, size measurement, lexicon, contact sheets, fixed synthetic sets). Phase 2:
`notebooks/phase2_recogniser.ipynb` (language model, training runs, validation, baseline,
test once); its tests need PyTorch (`pip install -r requirements-htr.txt` after PyTorch).

TUMMHCD comes from its authors, <http://agnigarh.tezu.ernet.in/~sarat/resources.html>, under
their own terms.
