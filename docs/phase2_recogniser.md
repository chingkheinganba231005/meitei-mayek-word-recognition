# Phase 2: the word recogniser

Started 25 September 2026. **Status: code ready, not yet run on TUMMHCD.** Everything below
was checked in the cloud session on the CPU (section 5); training needs the Colab A100
(`notebooks/phase2_recogniser.ipynb`). The decisions in section 6 wait for the owner's
confirmation.

## 1. What the recogniser does

Package `mayek_htr`. A word image in, its text out, with no cutting into characters:

1. **Image.** Contrast normalised (paper = the image's 90th percentile, ink = its 1st, so
   pale ink on grey paper looks like dark ink on white), scaled to 64 px high with its
   proportions kept (at most 1,024 px wide), ink 1 and paper 0.
2. **Encoder.** The stem and first three stages of ConvNeXt-T, with the third stage's
   downsampling halving the height only: a 64 px word becomes 4 rows × 384 channels, one
   column per 8 px of width.
3. **Sequence.** Each column, all 4 rows together (a sign keeps its height above or below
   the letter), is projected to 512 numbers and read by a 2-layer bidirectional LSTM
   (256 per direction).
4. **Output.** 55 classes per column: the 54 characters and the CTC blank.
5. **Decoding.** Greedy (the best class of every column, repeats merged, blanks dropped),
   or CTC beam search with a character language model (section 2.7).

Text is in Unicode order, which follows the writing: a sign above, below or beside a letter
comes after the letter, and apun between the two letters it joins. CTC reads left to
right, so a sign above or below its letter is read in the same columns as the letter,
just after it.

## 2. Decisions

**2.1 Alphabet.** The 54 characters of Phase 1 (everyday spelling: ꯢ is written ꯏ), and the
blank. References are normalised the same way; a character outside the alphabet (lum iyek,
the Extensions) cannot be read and is dropped from the reference, with a count.

**2.2 Input size: 64 px high, a column per 8 px.** The synthetic words are about 65 px
high with letters about 32 px high (`results/synth_val.json`), so they are used at their
own size. CTC needs at least one column per character, plus one between two equal
characters. Measured on 3,000 synthetic words (TUMMHCD validation characters, the Phase 1
settings): a column per 8 px gives 3.6 columns per needed column at the median and 1.8 at
the 1st percentile, and no word has too few; words are 5.9 characters long on average.
Finer columns (4 px) cost more LSTM steps and gain nothing here; coarser (16 px) would
leave words with signs above and below short of columns.

**2.3 Encoder and its starting weights.** ConvNeXt-T, as in the first paper, three ways:

| Run | Encoder | Starting weights |
|---|---|---|
| `convnext_tummhcd` (main) | ConvNeXt-T, stages 1–3 | the first paper's ConvNeXt-T on TUMMHCD (grey input, without size features), from the Hugging Face release |
| `convnext_imagenet` | same | ImageNet (`convnext_tiny.fb_in22k_ft_in1k`, where the first paper started) |
| `crnn_scratch` | a plain CNN, one column per 4 px | none: the CRNN baseline (Shi et al., TPAMI 2017), as used for Indic handwritten words (Gongidi and Jawahar, ICDAR 2021, add a spatial transformer and a ResNet) |

The comparison of the first two tells whether training on isolated TUMMHCD characters
helps a word reader; the third, whether a pretrained encoder does. The fourth stage of
ConvNeXt-T is dropped: its 32 px stride is too coarse for words. Loading the TUMMHCD
weights checks that every encoder tensor is found (`build` stops otherwise).

**2.4 Sequence model: BiLSTM with CTC first.** The standard, and robust with synthetic
data. An attention decoder is the next step, once CTC results are in.

**2.5 Training.** 60,000 steps of 64 synthetic words (3.84 million words; each training
lexicon word about 45 times on average, frequent words more), rendered on the fly by the CPU's worker processes:
item i of the training stream is always the same image (seed 1,000; the fixed sets use
seeds 1 and 2), and after an interruption the stream goes on with new words. AdamW,
learning rate 3e-4 with 2,000 warm-up steps and a cosine to 1%, weight decay 0.05,
gradients clipped at 5, bfloat16 for the encoder and float32 for the LSTM, an exponential
moving average of the weights (0.999) evaluated and kept, as in the first paper. Words
are grouped by width so that a batch pads little. Every 2,000 steps: greedy CER and WER on
the fixed synthetic validation set (5,000 words); `best.pt` keeps the lowest CER.
Checkpoints every 1,000 steps on local disk and Drive; a run started again goes on.

**2.6 Augmentation (GPU), towards photographed handwriting.** The synthesiser already
varies size, spacing, slant, pen, blur and ink. On top, for every word: rotation up to 2°,
shear up to 0.2, width 0.85–1.0 and height 0.85–1.1 times, vertical shift up to 5%, a
smooth warp (half of the words, 1.5 px), strokes 1 px thicker (15%) or thinner (10%), blur
(25%), ink strength 0.55–1, uneven grey paper (half), a ruled line (15%) and noise (half).
No flips (a mirrored character is another character or none).

**2.7 Language model and decoding.** A character n-gram model of words (interpolated
Kneser-Ney, discount 0.75), trained on the training lexicon's words weighted by
count ** power, with numbers and full stops in the synthesiser's shares. Order (3–7) and
power (0, 0.5, 1) are chosen by perplexity on the validation words (weighted as the
synthetic validation words are drawn). The test words are not in its training words (the
lexicon is split by word): it helps through the syllables and endings it has seen. Beam
search (16 prefixes): CTC log probability + alpha × the model's log probability (end of
word included) + beta per character; alpha and beta are chosen on the synthetic
validation set by CER (then WER).

**2.8 Metrics** (`mayek_htr/metrics.py`). CER (edit distance over all words / reference
characters), WER (share of words not read exactly; word accuracy with its 95% Wilson
interval), and for each confusable pair of the first paper (ꯦ/꯰, ꯨ/ꯁ, ꯗ/ꯘ) the share of
its characters read correctly and how many were read as the other. For the synthetic sets
also by kind of word: lexicon words, words composed of syllables (15%), numbers (3%).

**2.9 The baseline to beat** (`scripts/oracle_baseline.py`). Segment, then classify with
the first paper's ensemble, with the segmentation perfect: the validation and test words
are rendered again from their settings, which gives every character's box. Each character
is classified two ways: its original TUMMHCD image, as in the first paper (an upper bound:
no neighbours), and cut from the word image by its box, stretched to 24 × 24 like a
TUMMHCD image (neighbours' strokes that reach into the box stay). ꯢ and ꯏ are one letter
(their probabilities added). Decoding: the best class per character; with zones, after
Hijam and Saharia (2024): a character above the line can only be ꯥ, ꯩ or ꯪ, below it only
ꯨ or apun, raised beside its letter only ꯦ, ꯣ or ꯧ, and on the line only the rest; the
zones are perfect (each character's own; from the boxes alone 1% of characters would fall
in the wrong zone, since a raised ꯣ can sit higher than a low ꯥ); and each with the same
character language model, its weight chosen on validation. With perfect segmentation and
zones this baseline is stronger than any real segment-then-classify system, so beating it
is a strong result. The CRNN from scratch (2.3) is the segmentation-free baseline.

**2.10 Protocol.** Everything is chosen on the synthetic validation set; the synthetic
test set is used once, with the notebook's `RUN_TEST` switch. Final numbers from at least
two seeds (weights and training words). The synthetic sets are a check: the main
evaluation is the real set of Phase 3.

## 3. Files

| File | What |
|---|---|
| `mayek_htr/labels.py` | alphabet and class ids |
| `mayek_htr/images.py` | contrast normalisation, height, padding; cutting a word out of a photo |
| `mayek_htr/data.py` | training stream (DataLoader workers), fixed sets (folder or .tar), kinds of word |
| `mayek_htr/augment.py` | GPU augmentation |
| `mayek_htr/model.py` | encoders, BiLSTM and CTC head, loading the first paper's weights |
| `mayek_htr/train.py` | training loop, EMA, checkpoints, resuming, greedy validation |
| `mayek_htr/lm.py`, `decode.py`, `metrics.py` | language model, decoding, scores |
| `scripts/build_char_lm.py` | the language model → `results/phase2_lm.json` |
| `scripts/train_recogniser.py` | a training run → `results/phase2_train_<run>.json` |
| `scripts/eval_recogniser.py` | scores, alpha and beta, predictions, a sheet of misread words → `results/phase2_{val,test}_<run>.json` |
| `scripts/oracle_baseline.py` | the baseline → `results/phase2_baseline_{val,test}.json` |
| `notebooks/phase2_recogniser.ipynb` | all of it on Colab |

## 4. Time on the A100

Measured on the first run (`convnext_tummhcd`, Colab A100 40 GB, owner, 25 September 2026):
25,000 steps in 67 minutes, about 6 steps (400 words) per second, so about 2.7 hours per run
of 60,000 steps and about 8 hours for the three. The estimate made before (1 to 1.5 hours,
from rendering at about 7 ms per word per CPU core in the cloud session) was too optimistic.
The log's `waiting for data` share tells whether the CPU rendering the words or the GPU
sets the pace; an A100 80 GB has the same compute as the 40 GB card (more memory, which
this model does not need, and about a quarter more memory bandwidth), so it would help
little either way.

## 5. Checks done in the cloud session (CPU)

- Shapes: 64 × W px → W / 8 columns (ConvNeXt), W / 4 (small CNN); 16.3 M and 8.7 M
  parameters with the head.
- The first paper's weights: a folder in the release format (config.json and one state
  dict per member, written with the first project's own `make_model`) loads into the
  encoder with every tensor found and unchanged. The real download (Hugging Face) is
  blocked in this session, so it happens in the notebook.
- Training stream, fixed sets from a folder or a .tar, word kinds, augmentation, a CTC
  model learning a batch, and a training run written, resumed and finished: tests
  (`tests/test_htr.py`, `tests/test_htr_torch.py`; the second is skipped without PyTorch).
- Learning: the small CNN on the CPU, 1,500 steps of 16 words (light augmentation,
  learning rate 1e-3), TUMMHCD validation characters for both training and checking, the
  Phase 1 development word list (7,000 words) for training and 300 words of other words for
  checking: greedy CER 0.96 at step 250, 0.77 at 750, 0.42 at 1,000, 0.25 at 1,500 (WER
  0.74). Beam search with the language model (alpha 0.25 and beta 1.5, chosen on the same
  300 words) brought it to CER 0.20 and WER 0.61, and ꯗ/ꯘ from 66% to 79% read correctly.
  A check that the chain works, not a result: the characters were the same, the run short.
- The whole notebook, run top to bottom in this session with Colab stubbed out, the
  simulated release, the development data, the CPU and a few training steps (the ImageNet
  run left out, as Hugging Face is blocked here): every section works, including going on
  after a stop, choosing alpha and beta, the baseline (40 of 40 words rendered again
  identically) and the test cells. It found two faults, both fixed and tested: two runs
  whose folders had the same name shared their local checkpoint copy, and a run folder
  emptied to start again was resumed from the old local copy.
- The language model on the development word list (7,000 words): order 4, every distinct
  word once, perplexity 10.4 per character on the validation words; about 3 seconds for
  the whole grid.
- The baseline on 60 words with the simulated release: every word rendered again came out
  identical to the stored image.

## 6. For the owner to confirm

1. Input 64 px high and a column per 8 px (2.2).
2. The three runs (2.3): ConvNeXt-T from TUMMHCD (main), from ImageNet, and the small CNN
   from scratch.
3. 60,000 steps of 64 words per run (2.5), about 2.7 hours each (section 4).
4. The baseline with perfect segmentation and perfect zones (2.9), in its two forms.
5. The language model from the training lexicon only, alpha and beta chosen on the
   synthetic validation set, and the synthetic test set used once (2.7, 2.10).
6. A suggestion: a small real development set (for example 100 to 200 words in the
   owner's own hand, photographed with a phone), kept apart from the Phase 3 test set. The
   synthetic validation words cannot show how the recogniser does on real writing; a few
   real words would, and would guide the augmentation before Phase 3.

## 7. Runs

None on TUMMHCD yet.
