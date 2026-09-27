# Round 3: the signs above written beside their letter

Owner's request (27 September 2026), after the real test and the demo: "you pointed out some
errors and said we could include them in our training data and train again. So lets do that
so that our model becomes even more better. Create synthetic data which imitates the new
patterns."

## 1. The errors

The 100 real words, read once by every Phase 2 run (`results/phase3_real_<run>.json`, the
`top_errors` of each run with the language model; every error of every run is listed there):
over the seven runs, 126 character errors, of which 118 (94%) are the three signs that print
places above the letter:

| error | runs' total |
|---|---|
| ꯥ left out | 45 |
| ꯥ read as ꯣ | 30 |
| ꯩ read as ꯧ | 28 |
| ꯪ read as ꯦ / ꯣ | 7 / 2 |
| ꯥ read as ꯦ | 6 |
| ꯫ left out | 7 |
| ꯨ left out | 1 |

The seven ꯫ are one word (item 35) read the same way by every run: the writer left out its
full stop (no stray ink on the page, no flag from the cutter), so the reading follows the ink
and the label does not. Proposed: correct that label (owner to decide); every run's word error
rate would fall by one point, no reading changes.

## 2. Where the writer puts the signs (`scripts/measure_sign_geometry.py`)

Measured on the ink of the real words (`results/round3_sign_geometry_real.json`; 59 of the 76
words with a sign above or beside at the top; the other 17 have a sign touching its letter or
a letter in pieces, so their pieces cannot be matched to the text), in L, the height of the
letters' band; 10th, 50th and 90th percentiles:

| sign | n | centre after its letter's right edge | top above the letters' top | bottom | slant | elongation |
|---|---|---|---|---|---|---|
| ꯥ | 36 | 0.01 / 0.18 / 0.27 | 0.08 / 0.18 / 0.30 | -0.24 / -0.19 / -0.04 | 43 / 52 / 60° | 3.2 / 4.2 / 5.1 |
| ꯣ | 13 | 0.14 / 0.28 / 0.36 | 0.05 / 0.20 / 0.31 | -0.53 / -0.42 / -0.31 | 75 / 81 / 86° | 2.0 |
| ꯦ | 23 | 0.30 / 0.36 / 0.48 | -0.02 / 0.03 / 0.09 | -0.53 / -0.41 / -0.32 | 91 / 105 / 128° | 1.2 |
| ꯩ | 3 | 0.26 (median) | 0.32 | -0.27 | 96° | 1.5 |
| ꯪ | 2 | 0.29 (median) | 0.18 | -0.46 | 97° | 1.9 |

The writer draws ꯥ as TUMMHCD's writers do, a straight stroke falling to the right (TUMMHCD's
191 validation images of ꯥ, drawn at the size the synthesiser gives them: 97% fall to the
right, median 51°, elongation 6.4; the font's ꯥ: 46°, 3.4), but puts it after the letter at
the top, where ꯣ and ꯦ stand, reaching down beside the letter. ꯩ and ꯪ go there too. Print,
and every synthetic word so far, put ꯥ over the right half of the letter and wholly above its
top line: on 1,000 validation words drawn with TUMMHCD's validation characters, ꯥ is centred
0.29 L before the letter's right edge (10th to 90th percentile -0.47 to 0.65; the upper tail is
signs over a narrow letter), its bottom 0.07 L above the top line
(`results/round3_sign_geometry_tummhcd.json`; with the font's characters -0.34 and 0.08 L,
`results/round3_sign_geometry_font.json`). A recogniser that has only seen ꯥ over the letter
meets a stroke in ꯣ's and ꯦ's place and reads ꯣ or ꯦ, or nothing.

## 3. The change (`mayek_words/synth.py`, `Config.p_marks_beside`)

With chance `p_marks_beside` a word has every sign above (ꯥ ꯩ ꯪ) on the line after the
character before it, as ꯣ is: its left end -0.1 to 0.1 L from that character's right edge
(writer -0.14 to 0.12), its top 0.06 to 0.30 L above its letter's top (writer 0.08 to 0.30),
reaching at most 0.5 L below that top (a longer sign is drawn smaller: TUMMHCD's ꯩ is 1.5
times the font's height). From there on it is placed like ꯦ, ꯣ and ꯧ: never nearer the next
letter than its own, never touched by it, never into its letter. The shapes stay TUMMHCD's.

- With `p_marks_beside = 0`, the default, no random number is drawn for it and every word
  renders as before (30 renders compared byte for byte before and after the change;
  `tests/test_synth.py` checks that nothing is drawn for it): the fixed sets and the earlier
  runs stay valid.
- The same 1,000 words with every sign above beside (TUMMHCD's validation characters): ꯥ is
  centred 0.20 L after its letter, its top 0.15 L above the letter's top and its bottom 0.27 L
  below it, at 52° (writer 0.18, 0.18, -0.19, 52°; `results/round3_sign_geometry_tummhcd.json`;
  font characters 0.24 L, `results/round3_sign_geometry_font.json`).
- The placement rules hold for the new placement (`scripts/check_signs.py --marks-beside`,
  TUMMHCD's validation characters, `results/sign_placement_beside_tummhcd.json`): the body of
  ꯥ nearer the next letter than its own in 0 of 316 words, of ꯩ in 1 of 303, of ꯪ in 0 of 220;
  never touching its letter or crossing into it; the next letter never touching it (with the
  font's characters, 150 words a sign, none: `results/sign_placement_beside_font.json`).
- Training: `TrainConfig.marks_beside`, `scripts/train_recogniser.py --marks-beside`;
  `scripts/render_words.py --marks-beside` for fixed sets. A run refuses training words
  drawn with another setting.

**Round 3** is the final recipe of round 2 (ConvNeXt-T from TUMMHCD, 10% scrambled words, 60,000
steps, seeds 0 and 1 with training words 1000 and 1001) with `marks_beside = 0.5`: half the
training words that have a sign above have it beside. Nothing else changes, so round 2 against
round 3 measures this change alone.

## 4. Protocol

- **The first 100 real words** were round 2's test, used once; those results stand. Their
  errors shaped round 3, so for round 3 they are a development set: they show whether the
  change does what it was made for (`results/round3_dev_<run>.json`), not how well round 3
  reads unseen handwriting.
- **The test of round 3**: new words by the same writer, pages 7-12 of the writing pages
  (items 107-209: 100 words and 3 numbers, 623 characters, 56 items with a sign above: 61 ꯥ,
  6 ꯩ, 7 ꯪ), written as before and cut into `real_test2.tar`; round 2 and round 3, both seeds,
  read them once (`results/phase3_real2_<run>.json`). The writer is the one whose hand shaped
  the change, so this measures the fix for this hand. Whether other writers place their signs
  the same way is not known; the synthetic words now hold both placements.
- **Synthetic**: the checkpoint and the language model's weights are chosen on `val.tar`
  (print placement) as in round 2. `val_beside.tar`, the same 5,000 words and word styles with
  every sign above beside, shows the effect of the placement alone (round 2 against round 3);
  the synthetic test is read once.

## 5. The notebook (`notebooks/round3_signs_beside.ipynb`)

1. A look at the new training words: the same words with the signs above as in print and
   beside; the first 48 words of the first run; TUMMHCD's images of ꯥ, ꯩ, ꯪ.
2. `val_beside.tar`, and the placement measured on the real words, `val.tar`, `val_beside.tar`
   and TUMMHCD's sign images (`results/round3_sign_geometry.json`), and the placement rules
   (`results/sign_placement_beside_tummhcd.json`).
3. Round 2 reads `val_beside.tar` (`results/round3_val_beside_<run>.json`).
4. Training, two runs (about 2.5 hours each).
5. Validation of round 3 on both sets (`results/phase2_val_<run>.json`,
   `results/round3_val_beside_<run>.json`).
6. The 100 real words, as a development set (`results/round3_dev_<run>.json`,
   `results/round3_dev_comparison.json`).
7. The new real words, once (`RUN_REAL2`; `results/phase3_real2_<run>.json`,
   `results/phase3_real2_comparison.json`).
8. The synthetic test, once (`RUN_TEST`; `results/phase2_test_<run>.json`).

Checked here end to end (a dry run: Colab stubbed, the font's characters for TUMMHCD, the
development lexicon, tiny networks, the 100 real words).
