# Phase 3: the real word set

Started 26 September 2026. **Status (27 September): the owner wrote 100 of the 500 items
and decided that these 100 words are the real test set for now (section 9); the test over
the seven Phase 2 runs, once, and a Hugging Face demo like the first project's (section 10)
are ready to run in `notebooks/phase3_real_words_and_demo.ipynb`.**

## 1. The owner's decision (26 September 2026)

> "I dont have any people to write actual words for me. I want to do it myself. I can write
> on my ipad many words in one page for many pages and you can crop them into the required
> real word test dataset."

So the real set is written by one writer, the owner (a native writer), with an Apple Pencil
on an iPad, on pages that print each word above an empty box. This replaces the plan of 50
or more volunteer writers. What it means for the evaluation:

- The words are text-disjoint from training: test words of the lexicon, which neither the
  recogniser nor the language model has seen. The writer is not one of TUMMHCD's. But one
  writer shows how the recogniser reads one unseen hand, not how much that varies from hand
  to hand; results must be reported as such.
- Digital ink differs from TUMMHCD's scans of paper: even strokes of the app's pen on white,
  no paper texture. This is a second change of domain besides the writer's.
- Contribution (3) of the novelty statement ("a public, consented, writer-disjoint real word
  set (50+ writers)") no longer holds as written; a rewording is proposed in section 5.
- The same pages can be printed, written on paper and photographed: the cutter lines up
  photos too (section 4). More writers can be added later without changing anything.

## 2. The pages (`mayek_htr/pages.py`, `scripts/make_writing_pages.py`)

- **Page:** A4 at 200 dpi (1654 × 2339 px). Title and page number at the top, two lines of
  instructions ("Copy each word into the box below it, in black, in your everyday
  handwriting. To correct a word, erase it and write it again. Leave a box empty to skip its
  word."), a black square in each corner.
- **Items:** each word printed with its item number above an empty box with light blue
  lines. A box is 2.2 cm high (room for signs above and below letters about 9 mm high) and
  as wide as the word written at that size (1.25 times its printed width plus 1.4 cm, at
  least 4.2 cm). Boxes are packed in rows (a row takes the next word, then any of the next
  40 that still fit) and a row's boxes are widened to fill it; eight rows a page, about 17
  to 19 words. Items are numbered in reading order.
- **Printing:** the words are shaped with HarfBuzz (Pillow with RAQM; the script refuses to
  print without it, since unshaped signs would sit in the wrong place) in the bundled Noto
  Sans Meetei Mayek, letters 3.9 mm high.
- **PDF:** each page a lossless 200 dpi image on an A4 page; the layout (every item's page,
  box, text and kind) is embedded in the PDF as `manifest.json`, so the blank template PDF is
  all the cutter needs besides the written pages.

## 3. Words to write (`pages.choose_items`)

- Distinct words whose hash split (`lexicon.split_of`) is the test split: never training
  words, whatever lexicon they come from. 2 to 14 characters, no digit or full stop, a count
  of at least 2 in the corpora (words seen once are often typing errors).
- Drawn without replacement with probability proportional to count ** 0.5, as the
  synthetic sets draw words. Before that, every letter and sign gets 8 occurrences where the
  test words have them (rarest first; words seen once allowed), so that rare letters such as
  ꯘ, ꯓ, ꯙ appear at all and the pairs ꯦ/꯰, ꯨ/ꯁ, ꯗ/ꯘ can be measured.
- 5% numbers (`lexicon.number`, 1 to 4 digits; each digit at least 4 times), 2% of the
  words ending with a full stop (꯫), as in the synthetic sets; items in random order.
- `--summary` records the counts only (the pages, the kinds, each character's count), never
  the words: word lists stay out of the repository.

## 4. Cutting (`pages.Cutter`, `scripts/cut_writing_pages.py`)

The written pages, exported from the iPad as a PDF (Markup's ink is rendered with the
page) or as images, in any order:

1. **Alignment:** the four corner squares give a projective map onto the template; a page
   exported as printed is used as it is. For photos, the paper is first made white by
   dividing by the paper level around each pixel.
2. **Which page:** the page is matched to the template page whose distinct printed ink
   (words, item numbers, page number) it shows entirely; a page of another template, or an
   unclear match, is refused.
3. **Handwriting:** what is at least 60 grey levels darker than the printed page nearby
   (within 2 px, 4 px for a resampled page), so the printed words and box lines drop out, in
   any pen colour.
4. **Boxes:** each connected piece of handwriting belongs to the box it overlaps most
   (flagged `shared` if it overlaps two); a piece outside every box but within 5 mm of one
   goes to the nearest (flagged `outside`), so a word written over its box line stays whole.
   Handwriting away from every box is listed per page as stray.
5. **Word image:** the word's own handwriting on white, cut with a margin of 0.15 times its
   height (as `images.crop_ink`); a box without handwriting is empty and left out.

Output: `images/NNNNNN.png` (the item number) and `labels.tsv` as in the synthetic sets,
read by `mayek_htr.data.FixedSet`, with `manifest.json` (every item's text, kind, page,
status and flags; `FixedSet` takes the kinds from it) instead of `config.json`; a contact
sheet per page, the printed word beside the cut-out handwriting, to check every word.

**Checked on simulated pages** (synthetic words from TUMMHCD validation characters written
into 120 boxes of a 7-page sample template, from the development word list): pages exported
in reverse order, one as a scaled and rotated image, one with a vector ink annotation as
Markup writes it, and one given a second time as a phone photo (perspective, uneven light,
blur, JPEG; the later copy is used). Every page was found and the pages of another template
refused; the two empty boxes found; the words over a box line and a stroke written above
its box kept and flagged; a word in blue cut; the stray scribble reported. The cut words
keep their ink: dark pixels 1.00 to 1.01 times those written on the exported pages (81
words, median 1.003), 0.99 to 1.01 on the scaled image, 0.89 to 1.04 on the photo (blur and
resampling move pixels across the threshold). The planted cases: a word partly over a
printed word 0.97, the extra annotation 1.40. Tests: `tests/test_pages.py`.

## 5. Decisions (proposed 26 September, confirmed by the owner 27 September 2026)

1. **Size:** 500 items (29 pages; about two hours of writing).
2. **Trial page first:** 17 validation words (never test words), to check the iPad's export
   and the cutting on real writing before the long pages; never scored.
3. **Contribution (3), reworded:** "a real handwritten word set, about 500 words written by
   a native writer on a tablet, text-disjoint from training, with a fixed protocol (CER,
   WER, the confusable pairs), released with the tools to add writers".
4. **Evaluation:** all seven Phase 2 runs, greedy and with the language model at the alpha
   and beta chosen on synthetic validation; the set used once. The baseline cannot run on
   real words: it needs the synthesiser's character boxes.

## 6. The trial page (written by the owner, 27 September 2026)

- **Export:** written in the iPad's Files app (Markup, iOS 18.5) and shared as a PDF. The
  page keeps its size and the embedded manifest; the drawing is saved as one annotation,
  which PyMuPDF renders with the page. Corner squares found, the page matched.
- **Cutting:** 17 of 17 words cut whole; 3 flagged `outside` (a stroke of a sign or of apun
  over a box line), all complete on the contact sheet; no stray ink.
- **Pen (`scripts/measure_strokes.py`, `results/phase3_trial_strokes.json`):** letters about
  12 mm high on the page, strokes about 0.5 mm (4 px at 200 dpi): 0.043 of the letter
  height, where the synthesiser's pens are 0.06-0.14 L. As the recogniser sees a word
  (64 px high) the strokes are half as thick as those of synthetic words: median 2.0 px
  against 4.0 px (synthetic 10th percentile 3.0), 0.061 against 0.121 of the ink band, ink
  6.7% of the image against 13.5% (500 synthetic words rendered here from the TUMMHCD
  validation characters). Writing on glass, letters come out large; a ballpoint on paper
  gives strokes about a tenth of the letter height.
- **Proposed (owner to decide):** a pen about twice as wide for the test pages (strokes
  about 1 mm at this letter size, 0.08-0.09 of the letter height, inside the training
  range), the same pen throughout; page 1 sent first to check it. Keeping the thin pen is
  possible, but the test would then measure strokes thinner than any the recogniser was
  trained on, on top of the change of writer.
- **Two pens tried on page 1 (owner, 27 September; `results/phase3_pen_choice.json`):** row
  1 (items 1-2) with a wide pen: strokes 2.2 mm, 0.15 of the letter height; at the
  recogniser's input 6.0 px, 0.194 of the ink band, ink 23.1%: thicker than nearly all
  training words (their 90th percentile is 5 px). Row 2 (items 3-5): strokes 1.2 mm, 0.10 of
  the letter height; 4.0 px, 0.129 of the band, ink 14.0%: as the training words (4.0 px,
  0.121, 13.5%). Recommended: the second pen for every page, items 1 and 2 rewritten with it.

## 7. The test pages (27 September 2026)

From the owner's test lexicon (`WORK/lexicon/test.tsv`: 3,761 words, all in the test split
by the hash; 3,725 usable, 1,834 seen at least twice), seed 0: 500 items on 29 pages (15 to
20 a page): 475 words and 25 numbers, 7 ending with a full stop, 3,110 characters (6.2 an
item, at most 14). Every letter and sign at least 8 times, except ꯘ: 5, every test word that
has it (ꯗ/ꯘ will rest on few ꯘ); ꯓ and ꯙ 8; digits 4 to 10 times each
(`results/phase3_pages.json`). The PDF carries the words (its manifest), so it stays off the
repository: Drive `WORK/real/real_words_pages.pdf`.

## 8. Writing the pages (for the owner)

1. Open the PDF on the iPad (Files, then Markup; or import it into GoodNotes or
   Notability).
2. Pen tool, black, about twice as wide as on the trial page (section 6), the same pen for
   every page. Write at your normal size with the page fitting the screen.
3. Copy each printed word into the box under it, as you write every day; copy it exactly,
   even if a word looks unfamiliar.
4. Stay inside the box where you can (a stroke over the line is recovered, but check it on
   the contact sheet). To correct a word, erase it with the eraser and write it again; do
   not cross it out. Leave a box empty to skip a word.
5. Export the whole PDF (Share, then Save to Files, or the app's PDF export) and send it;
   page 1 first, to check the pen.

## 9. The real test set: 100 words (owner's decision, 27 September 2026)

> "I have done 100 words . So i wanna only test using 100 words and launch a similar hugging
> face space demo for my final model as the character model which hit 98% accuracy . And in
> the future , i can collect more real data and improve it. For now this is okay."

- **Written:** items 1 to 100 of the test pages (pages 1-6; items 101-106 on page 6 and
  pages 7-29 not written, kept for later), with the pen chosen in section 6 throughout, also
  for items 1-2 (rewritten). Exported from the iPad's Markup as one PDF; pages 1-6 cut
  (`cut_writing_pages.py`).
- **The set (`results/phase3_real_set.json`):** 100 words, 90 from the lexicon and 10
  numbers; 593 characters; 52 of the 54 characters (not ꯳ and ꯓ; ꯘ, ꯙ, ꯚ and ꯴ once). For
  the confusable pairs: ꯦ 29, ꯰ 3, ꯨ 16, ꯁ 39, ꯗ 19, ꯘ 1, so ꯗ/ꯘ and ꯰ can hardly be measured
  here. Every word cut whole; 20 flagged `outside` (a stroke over a box line), all complete
  on the contact sheets; no stray ink. The owner checks on the sheets that each word matches
  its printed word before the test.
- **The pen (`results/phase3_real_strokes.json`):** the same on all six pages and as the
  training words at the recogniser's input: strokes 4.0 px, 0.125 of the ink band, ink 14.7%
  of the image (synthetic words 4.0 px, 0.121, 13.5%); about 1.1 mm on the page.
- **What 100 words can show:** a first measure on real handwriting. The intervals are wide:
  at 80 words right of 100 the 95% interval is 71-87%. Runs that differ by a few words cannot
  be told apart; the final recipe's two seeds are reported with their mean.
- **Contribution (3), to reword (proposal, owner to confirm):** "a first real handwritten
  word set, 100 words by a native writer on a tablet, text-disjoint from training, with a
  fixed protocol and the tools to extend it".
- **Protocol:** the notebook's section 1, with `RUN_REAL_TEST = True`, once: every Phase 2 run
  (the four of the second round, the three of the first) reads the 100 words with the
  settings it was given on the synthetic validation set (`eval_recogniser.py --tuned`),
  greedily and with the language model. Writes `results/phase3_real_<run>.json`, and per run
  `real_predictions.tsv` and `real_errors.png`. A run already read is not read again.

## 10. The demo (Hugging Face, as the first project's)

The first project's demo is a static Space: the network runs in the visitor's browser with
ONNX Runtime Web, nothing is uploaded, and the weights sit in a model repository. The word
demo does the same (`Chingkheinganba/handwritten-meitei-mayek-word-recognition`, model
repository and Space):

- **Network:** of the final recipe's two seeds, the one better on synthetic validation (lower
  CER with the language model, then WER): `round2_convnext_tummhcd_seed1` (CER 0.253% for
  both, WER 1.64% against 1.68%), with its decoding weights (alpha 0.5, beta 0). Exported to
  ONNX for one word of any width (`mayek_htr.web.export_onnx`: the word padded with paper to
  a multiple of 32 px, the LSTM reading only the word's own columns, as in the evaluation);
  weights stored as float16 (about 33 MB) unless that changes more than 1% of the greedy
  readings of synthetic validation words.
- **Language model in the browser:** `CharLM` written in back-off form (every n-gram seen with
  its log probability, every context seen with its log back-off weight), which gives exactly
  its probabilities; about 0.74 MB compressed for the development word list, expected about
  7-8 MB for the full one.
- **Page (`web/`):** write a word on a wide canvas (pen about a tenth of the letter height) or
  upload a photo; the reading with the language model, the network's own reading, the
  alternatives kept by the beam search, what the network sees, and the standard spelling on
  request (`charset.standard_spelling`: ꯢ after ꯥ, ꯣ, ꯨ). Preprocessing (`images.py`, with
  Pillow's resampling) and decoding (`decode.py`, `lm.py`) are ported to JavaScript.
- **Checks:** on test inputs the JavaScript gives the same normalised images, pixel for pixel
  (seven sizes, Pillow's reduce and resize and the width limit), the same language model
  probabilities and the same readings (40 test inputs, greedy and beam search with and
  without the language model) as Python (`tests/test_web.py`, with Node); the ONNX network
  matches PyTorch; in headless Chromium, writing on the page gives the reading Python gives
  for the same image. The notebook was dry-run end to end with stand-in networks.
- **Python:** `mayek_htr.reader.Reader("word_model").read("word.jpg")`, as in the model card.
- **Build and upload:** the notebook's sections 3 and 4 (`scripts/build_demo.py`; upload with
  a Hugging Face token in Colab secrets, `UPLOAD = True`). The cards quote the synthetic test
  and, after section 1, the real words.
- **Licences:** network MIT (as the code and the first project's weights); the language model
  is built from Wikipedia (CC BY-SA 4.0) and FineWeb-2 (ODC-By 1.0) and is released under
  CC BY-SA 4.0 with attribution, as the model card says.

