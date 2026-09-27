# Phase 3: the real word set

Started 26 September 2026. **Status (27 September): tools ready and tested; a trial page is
with the owner.** The pages of the test set are made once the owner sends the test lexicon
(`WORK/lexicon/test.tsv`) and confirms the proposals in section 5.

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

## 5. Proposals awaiting the owner

1. **Size:** 500 items (about 28 pages; about two hours of writing).
2. **Trial page first:** 17 validation words (never test words), to check the iPad's export
   and the cutting on real writing before the long pages; never scored.
3. **Contribution (3), reworded:** "a real handwritten word set, about 500 words written by
   a native writer on a tablet, text-disjoint from training, with a fixed protocol (CER,
   WER, the confusable pairs), released with the tools to add writers".
4. **Evaluation:** all seven Phase 2 runs, greedy and with the language model at the alpha
   and beta chosen on synthetic validation; the set used once. The baseline cannot run on
   real words: it needs the synthesiser's character boxes.

## 6. Writing the pages (for the owner)

1. Open the PDF on the iPad (Files, then Markup; or import it into GoodNotes or
   Notability).
2. Pen tool, black, a thin or medium width, as a ballpoint would write; the same pen for
   every page. Write at your normal size with the page fitting the screen; zooming in to
   write is fine if it keeps your size.
3. Copy each printed word into the box under it, as you write every day; copy it exactly,
   even if a word looks unfamiliar.
4. Stay inside the box where you can (a stroke over the line is recovered, but check it on
   the contact sheet). To correct a word, erase it with the eraser and write it again; do
   not cross it out. Leave a box empty to skip a word.
5. Export the whole PDF (Share, then Save to Files, or the app's PDF export) and send it.
