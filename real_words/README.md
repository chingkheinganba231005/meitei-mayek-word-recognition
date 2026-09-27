# Handwritten Meitei Mayek words: a first real test set

100 handwritten Meitei Mayek words (90 words of running text and 10 numbers; 592 characters,
52 of the 54 characters of everyday spelling), written by one native writer, the author, with
an Apple Pencil on an iPad. It is the real-handwriting test set of the paper "Beyond isolated
characters: the first segmentation-free recogniser of handwritten Meitei Mayek words"
(C. Rajkumar, 2026) and of this repository's recogniser.

## Files

| File | What |
|---|---|
| `images/000001.png` ... | one word per image, greyscale, cut from its own ink with a margin of 0.15 of its height |
| `labels.tsv` | image file, then the word as written (tab separated, UTF-8) |
| `manifest.json` | each item: its number on the writing pages, page, kind (`lexicon` or `number`), flags, size; the corrections made to labels |
| `corrections.json` | the corrections, as given to `scripts/release_real_set.py` |

## How it was made

- **Words:** distinct words of the test part of the project's word list (Meitei Wikipedia and
  FineWeb-2 text, split by word with a hash), so none of them is a training word of the
  recogniser; 2 to 14 characters, seen at least twice in the text; every letter and sign
  about eight times where the test words allow; 5% numbers, 2% with a full stop. Everyday
  spelling: ꯏ for every i.
- **Writing:** the words were printed on A4 pages, each above an empty box; the writer wrote
  each word in its box on an iPad (Markup, one pen about 0.1 of the letter height wide) and
  exported the pages as PDF. Items 1 to 100 of the pages were written.
- **Cutting:** `scripts/cut_writing_pages.py` aligned each page by its corner squares, took the
  ink darker than the printed page and gave each piece of ink to its box. 20 words are flagged
  `outside` (a stroke crosses the box line); all are complete.
- **Correction:** item 35 (`000035.png`) was printed with a full stop that the writer did not
  write; its label was corrected to the word as written (27 September 2026). The recorded
  results use the corrected label.

## Use

A test set: scores on it are in the repository's `results/phase3_real_*.json`. One writer,
digital ink and 100 words make it a first check on real handwriting, not a benchmark of
writers: the 95% interval of a word accuracy of 80% on 100 words is about 71% to 87%.

## Licence and citation

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`). The writer is the
author, and the set contains no other person's data. Please cite the paper and the
repository, https://github.com/chingkheinganba231005/meitei-mayek-word-recognition.
