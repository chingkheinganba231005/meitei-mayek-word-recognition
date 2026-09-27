# Handwritten Meitei Mayek word recognition

The first segmentation-free recogniser of handwritten Meitei Mayek words. It reads a whole
word image without cutting it into characters, and it is trained only on synthetic words
composed from the isolated characters of the Tezpur University Meitei Mayek Handwritten
Character Database (TUMMHCD).

- **Demo** (runs in the browser): [huggingface.co/spaces/Chingkheinganba/handwritten-meitei-mayek-word-recognition](https://huggingface.co/spaces/Chingkheinganba/handwritten-meitei-mayek-word-recognition)
- **Trained model and language model:** [huggingface.co/Chingkheinganba/handwritten-meitei-mayek-word-recognition](https://huggingface.co/Chingkheinganba/handwritten-meitei-mayek-word-recognition)
- **Real handwritten word set:** [`real_words/`](real_words) (100 words, CC BY 4.0)

## Results

Character error rate (CER) and word error rate (WER) of the final model (ConvNeXt-T encoder
initialised from TUMMHCD, bidirectional LSTM, CTC, character 6-gram language model; mean of
two training seeds):

| Test set | CER | WER |
|---|---:|---:|
| 5,000 synthetic words (TUMMHCD test characters) | 0.24% | 1.57% |
| The same words, character ensemble with perfect segmentation, perfect zones and the same language model | 4.26% | 20.98% |
| 100 real handwritten words (one native writer, tablet, words never seen in training) | 2.96% | 16.0% |

Every number comes from a results file in [`results/`](results), written by the code.

## How it works

1. **Synthetic words** (`mayek_words`). TUMMHCD's characters are 24 x 24 images with the ink
   stretched to the frame, so their size and position are lost. A word is laid out in units of
   the letter height from the Noto Sans Meetei Mayek font; each character's handwritten size
   is recovered from its stroke thickness; letter spacing follows measurements on real
   handwriting; vowel signs are placed on the ink and kept with their syllable; characters
   are matched in style. Words are rendered while training (a few milliseconds each).
2. **Recogniser** (`mayek_htr`). The stem and first three stages of ConvNeXt-T (the third
   stage halving the height only), a two-layer bidirectional LSTM and connectionist temporal
   classification over the 54 characters of everyday spelling. Decoding is greedy or by beam
   search with a Kneser-Ney character n-gram language model.
3. **Spelling of i.** ꯢ and ꯏ are one letter in everyday writing; the recogniser reads one
   letter and can render the standard spelling by rule (`charset.standard_spelling`).
4. **Real words.** Words are printed on A4 pages above empty boxes, written by hand, and cut
   out automatically (`mayek_htr/pages.py`, `scripts/make_writing_pages.py`,
   `scripts/cut_writing_pages.py`).
5. **Browser demo.** The network runs as ONNX in the browser, with the preprocessing, the
   language model and the beam search ported to JavaScript (`mayek_htr/web.py`, `web/`).

## Using the trained recogniser

```bash
git clone https://github.com/chingkheinganba231005/meitei-mayek-word-recognition
cd meitei-mayek-word-recognition
pip install -r requirements.txt -r requirements-htr.txt      # PyTorch: see pytorch.org
huggingface-cli download Chingkheinganba/handwritten-meitei-mayek-word-recognition --local-dir word_model
```

```python
from mayek_htr.reader import Reader

reader = Reader("word_model")
print(reader.read("word.jpg"))                  # one word, dark ink on light paper
print(reader.read("word.jpg", use_lm=False))    # the network alone (greedy decoding)
```

## Reproducing the results

The notebooks run on Google Colab with a working folder on Google Drive; training needs one
A100 GPU (about 2.5 hours per run).

| Notebook | What it does |
|---|---|
| `notebooks/1_data_audit.ipynb` | audits TUMMHCD and counts the Meitei Mayek text corpora |
| `notebooks/2_synthetic_words.ipynb` | character stores, handwritten sizes, lexicon, fixed synthetic sets |
| `notebooks/3_recogniser.ipynb` | language model, training, validation, baseline, synthetic test |
| `notebooks/4_real_words_and_demo.ipynb` | the real test, and the Hugging Face model and demo |
| `notebooks/5_signs_beside.ipynb` | training with the signs above placed as the real writer places them |
| `notebooks/collect_results.ipynb` | gathers the small result files from Drive in one zip |

Tests: `pip install -r requirements-dev.txt -r requirements-htr.txt && pytest -q`.

## Data

- **TUMMHCD** is available from its authors at
  <http://agnigarh.tezu.ernet.in/~sarat/resources.html>; it is not included here.
- **Text** for the lexicon and the language model: the Meitei Wikipedia dump (CC BY-SA 4.0)
  and FineWeb-2's Meitei Mayek text (ODC-By 1.0), downloaded by notebook 1.
- **Real word set:** [`real_words/`](real_words), 100 handwritten words with their labels.

## Repository

```
mayek_words/   synthetic words: alphabet, character stores, layout, drawing, lexicon
mayek_htr/     the recogniser: data, augmentation, model, training, language model,
               decoding, metrics, writing pages, reader, browser export
scripts/       command-line tools used by the notebooks (each explains itself)
notebooks/     the experiments, in order
web/, space/   the browser demo and the Hugging Face cards
real_words/    the real word set
results/       results files written by the code
tests/         pytest
```

## Citation

```bibtex
@misc{rajkumar2026words,
  author       = {Rajkumar, Chingkheinganba},
  title        = {Handwritten {Meitei Mayek} word recognition},
  year         = {2026},
  howpublished = {\url{https://github.com/chingkheinganba231005/meitei-mayek-word-recognition}}
}
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The real word set: CC BY 4.0. The bundled font
(`mayek_words/assets/NotoSansMeeteiMayek-Regular.ttf`): SIL Open Font Licence 1.1. The
language model on Hugging Face: CC BY-SA 4.0 (built from Wikipedia and FineWeb-2 text).
