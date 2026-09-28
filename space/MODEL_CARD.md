---
license: mit
library_name: pytorch
pipeline_tag: image-to-text
tags:
  - handwriting-recognition
  - meitei-mayek
  - ocr
  - ctc
---

# Handwritten Meitei Mayek word recognition

The first segmentation-free recogniser of handwritten Meitei Mayek words: the stem and first
three stages of ConvNeXt-T (started from a network trained on the TUMMHCD characters), a two-layer bidirectional LSTM and CTC over the 54 characters of everyday
spelling, decoded by beam search with a character 6-gram language model (interpolated
Kneser-Ney). Trained only on synthetic words composed from TUMMHCD training characters.{training}

{results}{versions}

| File | What |
|---|---|
| `recogniser.pt` | the network (EMA weights) and its training settings; run `{run}`, step {step} |
| `char_lm.pkl` | the character language model (`mayek_htr.lm.CharLM`) |
| `decoding.json` | the language model's weight and the bonus per character, chosen on the synthetic validation set |
| `results.json` | the scores above, as written by the evaluation |
| `web/model.onnx`, `web/lm.bin.gz` | the same, for the browser demo (weights stored as float16) |

## Use

```bash
git clone {repo_url}
cd meitei-mayek-word-recognition && pip install -r requirements.txt -r requirements-htr.txt
huggingface-cli download {hf_id} --local-dir word_model
```

```python
from mayek_htr.reader import Reader

reader = Reader("word_model")
print(reader.read("word.jpg"))                  # one word, dark ink on light paper
print(reader.read("word.jpg", use_lm=False))    # the network alone (greedy decoding)
```

The image is cut to its ink and prepared as in the evaluation; the language model's weight
comes from `decoding.json`.

Words are written in everyday spelling (ꯏ for every i); `mayek_words.charset.standard_spelling`
renders the standard spelling (ꯢ after ꯥ, ꯣ, ꯨ).

## Limitations

Real handwriting has been measured on one writer on a tablet only; other hands, pens and
photographs will be read less reliably. Characters outside TUMMHCD (lum iyek ꯬, the
Extensions) cannot be read.

## Licences and data

Network: MIT, like the code. The language model is built from Meitei Mayek Wikipedia
(CC BY-SA 4.0) and FineWeb-2 (ODC-By 1.0) and is released under CC BY-SA 4.0, with
attribution to those sources. Characters: D. Hijam and S. Saharia, "On developing complete
character set Meitei Mayek handwritten character database", *The Visual Computer* 38,
525–539 (2022).

Demo: [huggingface.co/spaces/{hf_id}](https://huggingface.co/spaces/{hf_id}).
Code and every experiment: [{repo_url}]({repo_url}).
