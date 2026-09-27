---
title: Handwritten Meitei Mayek Word Recognition
emoji: ✍️
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Handwritten Meitei Mayek words read in your browser
models:
  - {hf_id}
---

# Handwritten Meitei Mayek word recognition

Write a whole Meitei Mayek word, or upload a photo of one, and see what the first
segmentation-free recogniser of handwritten Meitei Mayek words reads. It reads the word
without cutting it into characters (a convolutional encoder, a
bidirectional LSTM and CTC over 54 characters), and a character language model helps it
choose between look-alikes such as ꯦ and ꯰, or ꯨ and ꯁ. Words are read in everyday
spelling (ꯏ for every i); the standard spelling (ꯢ after ꯥ, ꯣ, ꯨ) is one tick away.

Everything runs in your browser with ONNX Runtime Web: nothing you write or upload is sent
anywhere. {results}

The trained network, the language model and how to use them in Python:
[{hf_id}](https://huggingface.co/{hf_id}). Code and experiments: [{repo_url}]({repo_url}).
Characters one at a time: [the character demo](https://huggingface.co/spaces/Chingkheinganba/handwritten-meitei-mayek-recognition).
