"""Reading words with a trained recogniser: an image in, its text out.

    from mayek_htr.reader import Reader
    reader = Reader("word_model")              # recogniser.pt, char_lm.pkl, decoding.json
    reader.read("word.jpg")                    # one word, dark ink on light paper

The image is cut to its ink (``images.crop_ink``), normalised as in the evaluation
(``images.normalise``), read by the network with the word padded to a multiple of 32 px,
and decoded by beam search with the language model at the weights chosen on validation
(``decoding.json``), or greedily without it. The browser demo does the same in JavaScript.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import images
from .decode import beam_search, greedy
from .lm import CharLM
from .model import MEAN, STD
from .train import load_checkpoint


def load_gray(image):
    """A path, a PIL image or an array -> uint8 greyscale array (transparency on white)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            paper = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(paper, image)
        return np.asarray(image.convert("L"))
    a = np.asarray(image)
    if a.ndim == 3:
        return np.asarray(Image.fromarray(a.astype(np.uint8)).convert("L"))
    return a.astype(np.uint8)


class Reader:
    def __init__(self, folder, device="cpu"):
        folder = Path(folder)
        self.device = device
        self.model, self.checkpoint = load_checkpoint(folder / "recogniser.pt", device)
        self.lm = CharLM.load(folder / "char_lm.pkl") if (folder / "char_lm.pkl").exists() else None
        f = folder / "decoding.json"
        self.decoding = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"alpha": 0.5, "beta": 0.0,
                                                                                      "beam": 16}

    @torch.no_grad()
    def logprobs(self, gray, crop=True):
        """Natural log probabilities (frames, 55) of one word image."""
        gray = load_gray(gray)
        norm = images.normalise(images.crop_ink(gray) if crop else gray)
        x, widths = images.pad_batch([norm])
        x = torch.from_numpy(x).unsqueeze(1).float().div_(255).to(self.device)
        logits, lengths = self.model((x - MEAN) / STD, torch.from_numpy(widths).to(self.device))
        return logits[0, :int(lengths[0])].float().log_softmax(-1).cpu().numpy()

    def read(self, image, crop=True, use_lm=True):
        """The word's text, in everyday spelling."""
        lp = self.logprobs(image, crop)
        if use_lm and self.lm is not None:
            d = self.decoding
            return beam_search(lp, self.lm, d["alpha"], d["beta"], d.get("beam", 16))
        return greedy(lp)
