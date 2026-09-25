"""Word images as the recogniser sees them: ink 1, paper 0, a fixed height, any width.

The same steps serve the synthetic words and, later, photographed real words:
- contrast: the paper level is the 90th percentile of the image and the ink level its
  1st percentile, so pale writing and grey paper come out alike;
- height: scaled to HEIGHT pixels with the aspect ratio kept (a synthetic word is about
  65 px high with letters 32 px high, so it is used close to its own size); a word
  wider than MAX_WIDTH is squeezed to it;
- batches: right-padded with paper to a common width.
Arrays are kept as uint8 (0-255) until they reach the GPU.
"""

import numpy as np
from PIL import Image

HEIGHT = 64
MAX_WIDTH = 1024


def ink_image(gray):
    """uint8 greyscale, dark ink on light paper -> float32 in [0, 1] with ink high."""
    g = np.asarray(gray, np.float32)
    paper, ink = np.percentile(g, 90), np.percentile(g, 1)
    return np.clip((paper - g) / max(paper - ink, 16.0), 0, 1).astype(np.float32)


def resize_height(x, height=HEIGHT, max_width=MAX_WIDTH):
    """float image -> height rows, width scaled in proportion (at least 1, at most max_width)."""
    h, w = x.shape
    new_w = int(min(max(round(w * height / h), 1), max_width))
    if (h, w) == (height, new_w):
        return x
    shrink = h / height
    im = Image.fromarray(np.ascontiguousarray(x, np.float32), "F")
    if shrink > 2:  # average over the source pixels when shrinking a lot (photos)
        im = im.reduce(int(shrink // 2) or 1)
    return np.asarray(im.resize((new_w, height), Image.BILINEAR), np.float32)


def normalise(gray, height=HEIGHT, max_width=MAX_WIDTH):
    """uint8 greyscale word image -> uint8 (height, width), ink 255, paper 0."""
    x = resize_height(ink_image(gray), height, max_width)
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def crop_ink(gray, margin=0.15):
    """Cut a word out of a larger photo: the ink's bounding box plus margin x its height
    on every side. For real images; the synthetic words come cropped."""
    x = ink_image(gray)
    ys, xs = np.nonzero(x > 0.5)
    if not len(xs):
        return np.asarray(gray)
    pad = int(margin * (ys.max() - ys.min() + 1)) + 1
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, x.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, x.shape[1])
    return np.asarray(gray)[y0:y1, x0:x1]


def pad_batch(images, multiple=32):
    """uint8 (H, w_i) images -> uint8 (B, H, W) with W the largest w_i rounded up to
    `multiple`, and the widths."""
    widths = np.array([im.shape[1] for im in images], np.int64)
    W = int(-(-widths.max() // multiple) * multiple)
    out = np.zeros((len(images), images[0].shape[0], W), np.uint8)
    for k, im in enumerate(images):
        out[k, :, :im.shape[1]] = im
    return out, widths
