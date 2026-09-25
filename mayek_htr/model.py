"""The word recogniser (PyTorch): a convolutional encoder, a bidirectional LSTM and a CTC
output layer over the 54 characters and the blank.

Encoders:
- ``ConvNeXtEncoder``: the stem and first three stages of ConvNeXt-T (timm), one of the
  first paper's three architectures. The third stage's downsampling is made to halve the
  height only, so a 64 px word comes out as 4 rows of 384 channels, one column per 8 px:
  about four columns per letter, enough for a letter with signs above, below and beside
  it (CTC needs a column per character). Initialised from the first paper's network
  trained on TUMMHCD (``load_tummhcd``), from ImageNet, or at random.
- ``SmallCNN``: a plain CNN trained from scratch, as in CRNN (Shi et al., TPAMI 2017), the
  usual baseline for Indic handwritten words (Gongidi and Jawahar's adds a spatial
  transformer and a ResNet): 4 rows of 512 channels, one column per 4 px.

The columns (all rows and channels of a column together, so a sign's height above or
below the letter is kept) are projected, read by the LSTM in both directions (padding
excluded) and classified.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .images import HEIGHT
from .labels import NUM_CLASSES

MEAN, STD = 0.15, 0.3    # input normalisation (ink 1, paper 0)


class _PadRight(nn.Module):
    """A (2, 2) convolution with stride (2, 1), the input padded by one column on the right,
    so the width is kept while the height halves."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.conv.stride = (2, 1)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 0)))


# the ImageNet weights the first paper started from
TIMM_ARCH = {"convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k"}


class ConvNeXtEncoder(nn.Module):
    stride = (16, 8)

    def __init__(self, arch="convnext_tiny", init="imagenet", drop_path=0.1):
        super().__init__()
        import timm

        net = timm.create_model(TIMM_ARCH.get(arch, arch), pretrained=init == "imagenet", num_classes=0,
                                in_chans=1, drop_path_rate=drop_path)
        self.stem = net.stem
        self.stages = nn.ModuleList(net.stages[:3])
        self.stages[2].downsample[1] = _PadRight(self.stages[2].downsample[1])
        self.channels = net.feature_info[2]["num_chs"]

    def forward(self, x):
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return x

    def load_backbone(self, state):
        """Weights of a whole timm ConvNeXt (keys as in net.state_dict()); the fourth stage,
        the final norm and any head are ignored. Returns the keys that were not found."""
        own = self.state_dict()
        mapped = {}
        for k, v in state.items():
            if k.startswith("stem."):
                mapped[k] = v
            elif k.startswith("stages."):
                i = int(k.split(".")[1])
                if i < 3:
                    mapped[k.replace("downsample.1.", "downsample.1.conv.") if i == 2 else k] = v
        missing = [k for k in own if k not in mapped]
        own.update({k: v for k, v in mapped.items() if k in own})
        self.load_state_dict(own)
        return missing


class SmallCNN(nn.Module):
    stride = (16, 4)
    channels = 512

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.body = nn.Sequential(
            *block(1, 64), nn.MaxPool2d(2),
            *block(64, 128), nn.MaxPool2d(2),
            *block(128, 256), *block(256, 256), nn.MaxPool2d((2, 1)),
            *block(256, 512), *block(512, 512), nn.MaxPool2d((2, 1)))

    def forward(self, x):
        return self.body(x)


class Recogniser(nn.Module):
    def __init__(self, encoder, height=HEIGHT, hidden=256, layers=2, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        rows = height // encoder.stride[0]
        self.proj = nn.Sequential(nn.Linear(encoder.channels * rows, 2 * hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(2 * hidden, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def frames(self, widths):
        """Number of output columns of a word `widths` pixels wide."""
        s = self.encoder.stride[1]
        return torch.div(widths + s - 1, s, rounding_mode="floor")

    def forward(self, x, widths):
        """x: float (B, 1, H, W) normalised; -> (logits (B, T, classes), lengths (B,))."""
        f = self.encoder(x)                                        # (B, C, h, T)
        B, C, h, T = f.shape
        with torch.autocast(device_type=f.device.type, enabled=False):   # the LSTM in float32
            f = self.proj(f.float().permute(0, 3, 1, 2).reshape(B, T, C * h))
            lengths = self.frames(widths).clamp(1, T)
            packed = nn.utils.rnn.pack_padded_sequence(f, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
            return self.head(out), lengths


def build(encoder="convnext_tiny", init="imagenet", tummhcd_dir=None, height=HEIGHT, **kw):
    """encoder: 'convnext_tiny' or 'small_cnn'; init: 'tummhcd', 'imagenet' or 'none'."""
    if encoder == "small_cnn":
        enc = SmallCNN()
    else:
        enc = ConvNeXtEncoder(encoder, init=init if init == "imagenet" else "none")
        if init == "tummhcd":
            missing = enc.load_backbone(load_tummhcd(tummhcd_dir, encoder))
            if missing:
                raise RuntimeError(f"TUMMHCD weights lack {len(missing)} encoder tensors, e.g. {missing[:3]}")
    return Recogniser(enc, height=height, **kw)


def find_model_dir(root):
    """The first paper's model folder (written by ``mayek.recognizer.export``: config.json
    with its members and one state dict per member), anywhere below root, such as a
    download of the Hugging Face release; of several, the one with the most members."""
    found = []
    for p in sorted(Path(root).rglob("config.json")):
        try:
            members = json.loads(p.read_text(encoding="utf-8")).get("members")
        except (ValueError, UnicodeDecodeError, AttributeError):
            continue
        if members:
            found.append((len(members), p.parent))
    if not found:
        raise FileNotFoundError(f"no model folder (config.json with members) under {root}")
    return max(found, key=lambda f: f[0])[1]


def load_tummhcd(model_dir, arch="convnext_tiny"):
    """The backbone weights of the first paper's network of this architecture on grey input
    and without size features (see ``find_model_dir``)."""
    folder = find_model_dir(model_dir)
    members = json.loads((folder / "config.json").read_text(encoding="utf-8"))["members"]
    ok = [m for m in members if m["cfg"]["arch"].split(".")[0] == arch and m["cfg"]["channels"] == "gray"]
    if not ok:
        raise KeyError(f"no grey {arch} member in {folder}: {[m['name'] for m in members]}")
    ok.sort(key=lambda m: bool(m["cfg"].get("meta")))           # plain before size-aware
    state = torch.load(folder / ok[0]["file"], map_location="cpu", weights_only=True)
    return {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
