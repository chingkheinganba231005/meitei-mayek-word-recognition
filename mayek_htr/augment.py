"""Augmentation of word batches on the GPU (PyTorch), towards photographed handwriting.

The synthesiser already varies size, spacing, slant, pen, blur and ink; this adds what a
phone photo of a page brings and TUMMHCD scans lack: small rotation and slant, smooth
local warps, thicker or thinner strokes, blur, paler ink, uneven shading of the paper,
ruled lines, and noise. Inputs are float (B, 1, H, W), ink 1 and paper 0; each word is
transformed about the centre of its own width (the rest of the batch is padding).
No flips: a mirrored Meitei Mayek character is another character or none.
"""

import math

import torch
import torch.nn.functional as F

AUG = dict(rot=2.0, shear=0.2, scale_x=(0.85, 1.0), scale_y=(0.85, 1.1), shift_y=0.05,
           elastic=1.5, p_elastic=0.5,                    # px, at 64 px height
           p_thick=0.15, p_thin=0.1,                      # stroke width +-1 px
           p_blur=0.25, blur=(0.5, 1.2),                  # sigma, px
           ink=(0.55, 1.0),                               # ink strength
           p_shade=0.5, shade=(0.05, 0.3),                # uneven grey paper
           p_lines=0.15, lines=(0.2, 0.55),               # a ruled line, its strength
           p_noise=0.5, noise=(0.01, 0.06))
PRESETS = {"full": AUG, "none": None,
           "light": dict(AUG, rot=1.0, shear=0.1, elastic=0.8, p_blur=0.1, ink=(0.8, 1.0), p_shade=0.2,
                         p_lines=0.05, p_noise=0.3)}


def _u(B, lo, hi, dev):
    return torch.empty(B, device=dev).uniform_(lo, hi)


def _chance(B, p, dev):
    return (torch.rand(B, device=dev) < p).float()


def _geometry(x, widths, a):
    B, _, H, W = x.shape
    dev = x.device
    ang = _u(B, -a["rot"], a["rot"], dev) * math.pi / 180
    sh = _u(B, -a["shear"], a["shear"], dev)
    sx, sy = _u(B, *a["scale_x"], dev), _u(B, *a["scale_y"], dev)
    ty = _u(B, -a["shift_y"], a["shift_y"], dev) * H
    # forward map: p -> c + t + A (p - c), A = [[sx, sh], [0, sy]] R(ang); sample at its inverse
    cos, sin = ang.cos(), ang.sin()
    A = torch.stack([torch.stack([sx * cos + sh * sin, -sx * sin + sh * cos], 1),
                     torch.stack([sy * sin, sy * cos], 1)], 1)                      # (B, 2, 2)
    inv = torch.linalg.inv(A)
    cx, cy = widths.to(dev).float() / 2, torch.full((B,), H / 2, device=dev)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    du = xs.view(1, H, W) - cx.view(B, 1, 1)
    dv = ys.view(1, H, W) - cy.view(B, 1, 1) - ty.view(B, 1, 1)
    px = cx.view(B, 1, 1) + inv[:, 0, 0].view(B, 1, 1) * du + inv[:, 0, 1].view(B, 1, 1) * dv
    py = cy.view(B, 1, 1) + inv[:, 1, 0].view(B, 1, 1) * du + inv[:, 1, 1].view(B, 1, 1) * dv
    if a["elastic"]:
        field = torch.randn(B, 2, 3, max(W // 32, 2), device=dev) * a["elastic"]
        field = field * _chance(B, a["p_elastic"], dev).view(B, 1, 1, 1)
        field = F.interpolate(field, size=(H, W), mode="bicubic", align_corners=False)
        px, py = px + field[:, 0], py + field[:, 1]
    grid = torch.stack([px / W * 2 - 1, py / H * 2 - 1], -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _blur(x, sigma):
    """Gaussian blur with one sigma per image (grouped convolution)."""
    B, _, H, W = x.shape
    r = 3
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype).view(1, -1)
    k = torch.exp(-t ** 2 / (2 * sigma.view(-1, 1) ** 2))
    k = k / k.sum(1, keepdim=True)
    y = F.conv2d(F.pad(x.view(1, B, H, W), (r, r, 0, 0), mode="replicate"), k.view(B, 1, 1, -1), groups=B)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(B, 1, -1, 1), groups=B)
    return y.view(B, 1, H, W)


def augment(x, widths, aug=AUG):
    """x float (B, 1, H, W) in [0, 1], ink high; widths (B,) of the words before padding."""
    if aug is None:
        return x
    B, _, H, W = x.shape
    dev = x.device
    x = _geometry(x, widths, aug)

    pick = torch.rand(B, device=dev).view(B, 1, 1, 1)
    thick, thin = F.max_pool2d(x, 3, 1, 1), -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(pick < aug["p_thick"], thick, torch.where(pick > 1 - aug["p_thin"], thin, x))

    blur = _chance(B, aug["p_blur"], dev).view(B, 1, 1, 1)
    if blur.any():
        x = blur * _blur(x, _u(B, *aug["blur"], dev)) + (1 - blur) * x

    x = x * _u(B, *aug["ink"], dev).view(B, 1, 1, 1)

    shade = F.interpolate(torch.rand(B, 1, 3, max(W // 48, 2), device=dev), size=(H, W), mode="bicubic",
                          align_corners=False).clamp(0, 1)
    level = _u(B, *aug["shade"], dev) * _chance(B, aug["p_shade"], dev)
    x = x + (1 - x) * shade * level.view(B, 1, 1, 1)

    has = _chance(B, aug["p_lines"], dev).view(B, 1, 1)
    row = _u(B, 0.55, 0.95, dev) * H
    slope = _u(B, -0.01, 0.01, dev)
    xs = torch.arange(W, device=dev).view(1, 1, W).float()
    ys = torch.arange(H, device=dev).view(1, H, 1).float()
    dist = (ys - row.view(B, 1, 1) - slope.view(B, 1, 1) * xs).abs()
    line = (1 - (dist - 0.5).clamp(0, 1)) * _u(B, *aug["lines"], dev).view(B, 1, 1) * has
    x = torch.maximum(x, line.unsqueeze(1))

    sd = _u(B, *aug["noise"], dev) * _chance(B, aug["p_noise"], dev)
    x = x + torch.randn_like(x) * sd.view(B, 1, 1, 1)
    return x.clamp(0, 1)
