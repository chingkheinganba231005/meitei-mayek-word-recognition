// Word image preprocessing, ported from mayek_htr/images.py (numpy and Pillow), so the
// browser prepares a word exactly as the recogniser was evaluated. Free of browser APIs,
// so tests/test_web.py checks it against the Python version with Node.
//
// Images are {w, h, data}: data a typed array in row-major order.

export const HEIGHT = 64;
export const MAX_WIDTH = 1024;
const f32 = Math.fround;

export function grayFromRGBA(rgba, w, h) {
  // PIL's convert("L"): the same weights and rounding
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;
  }
  return { w, h, data: out };
}

function percentile(sorted, q) {
  // numpy.percentile, linear method
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const t = pos - lo;
  const d = sorted[hi] - sorted[lo];
  return f32(t >= 0.5 ? sorted[hi] - d * (1 - t) : sorted[lo] + d * t);
}

export function inkImage(img) {
  // images.ink_image: ink 1, paper 0; paper = the 90th percentile, ink = the 1st
  const s = Float32Array.from(img.data).sort();
  const paper = percentile(s, 90);
  const ink = percentile(s, 1);
  const denom = Math.max(f32(paper - ink), 16);
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(f32(paper - img.data[i]) / denom);
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return { w: img.w, h: img.h, data: out };
}

export function cropInk(img, margin = 0.15) {
  // images.crop_ink: the ink's bounding box plus margin x its height on every side
  const x = inkImage(img);
  let y0 = Infinity, y1 = -1, x0 = Infinity, x1 = -1;
  for (let y = 0; y < img.h; y++) {
    for (let c = 0; c < img.w; c++) {
      if (x.data[y * img.w + c] > 0.5) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        if (c < x0) x0 = c;
        if (c > x1) x1 = c;
      }
    }
  }
  if (y1 < 0) return img;
  const pad = Math.trunc(margin * (y1 - y0 + 1)) + 1;
  const ya = Math.max(y0 - pad, 0), yb = Math.min(y1 + pad + 1, img.h);
  const xa = Math.max(x0 - pad, 0), xb = Math.min(x1 + pad + 1, img.w);
  const out = new img.data.constructor((yb - ya) * (xb - xa));
  for (let y = ya; y < yb; y++) out.set(img.data.subarray(y * img.w + xa, y * img.w + xb), (y - ya) * (xb - xa));
  return { w: xb - xa, h: yb - ya, data: out };
}

function reduce(img, k) {
  // PIL Image.reduce(k) of a float image: the mean of each k x k block (partial blocks
  // at the right and bottom averaged over what they hold)
  const w = Math.ceil(img.w / k), h = Math.ceil(img.h / k);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ya = y * k, yb = Math.min(ya + k, img.h);
    for (let x = 0; x < w; x++) {
      const xa = x * k, xb = Math.min(xa + k, img.w);
      let s = 0;
      for (let yy = ya; yy < yb; yy++) for (let xx = xa; xx < xb; xx++) s += img.data[yy * img.w + xx];
      out[y * w + x] = s / ((yb - ya) * (xb - xa));
    }
  }
  return { w, h, data: out };
}

function coefficients(inSize, outSize) {
  // Pillow's precompute_coeffs for the bilinear filter (support 1, widened when shrinking)
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    const k = [];
    let ww = 0;
    for (let x = xmin; x < xmax; x++) {
      const t = Math.abs((x - center + 0.5) * ss);
      const wt = t < 1 ? 1 - t : 0;
      k.push(wt);
      ww += wt;
    }
    out.push({ xmin, k: ww ? k.map((v) => v / ww) : k });
  }
  return out;
}

function resizeBilinear(img, w, h) {
  // PIL Image.resize((w, h), BILINEAR) of a float image: horizontal pass, then vertical
  let cur = img;
  if (w !== img.w) {
    const cx = coefficients(img.w, w);
    const out = new Float32Array(w * img.h);
    for (let y = 0; y < img.h; y++) {
      for (let x = 0; x < w; x++) {
        const { xmin, k } = cx[x];
        let s = 0;
        for (let i = 0; i < k.length; i++) s += img.data[y * img.w + xmin + i] * k[i];
        out[y * w + x] = s;
      }
    }
    cur = { w, h: img.h, data: out };
  }
  if (h !== cur.h) {
    const cy = coefficients(cur.h, h);
    const out = new Float32Array(cur.w * h);
    for (let y = 0; y < h; y++) {
      const { xmin, k } = cy[y];
      for (let x = 0; x < cur.w; x++) {
        let s = 0;
        for (let i = 0; i < k.length; i++) s += cur.data[(xmin + i) * cur.w + x] * k[i];
        out[y * cur.w + x] = s;
      }
    }
    cur = { w: cur.w, h, data: out };
  }
  return cur;
}

function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function resizeHeight(img, height = HEIGHT, maxWidth = MAX_WIDTH) {
  // images.resize_height
  const newW = Math.trunc(Math.min(Math.max(roundHalfEven((img.w * height) / img.h), 1), maxWidth));
  if (img.h === height && img.w === newW) return img;
  const shrink = img.h / height;
  let cur = img;
  if (shrink > 2) {
    const k = Math.trunc(Math.floor(shrink / 2)) || 1;
    if (k > 1) cur = reduce(cur, k);
  }
  return resizeBilinear(cur, newW, height);
}

export function normalise(img, height = HEIGHT, maxWidth = MAX_WIDTH) {
  // images.normalise: uint8 word image -> uint8 (height, width), ink 255, paper 0
  const x = resizeHeight(inkImage(img), height, maxWidth);
  const out = new Uint8Array(x.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(f32(f32(x.data[i]) * 255) + 0.5);
    out[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
  }
  return { w: x.w, h: x.h, data: out };
}

export function modelInput(norm, stride = 8, multiple = 32) {
  // the network's input as in the evaluation (train.prepare, images.pad_batch): ink in
  // [0, 1], padded with paper on the right to a multiple of 32 px; frames: the columns the
  // word itself fills (Recogniser.frames), the rest are padding
  const W = Math.ceil(norm.w / multiple) * multiple;
  const x = new Float32Array(norm.h * W);
  for (let y = 0; y < norm.h; y++)
    for (let c = 0; c < norm.w; c++) x[y * W + c] = norm.data[y * norm.w + c] / 255;
  const frames = Math.min(Math.max(Math.ceil(norm.w / stride), 1), W / stride);
  return { data: x, width: W, frames };
}

export function standardSpelling(text) {
  // charset.standard_spelling: ꯢ for the i after ꯥ, ꯣ or ꯨ (the standard spelling)
  const after = new Set(["ꯥ", "ꯣ", "ꯨ"]);
  const chars = Array.from(text);
  return chars.map((c, i) => (c === "ꯏ" && i > 0 && after.has(chars[i - 1]) ? "ꯢ" : c)).join("");
}
