import { cropInk, grayFromRGBA, modelInput, normalise, standardSpelling } from "./pipeline.js";
import { beamSearch, CharLM, greedy } from "./decode.js";

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };

let config = null;
let session = null;
let lm = null;
let loaded = false;          // the recogniser and, if any, the language model
let mode = "draw";           // "draw" | "upload"
let uploaded = null;         // ImageBitmap of the uploaded photo
let last = null;             // the last reading, redrawn when the spelling option changes

// ---------- model and language model ----------

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status(`${label} ${Math.round((100 * got) / total)}%`);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

async function load() {
  config = await (await fetch("config.json")).json();
  if (config.about) $("about").textContent = config.about;
  if (config.repo_url) $("repo").href = config.repo_url;
  const ortBase = new URL(config.ort_base || "ort/", location.href).href;
  await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = ortBase + "ort.wasm.min.js";
    s.onload = resolve;
    s.onerror = () => reject(new Error("could not load ONNX Runtime Web"));
    document.head.appendChild(s);
  });
  ort.env.wasm.wasmPaths = ortBase;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const bytes = await fetchWithProgress(config.model_url || "model.onnx",
    `Downloading the recogniser${config.model_mb ? ` (${Math.round(config.model_mb)} MB, only the first time)` : ""}…`);
  status("Starting the recogniser…");
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  if (config.lm_url) {
    try {
      const gz = await fetchWithProgress(config.lm_url,
        `Downloading the language model${config.lm_mb ? ` (${Math.round(config.lm_mb)} MB)` : ""}…`);
      // decompress unless the server already did (gzip starts with 1f 8b)
      lm = new CharLM(gz[0] === 0x1f && gz[1] === 0x8b ? await gunzip(gz)
        : gz.buffer.slice(gz.byteOffset, gz.byteOffset + gz.byteLength));
    } catch (err) {
      console.error(err);
    }
  }
  if (!lm) {
    $("use-lm").checked = false;
    $("use-lm").disabled = true;
  }
  loaded = true;
  status(`Ready${lm ? "" : " (without the language model)"}. Write a word, then press Read the word.`);
  $("recognise").disabled = false;
}

// ---------- writing ----------

const pad = $("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
let drawing = false;
let strokes = 0;

function resetPad() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  strokes = 0;
}

function pos(e) {
  const r = pad.getBoundingClientRect();
  return [((e.clientX - r.left) * pad.width) / r.width, ((e.clientY - r.top) * pad.height) / r.height];
}

pad.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = Number($("pen").value);
  const [x, y] = pos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.01, y);
  ctx.stroke();
});
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  e.preventDefault();
  const [x, y] = pos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});
const endStroke = () => {
  if (!drawing) return;
  drawing = false;
  strokes++;
  if ($("live").checked && loaded) recognise();
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("clear").addEventListener("click", () => { resetPad(); clearResult(); });

// ---------- upload ----------

$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploaded = await createImageBitmap(file);
  const prev = $("upload-preview");
  const scale = Math.min(1, 900 / Math.max(uploaded.width, uploaded.height));
  prev.width = Math.round(uploaded.width * scale);
  prev.height = Math.round(uploaded.height * scale);
  const pctx = prev.getContext("2d");
  pctx.fillStyle = "#fff";
  pctx.fillRect(0, 0, prev.width, prev.height);
  pctx.drawImage(uploaded, 0, 0, prev.width, prev.height);
  prev.hidden = false;
  if (loaded) recognise();
});

for (const tab of document.querySelectorAll("[data-mode]")) {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    for (const t of document.querySelectorAll("[data-mode]")) t.setAttribute("aria-selected", t === tab);
    $("draw-panel").hidden = mode !== "draw";
    $("upload-panel").hidden = mode !== "upload";
    clearResult();
  });
}

function grayOf(source, w, h) {
  // composite onto white paper, then greyscale as PIL does
  const c = new OffscreenCanvas(w, h);
  const cx = c.getContext("2d");
  cx.fillStyle = "#fff";
  cx.fillRect(0, 0, w, h);
  cx.drawImage(source, 0, 0, w, h);
  return grayFromRGBA(cx.getImageData(0, 0, w, h).data, w, h);
}

// ---------- reading ----------

const text = (ids) => ids.map((k) => config.alphabet[k - 1]).join("");
const spelled = (t) => ($("standard").checked ? standardSpelling(t) : t);

function clearResult() {
  last = null;
  $("word").textContent = "";
  $("alone").textContent = "";
  $("nbest").innerHTML = "";
  const seen = $("seen");
  seen.getContext("2d").clearRect(0, 0, seen.width, seen.height);
}

// Strokes can end faster than the network answers; run one reading at a time and, if
// more input arrived meanwhile, once more at the end.
let busy = false;
let pending = false;

async function recognise() {
  if (busy) { pending = true; return; }
  busy = true;
  try {
    await recogniseNow();
  } finally {
    busy = false;
  }
  if (pending) { pending = false; recognise(); }
}

async function recogniseNow() {
  let gray;
  if (mode === "draw") {
    if (!strokes) { status("Write a word first."); return; }
    gray = grayOf(pad, pad.width, pad.height);
  } else {
    if (!uploaded) { status("Choose an image first."); return; }
    const scale = Math.min(1, 2400 / Math.max(uploaded.width, uploaded.height));
    gray = grayOf(uploaded, Math.round(uploaded.width * scale), Math.round(uploaded.height * scale));
  }
  const t0 = performance.now();
  const norm = normalise(cropInk(gray), config.height, config.max_width);
  showSeen(norm);
  const input = modelInput(norm, config.stride);
  const out = await session.run({
    image: new ort.Tensor("float32", input.data, [1, 1, norm.h, input.width]),
    length: new ort.Tensor("int64", BigInt64Array.from([BigInt(input.frames)]), [1]),
  });
  const T = out.logp.dims[0], C = out.logp.dims[1];
  const logp = Float64Array.from(out.logp.data);
  const alone = greedy(logp, T, C);
  const useLM = lm && $("use-lm").checked;
  const res = beamSearch(logp, T, C, useLM ? lm : null,
    useLM ? config.decoding : { ...config.decoding, alpha: 0, beta: 0 });
  last = { best: useLM ? res.ids : alone, alone, nbest: res.nbest, useLM, norm };
  showResult();
  status(`Done in ${Math.round(performance.now() - t0)} ms, on your device.`);
}

function showSeen(norm) {
  const c = $("seen");
  c.width = norm.w;
  c.height = norm.h;
  const cx = c.getContext("2d");
  const im = cx.createImageData(norm.w, norm.h);
  for (let i = 0; i < norm.data.length; i++) {
    const v = 255 - norm.data[i];
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v;
    im.data[4 * i + 3] = 255;
  }
  cx.putImageData(im, 0, 0);
}

function showResult() {
  if (!last) return;
  const best = text(last.best);
  $("word").textContent = best ? spelled(best) : "(nothing read)";
  const alone = text(last.alone);
  $("alone").innerHTML = last.useLM
    ? (alone === best ? "The network alone reads the same."
      : `The network alone reads <span class="mayek">${spelled(alone) || "(nothing)"}</span>.`)
    : "Read by the network alone (greedy decoding).";
  // the alternatives kept by the beam search, their scores as shares among them
  const top = last.nbest.slice(0, 5);
  const m = top.length ? top[0].score : 0;
  const w = top.map((b) => Math.exp(b.score - m));
  const sum = w.reduce((a, b) => a + b, 0);
  $("nbest").innerHTML = "";
  top.forEach((b, i) => {
    const p = (100 * w[i]) / sum;
    const li = document.createElement("li");
    li.innerHTML = `<span class="w">${spelled(text(b.ids)) || "–"}</span>`
      + `<span class="bar"><span style="width:${p.toFixed(1)}%"></span></span>`
      + `<span class="p">${p.toFixed(1)}%</span>`;
    $("nbest").appendChild(li);
  });
}

$("recognise").addEventListener("click", () => loaded && recognise());
$("pen").addEventListener("input", () => { $("pen-value").textContent = $("pen").value; });
$("standard").addEventListener("change", showResult);
$("use-lm").addEventListener("change", () => { if (loaded && last) recognise(); });

resetPad();
load().catch((err) => { status(`Could not load the recogniser: ${err.message}`); console.error(err); });

// exposed for automated checks
window.mayekWords = {
  ready: () => loaded, hasLM: () => !!lm,
  last: () => last && { text: text(last.best), alone: text(last.alone), nbest: last.nbest.map((b) => text(b.ids)),
                        norm: { w: last.norm.w, h: last.norm.h, data: Array.from(last.norm.data) } },
};
