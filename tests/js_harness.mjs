// Runs the page's JavaScript on inputs written by tests/test_web.py, for comparison with
// the Python code: node tests/js_harness.mjs job.json out.json
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { cropInk, normalise, standardSpelling } from "../web/pipeline.js";
import { beamSearch, CharLM, greedy } from "../web/decode.js";

const job = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = {};

if (job.images) {
  out.images = job.images.map(({ w, h, data, crop }) => {
    let img = { w, h, data: Uint8Array.from(data) };
    if (crop) img = cropInk(img);
    const n = normalise(img);
    return { w: n.w, h: n.h, data: Array.from(n.data), crop: [img.w, img.h] };
  });
}

if (job.decode) {
  const buf = gunzipSync(readFileSync(job.decode.lm));
  const lm = new CharLM(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  out.logprobs = job.decode.queries.map(([w, h]) => lm.logprob(w, h));
  out.decode = job.decode.matrices.map(({ T, C, data }) => {
    const logp = Float64Array.from(data);
    return {
      greedy: greedy(logp, T, C),
      plain: beamSearch(logp, T, C, null, { alpha: 0, beta: 0 }).ids,
      lm: beamSearch(logp, T, C, lm, job.decode.settings).ids,
    };
  });
}

if (job.spelling) out.spelling = job.spelling.map(standardSpelling);

writeFileSync(process.argv[3], JSON.stringify(out));
