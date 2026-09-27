// Reading a word from the network's per-column log probabilities: greedy CTC decoding
// and CTC prefix beam search with the character language model, ported from
// mayek_htr/decode.py and mayek_htr/lm.py. Free of browser APIs, so tests/test_web.py
// checks it against the Python version with Node.
//
// Classes: 0 is the CTC blank, 1..54 the characters (config.alphabet[k - 1]). The
// language model numbers its symbols the same way, with 0 for the start of a word and
// 55 for its end.

const NEG = -Infinity;

function lse(a, b) {
  if (a === NEG) return b;
  if (b === NEG) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

export function greedy(logp, T, C) {
  // the best class of every column, repeats merged, blanks dropped -> class ids
  const ids = [];
  let prev = 0;
  for (let t = 0; t < T; t++) {
    let best = 0;
    for (let c = 1; c < C; c++) if (logp[t * C + c] > logp[t * C + best]) best = c;
    if (best !== prev && best !== 0) ids.push(best);
    prev = best;
  }
  return ids;
}

export class CharLM {
  // lm.CharLM as written by mayek_htr.web.export_lm: every n-gram seen in training with
  // its log probability, every context seen with its log back-off weight (interpolated
  // Kneser-Ney written in back-off form, so the values are exactly the Python model's)
  constructor(buffer) {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== "MMLM") throw new Error("not a language model file");
    const len = dv.getUint32(8, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 16, len)));
    this.order = head.order;
    this.size = head.symbols;
    this.eos = head.eos;
    this.floor = head.unigram_floor;
    this.tables = {};
    for (const t of head.tables) {
      this.tables[`${t.kind}${t.n}`] = {
        keys: new Float64Array(buffer, t.keys_at, t.count),
        vals: new Float32Array(buffer, t.vals_at, t.count),
      };
    }
    this.cache = new Map();
  }

  static find(table, key) {
    if (!table) return undefined;
    const { keys, vals } = table;
    let lo = 0, hi = keys.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const k = keys[mid];
      if (k === key) return vals[mid];
      if (k < key) lo = mid + 1; else hi = mid - 1;
    }
    return undefined;
  }

  p(w, h, n) {
    // natural log P_n(w | the last n - 1 symbols of h)
    if (n === 1) {
      const v = CharLM.find(this.tables.ngram1, w);
      return v === undefined ? this.floor : v;
    }
    let ctx = 0;
    for (let i = h.length - (n - 1); i < h.length; i++) ctx = ctx * this.size + h[i];
    const v = CharLM.find(this.tables[`ngram${n}`], ctx * this.size + w);
    if (v !== undefined) return v;
    const back = CharLM.find(this.tables[`context${n}`], ctx);
    const lower = this.p(w, h, n - 1);
    return back === undefined ? lower : back + lower;
  }

  logprob(w, history) {
    // natural log P(w | the characters before), w a class id or this.eos
    const N = this.order;
    const h = new Array(N - 1).fill(0);
    const k = Math.min(history.length, N - 1);
    for (let i = 0; i < k; i++) h[N - 1 - k + i] = history[history.length - k + i];
    const key = `${w}|${h.join(",")}`;
    let v = this.cache.get(key);
    if (v === undefined) {
      v = this.p(w, h, N);
      if (this.cache.size > 200000) this.cache.clear();
      this.cache.set(key, v);
    }
    return v;
  }
}

export function beamSearch(logp, T, C, lm = null, { alpha = 0.5, beta = 0.0, beam = 16, prune = -10.0, topk = 8 } = {}) {
  // decode.beam_search -> {ids: the best prefix, nbest: [{ids, score}] best first}
  const useLM = lm !== null && alpha !== 0;
  let beams = new Map([["", { p: [], pb: 0, pnb: NEG }]]);
  const lmCache = new Map();
  const lmScore = (c, prefix, key) => {
    if (!useLM) return beta;
    const k = `${key}|${c}`;
    let v = lmCache.get(k);
    if (v === undefined) {
      v = alpha * lm.logprob(c, prefix) + beta;
      lmCache.set(k, v);
    }
    return v;
  };

  for (let t = 0; t < T; t++) {
    const row = logp.subarray(t * C, (t + 1) * C);
    let cand = [];
    for (let c = 1; c < C; c++) if (row[c] > prune) cand.push(c);
    if (cand.length > topk) cand = cand.sort((a, b) => row[b] - row[a]).slice(0, topk);
    const next = new Map();
    const add = (key, prefix, pb, pnb) => {
      const o = next.get(key);
      if (o) { o.pb = lse(o.pb, pb); o.pnb = lse(o.pnb, pnb); } else next.set(key, { p: prefix, pb, pnb });
    };
    for (const [key, { p: prefix, pb, pnb }] of beams) {
      const total = lse(pb, pnb);
      add(key, prefix, total + row[0], NEG);                            // a blank: the prefix stays
      if (prefix.length) add(key, prefix, NEG, pnb + row[prefix[prefix.length - 1]]);   // the last again
      for (const c of cand) {
        const via = prefix.length && c === prefix[prefix.length - 1] ? pb : total;   // a repeat needs a blank
        add(key ? `${key},${c}` : `${c}`, prefix.concat([c]), NEG, via + row[c] + lmScore(c, prefix, key));
      }
    }
    beams = new Map([...next.entries()]
      .sort((a, b) => lse(b[1].pb, b[1].pnb) - lse(a[1].pb, a[1].pnb)).slice(0, beam));
  }

  const nbest = [];
  for (const { p, pb, pnb } of beams.values()) {
    let s = lse(pb, pnb);
    if (useLM) s += alpha * lm.logprob(lm.eos, p);
    nbest.push({ ids: p, score: s });
  }
  let best = nbest[0];
  for (const b of nbest) if (b.score > best.score) best = b;
  return { ids: best.ids, nbest: nbest.slice().sort((a, b) => b.score - a.score) };
}
