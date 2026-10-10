// Shared helpers for the JS ONNX graph builders: state-dict access (with weight-norm fusing) and common layer emitters.
import { TorchTensor } from "./torchpkl.js";
import { F } from "./onnxwriter.js";

/** w = g * v / ||v||, where the norm is taken over every dim where g has size 1 (torch weight_norm, any `dim`). */
export function fuseWeightNorm(gT, vT) {
  const g = gT.f32(), v = vT.f32(), vs = vT.shape, gs = [...gT.shape];
  while (gs.length < vs.length) gs.push(1);
  const nd = vs.length, out = new Float32Array(v.length);
  // group index of every element = linear index over dims where g.shape != 1
  const keep = []; for (let d = 0; d < nd; d++) if (gs[d] !== 1) keep.push(d);
  const vstr = new Array(nd); { let s = 1; for (let d = nd - 1; d >= 0; d--) { vstr[d] = s; s *= vs[d]; } }
  const nGroups = keep.reduce((a, d) => a * vs[d], 1), norm = new Float64Array(nGroups);
  const gidx = new Int32Array(v.length);
  const idx = new Array(nd).fill(0);
  for (let i = 0; i < v.length; i++) {
    let gi = 0; for (const d of keep) gi = gi * vs[d] + idx[d];
    gidx[i] = gi; norm[gi] += v[i] * v[i];
    for (let d = nd - 1; d >= 0; d--) { if (++idx[d] < vs[d]) break; idx[d] = 0; }
  }
  for (let i = 0; i < v.length; i++) { const gi = gidx[i]; out[i] = (g[gi] * v[i]) / Math.sqrt(norm[gi]); }
  return { data: out, shape: [...vs] };
}

/** State-dict wrapper. Names the way the python modules store them. */
export class Params {
  constructor(sd) { this.sd = sd; this._c = new Map(); }
  has(k) { return k in this.sd; }
  keys() { return Object.keys(this.sd); }
  raw(k) { const t = this.sd[k]; if (!(t instanceof TorchTensor)) throw new Error(`missing weight "${k}"`); return t; }
  /** plain tensor -> {data, shape} (fp32) */
  get(k) { if (!this._c.has(k)) { const t = this.raw(k); this._c.set(k, { data: t.f32(), shape: [...t.shape] }); } return this._c.get(k); }
  /** `prefix.weight` possibly stored with weight_norm (weight_g/weight_v or parametrizations) -> {data, shape} */
  weight(prefix) {
    const key = `w:${prefix}`; if (this._c.has(key)) return this._c.get(key);
    let r;
    if (this.has(`${prefix}.weight`)) r = this.get(`${prefix}.weight`);
    else if (this.has(`${prefix}.weight_g`)) r = fuseWeightNorm(this.raw(`${prefix}.weight_g`), this.raw(`${prefix}.weight_v`));
    else if (this.has(`${prefix}.parametrizations.weight.original0`)) r = fuseWeightNorm(this.raw(`${prefix}.parametrizations.weight.original0`), this.raw(`${prefix}.parametrizations.weight.original1`));
    else throw new Error(`missing weight for "${prefix}"`);
    this._c.set(key, r); return r;
  }
  bias(prefix) { return this.has(`${prefix}.bias`) ? this.get(`${prefix}.bias`) : null; }
  /** number of consecutive indices `prefix.N.` present in the state dict */
  count(prefix) {
    const seen = new Set(), pre = `${prefix}.`;
    for (const k of Object.keys(this.sd)) if (k.startsWith(pre)) seen.add(k.slice(pre.length).split(".")[0]);
    let n = 0; while (seen.has(String(n))) n++; return n;
  }
}

/** Per-graph parameter registry so every initializer is emitted once. */
export function param(g, name, shape, data) {
  g._params ||= new Map();
  if (!g._params.has(name)) g._params.set(name, g.f32(name, shape, data));
  return g._params.get(name);
}

/** Conv1d with weights (possibly weight-normed) from `P` at `prefix`. x: [N, Cin, L]. */
export function conv1d(g, P, x, prefix, { dilation = 1, stride = 1, pad = 0, padR = null, groups = 1, scale = 1 } = {}) {
  const w = P.weight(prefix), b = P.bias(prefix);
  let wd = w.data, bd = b && b.data;
  if (scale !== 1) { wd = wd.map((v) => v * scale); if (bd) bd = bd.map((v) => v * scale); }
  const wn = param(g, `${prefix}.W${scale !== 1 ? "s" : ""}`, w.shape, wd);
  const ins = [x, wn]; if (b) ins.push(param(g, `${prefix}.B${scale !== 1 ? "s" : ""}`, b.shape, bd));
  return g.Conv(...ins, { kernel_shape: [w.shape[2]], dilations: [dilation], strides: [stride], pads: [pad, padR ?? pad], group: groups });
}

export function convT1d(g, P, x, prefix, { stride, pad }) {
  const w = P.weight(prefix), b = P.bias(prefix);
  const ins = [x, param(g, `${prefix}.W`, w.shape, w.data)]; if (b) ins.push(param(g, `${prefix}.B`, b.shape, b.data));
  return g.ConvTranspose(...ins, { kernel_shape: [w.shape[2]], strides: [stride], pads: [pad, pad], dilations: [1], group: 1 });
}

export const lrelu = (g, x, a) => g.LeakyRelu(x, { alpha: new F(a) });
