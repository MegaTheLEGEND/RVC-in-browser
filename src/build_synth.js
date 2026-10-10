// Builds the RVC synthesizer ONNX graph directly from a loaded .pth (see torchpkl.js). Batch=1, dynamic T.
// Inputs: phone[1,T,C] (pitch[1,T] i64, pitchf[1,T]) sid[1] i64, rnd[1,inter,T], (snd[1,T*upp,1]).  Output: audio[1,1,T*upp]
import { Graph, F } from "./onnxwriter.js";
import { Params, param, conv1d, convT1d, lrelu } from "./nn.js";

const SR = { "32k": 32000, "40k": 40000, "48k": 48000 };

export function buildSynth(ck) {
  const P = new Params(ck.weight), cfg = [...ck.config];
  let sr = cfg[17]; if (typeof sr === "string") sr = SR[sr] || Number(sr); if (ck.sr && typeof ck.sr === "number") sr = ck.sr; else if (typeof ck.sr === "string") sr = SR[ck.sr] || sr;
  const [, , inter, hidden, , nHeads, , , , resblock, rbKernels, rbDil, upRates, upInit, upKernels, , gin] = cfg;
  const f0 = ck.f0 === undefined ? 1 : Number(ck.f0), version = ck.version || "v1";
  const nSpk = P.get("emb_g.weight").shape[0], phoneDim = P.get("enc_p.emb_phone.weight").shape[1];
  const upp = upRates.reduce((a, b) => a * b, 1), dk = hidden / nHeads;
  const g = new Graph("rvc_synth");
  const phone = g.input("phone", "float32", [1, "T", phoneDim]);
  const pitch = f0 ? g.input("pitch", "int64", [1, "T"]) : null, pitchf = f0 ? g.input("pitchf", "float32", [1, "T"]) : null;
  const sid = g.input("sid", "int64", [1]), rnd = g.input("rnd", "float32", [1, inter, "T"]);
  const skip = g.input("skip", "int64", [1]); // frames dropped from the generator input (realtime context)
  const snd = f0 ? g.input("snd", "float32", [1, "S", 1]) : null;
  const cF = (v) => g.c(v);

  // ---- speaker embedding: g [1,gin,1]
  const gv = g.Unsqueeze(g.Gather(param(g, "emb_g", P.get("emb_g.weight").shape, P.get("emb_g.weight").data), sid, { axis: 0 }), g.i64([2]));

  // ---- text encoder
  const linear = (x, prefix) => { const w = P.get(`${prefix}.weight`), b = P.get(`${prefix}.bias`); // y = x W^T + b
    const wt = new Float32Array(w.data.length); const [o, i] = w.shape; for (let a = 0; a < o; a++) for (let c = 0; c < i; c++) wt[c * o + a] = w.data[a * i + c];
    return g.Add(g.MatMul(x, param(g, `${prefix}.Wt`, [i, o], wt)), param(g, `${prefix}.B`, b.shape, b.data)); };
  let x = linear(phone, "enc_p.emb_phone");
  if (f0) { const e = P.get("enc_p.emb_pitch.weight"); x = g.Add(x, g.Gather(param(g, "emb_pitch", e.shape, e.data), pitch, { axis: 0 })); }
  x = lrelu(g, g.Mul(x, cF(Math.sqrt(hidden))), 0.1);
  x = g.Transpose(x, { perm: [0, 2, 1] }); // [1,H,T]
  const nL = P.count("enc_p.encoder.attn_layers"), win = (P.get("enc_p.encoder.attn_layers.0.emb_rel_k").shape[1] - 1) / 2;
  // index helpers for relative attention (shared by all layers)
  const Tn = g.Gather(g.Shape(x), g.i64s(2), {}), Tv = g.Unsqueeze(Tn, g.i64([0]));
  const rng = g.Range(g.i64s(0), Tn, g.i64s(1));
  const ii = g.Unsqueeze(rng, g.i64([1])), jj = g.Unsqueeze(rng, g.i64([0]));
  const d = g.Add(g.Sub(jj, ii), g.i64s(win)); // j-i+w   [T,T]
  const expandTo = (t2d, last) => g.Expand(g.Unsqueeze(t2d, g.i64([0, 1])), g.Concat(g.i64([1, nHeads]), Tv, last, { axis: 0 }));
  const idxA = expandTo(g.Clip(d, g.i64s(0), g.i64s(2 * win)), Tv);
  const valA = g.Unsqueeze(g.Cast(g.Less(g.Abs(g.Sub(d, g.i64s(win))), g.i64s(win + 1)), { to: 1 }), g.i64([0, 1]));
  const mm = g.Unsqueeze(g.Range(g.i64s(0), g.i64s(2 * win + 1), g.i64s(1)), g.i64([0]));
  const e2 = g.Sub(g.Add(ii, mm), g.i64s(win)); // i+m-w  [T,2w+1]
  const idxB = expandTo(g.Clip(e2, g.i64s(0), g.Sub(Tn, g.i64s(1))), g.i64([2 * win + 1]));
  const valB = g.Unsqueeze(g.Cast(g.And(g.GreaterOrEqual(e2, g.i64s(0)), g.Less(e2, Tn)), { to: 1 }), g.i64([0, 1]));
  const heads = (t) => g.Reshape(t, g.i64([1, nHeads, dk, -1]));
  const ln = (t, prefix) => { const gm = P.get(`${prefix}.gamma`), bt = P.get(`${prefix}.beta`);
    const y = g.LayerNormalization(g.Transpose(t, { perm: [0, 2, 1] }), param(g, `${prefix}.g`, gm.shape, gm.data), param(g, `${prefix}.b`, bt.shape, bt.data), { axis: -1, epsilon: new F(1e-5) });
    return g.Transpose(y, { perm: [0, 2, 1] }); };
  for (let l = 0; l < nL; l++) {
    const a = `enc_p.encoder.attn_layers.${l}`;
    const q = g.Transpose(heads(conv1d(g, P, x, `${a}.conv_q`, { scale: 1 / Math.sqrt(dk) })), { perm: [0, 1, 3, 2] }); // [1,nh,T,dk] (pre-scaled)
    const k = heads(conv1d(g, P, x, `${a}.conv_k`)); // [1,nh,dk,T]
    const v = g.Transpose(heads(conv1d(g, P, x, `${a}.conv_v`)), { perm: [0, 1, 3, 2] });
    const ek = P.get(`${a}.emb_rel_k`), ev = P.get(`${a}.emb_rel_v`);
    const ekT = new Float32Array(ek.data.length); const M = 2 * win + 1; for (let m = 0; m < M; m++) for (let c = 0; c < dk; c++) ekT[c * M + m] = ek.data[m * dk + c];
    const relL = g.MatMul(q, param(g, `${a}.ekT`, [1, 1, dk, M], ekT)); // [1,nh,T,2w+1]
    let scores = g.Add(g.MatMul(q, k), g.Mul(g.GatherElements(relL, idxA, { axis: 3 }), valA));
    const p = g.Softmax(scores, { axis: -1 });
    const relW = g.Mul(g.GatherElements(p, idxB, { axis: 3 }), valB); // [1,nh,T,2w+1]
    let o = g.Add(g.MatMul(p, v), g.MatMul(relW, param(g, `${a}.ev`, [1, 1, M, dk], ev.data)));
    o = g.Reshape(g.Transpose(o, { perm: [0, 1, 3, 2] }), g.i64([1, hidden, -1]));
    o = conv1d(g, P, o, `${a}.conv_o`);
    x = ln(g.Add(x, o), `enc_p.encoder.norm_layers_1.${l}`);
    const f = `enc_p.encoder.ffn_layers.${l}`, ks = P.weight(`${f}.conv_1`).shape[2];
    let y = g.Relu(conv1d(g, P, x, `${f}.conv_1`, { pad: (ks - 1) >> 1, padR: ks >> 1 }));
    y = conv1d(g, P, y, `${f}.conv_2`, { pad: (ks - 1) >> 1, padR: ks >> 1 });
    x = ln(g.Add(x, y), `enc_p.encoder.norm_layers_2.${l}`);
  }
  const stats = conv1d(g, P, x, "enc_p.proj");
  const [m, logs] = g.nodeN("Split", [stats, g.i64([inter, inter])], { axis: 1 }, 2);
  let z = g.Add(m, g.Mul(g.Mul(g.Exp(logs), rnd), cF(0.66666)));

  // ---- flow (reverse): F3,C3,F2,C2,...
  let nCoup = 0; while (P.has(`flow.flows.${2 * nCoup}.pre.bias`)) nCoup++;
  const flipIdx = g.i64(Array.from({ length: inter }, (_, i) => inter - 1 - i));
  const half = inter / 2;
  const wn = (h, prefix, hid, gcond) => {
    const nl = P.count(`${prefix}.in_layers`);
    const cond = conv1d(g, P, gcond, `${prefix}.cond_layer`); let out = null;
    for (let i = 0; i < nl; i++) {
      const ks = P.weight(`${prefix}.in_layers.${i}`).shape[2], dil = 1; // dilation_rate is 1 in RVC (1**i)
      let xi = conv1d(g, P, h, `${prefix}.in_layers.${i}`, { dilation: dil, pad: ((ks * dil - dil) / 2) | 0 });
      const gl = g.Slice(cond, g.i64([i * 2 * hid]), g.i64([(i + 1) * 2 * hid]), g.i64([1]));
      xi = g.Add(xi, gl);
      const [a, b] = g.nodeN("Split", [xi, g.i64([hid, hid])], { axis: 1 }, 2);
      const rs = conv1d(g, P, g.Mul(g.Tanh(a), g.Sigmoid(b)), `${prefix}.res_skip_layers.${i}`);
      if (i < nl - 1) { const [r, s] = g.nodeN("Split", [rs, g.i64([hid, hid])], { axis: 1 }, 2); h = g.Add(h, r); out = out ? g.Add(out, s) : s; }
      else out = out ? g.Add(out, rs) : rs;
    }
    return out;
  };
  for (let c = nCoup - 1; c >= 0; c--) {
    z = g.Gather(z, flipIdx, { axis: 1 });
    const pre = `flow.flows.${2 * c}`;
    const [x0, x1] = g.nodeN("Split", [z, g.i64([half, half])], { axis: 1 }, 2);
    const h = conv1d(g, P, x0, `${pre}.pre`), hid = P.weight(`${pre}.pre`).shape[0];
    const mm2 = conv1d(g, P, wn(h, `${pre}.enc`, hid, gv), `${pre}.post`);
    z = g.Concat(x0, g.Sub(x1, mm2), { axis: 1 });
  }
  // (flip after coupling 0 in reverse order is the leading Flip of the list: flows = C0,F0,C1,F1,... reversed => F_last first; handled above)

  // ---- generator
  const BIG = g.i64([2147483647]), ax2 = g.i64([2]), ax1 = g.i64([1]);
  z = g.Slice(z, skip, BIG, ax2);
  let h = conv1d(g, P, z, "dec.conv_pre", { pad: 3 });
  h = g.Add(h, conv1d(g, P, gv, "dec.cond"));
  let har = null;
  if (f0) {
    const f = g.Unsqueeze(g.Slice(pitchf, skip, BIG, ax1), g.i64([2])); // [1,T,1]
    const ar = new Float32Array(upp); for (let i = 0; i < upp; i++) ar[i] = i + 1;
    const rad = g.Mul(g.Div(f, cF(sr)), param(g, "sg_ar", [1, 1, upp], ar));
    const last = g.Slice(rad, g.i64([upp - 1]), g.i64([upp]), g.i64([2]));
    const rad2 = g.Sub(g.Mod(g.Add(last, cF(0.5)), cF(1), { fmod: 1 }), cF(0.5));
    const acc = g.Mod(g.CumSum(rad2, g.i64s(1)), cF(1), { fmod: 1 });
    const accS = g.Concat(g.ConstantOfShape(g.i64([1, 1, 1]), { value: { __tensor: { dtype: "float32", dims: [1], data: [0] } } }), g.Slice(acc, g.i64([0]), g.i64([-1]), g.i64([1])), { axis: 1 });
    const radT = g.Reshape(g.Add(rad, accS), g.i64([1, -1, 1]));
    const sine = g.Mul(g.Sin(g.Mul(radT, cF(2 * Math.PI))), cF(0.1));
    const uv = g.Cast(g.Greater(f, cF(0)), { to: 1 });
    const uvU = g.Reshape(g.Mul(uv, param(g, "sg_ones", [1, 1, upp], new Float32Array(upp).fill(1))), g.i64([1, -1, 1]));
    const namp = g.Add(g.Mul(uvU, cF(0.003)), g.Mul(g.Sub(cF(1), uvU), cF(0.1 / 3)));
    const src = g.Add(g.Mul(sine, uvU), g.Mul(namp, g.Slice(snd, g.Mul(skip, g.i64([upp])), BIG, ax1)));
    const lw = P.get("dec.m_source.l_linear.weight"), lb = P.get("dec.m_source.l_linear.bias");
    har = g.Transpose(g.Tanh(g.Add(g.Mul(src, param(g, "lin_w", [1], lw.data)), param(g, "lin_b", [1], lb.data))), { perm: [0, 2, 1] }); // [1,1,S]
  }
  const nUp = upRates.length, nk = rbKernels.length;
  for (let i = 0; i < nUp; i++) {
    h = lrelu(g, h, 0.1);
    h = convT1d(g, P, h, `dec.ups.${i}`, { stride: upRates[i], pad: (upKernels[i] - upRates[i]) >> 1 });
    if (f0) {
      const s = i + 1 < nUp ? upRates.slice(i + 1).reduce((a, b) => a * b, 1) : 0;
      h = g.Add(h, s ? conv1d(g, P, har, `dec.noise_convs.${i}`, { stride: s, pad: s >> 1 }) : conv1d(g, P, har, `dec.noise_convs.${i}`));
    }
    let xs = null;
    for (let j = 0; j < nk; j++) {
      const r = `dec.resblocks.${i * nk + j}`, k = rbKernels[j], dl = rbDil[j];
      let y = h;
      if (String(resblock) === "1") for (let t = 0; t < 3; t++) {
        let xt = conv1d(g, P, lrelu(g, y, 0.1), `${r}.convs1.${t}`, { dilation: dl[t], pad: ((k * dl[t] - dl[t]) / 2) | 0 });
        xt = conv1d(g, P, lrelu(g, xt, 0.1), `${r}.convs2.${t}`, { pad: (k - 1) >> 1 });
        y = g.Add(xt, y);
      } else for (let t = 0; t < 2; t++) y = g.Add(conv1d(g, P, lrelu(g, y, 0.1), `${r}.convs.${t}`, { dilation: dl[t], pad: ((k * dl[t] - dl[t]) / 2) | 0 }), y);
      xs = xs ? g.Add(xs, y) : y;
    }
    h = g.Div(xs, cF(nk));
  }
  h = g.Tanh(conv1d(g, P, lrelu(g, h, 0.01), "dec.conv_post", { pad: 3 }));
  g.output(g.Identity(h), "float32", [1, 1, "S"]);
  // rename output to "audio"
  const bytes = g.toBytes();
  return { onnx: bytes, meta: { version, f0, sr, upp, inter_channels: inter, phone_dim: phoneDim, n_spk: nSpk, speakers: ck.speaker_info || [], inputs: ["phone", ...(f0 ? ["pitch", "pitchf"] : []), "sid", "rnd", ...(f0 ? ["snd"] : []), "skip"] } };
}
