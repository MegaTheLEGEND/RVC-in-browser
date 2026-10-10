// Builds the HuBERT/ContentVec ONNX (HF "HubertModel[WithFinalProj]" layout) directly from a state dict. Batch=1, dynamic N.
// Input audio[1,N] f32 (16 kHz, NOT normalised). Output feats[1,T,C]: v2 = layer-12 output (768), v1 = layer-9 output -> final_proj (256).
import { Graph, F } from "./onnxwriter.js";
import { Params, param, conv1d } from "./nn.js";

export const BASE_CFG = { convDim: 512, kernels: [10, 3, 3, 3, 3, 2, 2], strides: [5, 2, 2, 2, 2, 2, 2], layers: 12, hidden: 768, heads: 12, ffn: 3072, posK: 128, posGroups: 16, groupNormFirst: true };

/** sd: {name: TorchTensor-like}. Accepts any prefix ("hubert.", "m.", ...). cfg: HF config.json fields (optional). */
export function buildHubert(sd, version = "v2", cfg0 = {}) {
  const suf = "feature_projection.projection.bias", k0 = Object.keys(sd).find((k) => k.endsWith(suf));
  if (!k0) throw new Error("not a HuBERT state dict (no feature_projection.projection.bias)");
  const pre = k0.slice(0, -suf.length), sd2 = {}; for (const k of Object.keys(sd)) if (k.startsWith(pre)) sd2[k.slice(pre.length)] = sd[k];
  for (const k of Object.keys(sd)) if (!k.startsWith(pre) && !(k in sd2)) sd2[k] = sd[k]; // e.g. un-prefixed final_proj next to a "hubert." encoder
  const P = new Params(sd2);
  const c = { ...BASE_CFG, ...cfg0 };
  if (cfg0.conv_kernel) c.kernels = cfg0.conv_kernel; if (cfg0.conv_stride) c.strides = cfg0.conv_stride; if (cfg0.conv_dim) c.convDim = cfg0.conv_dim[0];
  for (const [a, b] of [["num_hidden_layers", "layers"], ["hidden_size", "hidden"], ["num_attention_heads", "heads"], ["intermediate_size", "ffn"], ["num_conv_pos_embeddings", "posK"], ["num_conv_pos_embedding_groups", "posGroups"]]) if (cfg0[a] !== undefined) c[b] = cfg0[a];
  const nConv = c.kernels.length, H = c.hidden, dk = H / c.heads, nLayers = version === "v1" ? Math.min(9, c.layers) : c.layers;
  const g = new Graph("hubert"), cF = (v) => g.c(v);
  const audio = g.input("audio", "float32", [1, "N"]);
  const gelu = (x) => g.Mul(g.Mul(x, cF(0.5)), g.Add(cF(1), g.Erf(g.Mul(x, cF(Math.SQRT1_2)))));
  const ln = (x, p) => { const w = P.get(`${p}.weight`), b = P.get(`${p}.bias`); return g.LayerNormalization(x, param(g, `${p}.g`, w.shape, w.data), param(g, `${p}.b`, b.shape, b.data), { axis: -1, epsilon: new F(1e-5) }); };
  const linear = (x, p) => { const w = P.get(`${p}.weight`), b = P.get(`${p}.bias`), [o, i] = w.shape, wt = new Float32Array(o * i);
    for (let a = 0; a < o; a++) for (let q = 0; q < i; q++) wt[q * o + a] = w.data[a * i + q];
    return g.Add(g.MatMul(x, param(g, `${p}.Wt`, [i, o], wt)), param(g, `${p}.B`, b.shape, b.data)); };

  let x = g.Unsqueeze(audio, g.i64([1])); // [1,1,N]
  for (let i = 0; i < nConv; i++) {
    x = conv1d(g, P, x, `feature_extractor.conv_layers.${i}.conv`, { stride: c.strides[i] });
    if (i === 0 && P.has("feature_extractor.conv_layers.0.layer_norm.weight")) { // GroupNorm(C, C) == per-channel InstanceNorm
      const w = P.get("feature_extractor.conv_layers.0.layer_norm.weight"), b = P.get("feature_extractor.conv_layers.0.layer_norm.bias");
      x = g.InstanceNormalization(x, param(g, "gn0.g", w.shape, w.data), param(g, "gn0.b", b.shape, b.data), { epsilon: new F(1e-5) });
    } else if (P.has(`feature_extractor.conv_layers.${i}.layer_norm.weight`)) { // layer_norm feat-extract variant
      x = g.Transpose(ln(g.Transpose(x, { perm: [0, 2, 1] }), `feature_extractor.conv_layers.${i}.layer_norm`), { perm: [0, 2, 1] });
    }
    x = gelu(x);
  }
  x = g.Transpose(x, { perm: [0, 2, 1] }); // [1,T,C]
  if (P.has("feature_projection.layer_norm.weight")) x = ln(x, "feature_projection.layer_norm");
  x = linear(x, "feature_projection.projection");
  // positional conv embedding (weight-normed, groups), drop the last frame, GELU, residual
  let pc = conv1d(g, P, g.Transpose(x, { perm: [0, 2, 1] }), "encoder.pos_conv_embed.conv", { pad: c.posK >> 1, groups: c.posGroups });
  if (c.posK % 2 === 0) pc = g.Slice(pc, g.i64([0]), g.i64([-1]), g.i64([2]));
  x = g.Add(x, g.Transpose(gelu(pc), { perm: [0, 2, 1] }));
  x = ln(x, "encoder.layer_norm");
  const split = (t) => g.Transpose(g.Reshape(t, g.i64([1, -1, c.heads, dk])), { perm: [0, 2, 1, 3] });
  for (let l = 0; l < nLayers; l++) {
    const a = `encoder.layers.${l}.attention`;
    const q = split(g.Mul(linear(x, `${a}.q_proj`), cF(1 / Math.sqrt(dk)))), k = g.Transpose(split(linear(x, `${a}.k_proj`)), { perm: [0, 1, 3, 2] }), v = split(linear(x, `${a}.v_proj`));
    const att = g.MatMul(g.Softmax(g.MatMul(q, k), { axis: -1 }), v);
    const o = linear(g.Reshape(g.Transpose(att, { perm: [0, 2, 1, 3] }), g.i64([1, -1, H])), `${a}.out_proj`);
    x = ln(g.Add(x, o), `encoder.layers.${l}.layer_norm`);
    const f = linear(gelu(linear(x, `encoder.layers.${l}.feed_forward.intermediate_dense`)), `encoder.layers.${l}.feed_forward.output_dense`);
    x = ln(g.Add(x, f), `encoder.layers.${l}.final_layer_norm`);
  }
  if (version === "v1") x = linear(x, "final_proj");
  g.output(g.Identity(x), "float32", [1, "T", version === "v1" ? P.get("final_proj.weight").shape[0] : H]);
  return { onnx: g.toBytes(), meta: { version, hidden: H, layers: nLayers } };
}
