// Builds the RMVPE (E2E DeepUnet + BiGRU) ONNX directly from rmvpe.pt's state dict. Batch=1.
// Input mel[1,128,T] f32 (T multiple of 32, like src/rmvpe.js feeds); output [1,T,360] (sigmoid salience). Sizes are read from the weights.
import { Graph, F } from "./onnxwriter.js";
import { Params, param } from "./nn.js";

export function buildRmvpe(sd0) {
  const sd = sd0.model && typeof sd0.model === "object" && !sd0.model.shape ? sd0.model : sd0;
  const P = new Params(sd);
  if (!P.has("cnn.weight") || !P.has("unet.encoder.bn.weight")) throw new Error("not an RMVPE state dict (missing cnn.weight / unet.encoder.bn.weight)");
  if (P.has("fc.0.gru.weight_ih_l1")) throw new Error("RMVPE with >1 GRU layer is not supported");
  const g = new Graph("rmvpe"), cF = (v) => g.c(v);
  const mel = g.input("mel", "float32", [1, 128, "T"]);
  const conv2d = (x, p, { pad = 1 } = {}) => { const w = P.get(`${p}.weight`), b = P.bias(p), ins = [x, param(g, `${p}.W`, w.shape, w.data)]; if (b) ins.push(param(g, `${p}.B`, b.shape, b.data));
    return g.Conv(...ins, { kernel_shape: [w.shape[2], w.shape[3]], pads: [pad, pad, pad, pad], strides: [1, 1], dilations: [1, 1], group: 1 }); };
  const bn = (x, p) => { const f = (n) => { const t = P.get(`${p}.${n}`); return param(g, `${p}.${n}`, t.shape, t.data); };
    return g.BatchNormalization(x, f("weight"), f("bias"), f("running_mean"), f("running_var"), { epsilon: new F(1e-5) }); };
  const relu = (x) => g.Relu(x);
  const convBlockRes = (x, p) => { // conv.0 / conv.1(bn) / relu / conv.3 / conv.4(bn) / relu, + shortcut (1x1 conv if channels change, else identity)
    let y = relu(bn(conv2d(x, `${p}.conv.0`), `${p}.conv.1`)); y = relu(bn(conv2d(y, `${p}.conv.3`), `${p}.conv.4`));
    return g.Add(y, P.has(`${p}.shortcut.weight`) ? conv2d(x, `${p}.shortcut`, { pad: 0 }) : x); };
  const resBlock = (x, p) => { const n = P.count(`${p}.conv`); for (let i = 0; i < n; i++) x = convBlockRes(x, `${p}.conv.${i}`); return x; };

  let x = g.Unsqueeze(g.Transpose(mel, { perm: [0, 2, 1] }), g.i64([1])); // [1,1,T,128]
  x = bn(x, "unet.encoder.bn");
  const skips = [], nEnc = P.count("unet.encoder.layers");
  for (let i = 0; i < nEnc; i++) { x = resBlock(x, `unet.encoder.layers.${i}`); skips.push(x); x = g.AveragePool(x, { kernel_shape: [2, 2], strides: [2, 2] }); }
  for (let i = 0; i < P.count("unet.intermediate.layers"); i++) x = resBlock(x, `unet.intermediate.layers.${i}`);
  for (let i = 0; i < P.count("unet.decoder.layers"); i++) {
    const p = `unet.decoder.layers.${i}`, w = P.get(`${p}.conv1.0.weight`);
    x = relu(bn(g.ConvTranspose(x, param(g, `${p}.conv1.0.W`, w.shape, w.data), { kernel_shape: [3, 3], strides: [2, 2], pads: [1, 1, 1, 1], output_padding: [1, 1], dilations: [1, 1], group: 1 }), `${p}.conv1.1`));
    x = g.Concat(x, skips[nEnc - 1 - i], { axis: 1 });
    const n = P.count(`${p}.conv2`); for (let j = 0; j < n; j++) x = convBlockRes(x, `${p}.conv2.${j}`);
  }
  x = conv2d(x, "cnn"); // [1,3,T,128]
  x = g.Reshape(g.Transpose(x, { perm: [0, 2, 1, 3] }), g.i64([1, -1, 384])); // transpose(1,2).flatten(-2) -> [1,T,384]

  // BiGRU (torch gate order r,z,n -> ONNX z,r,h; linear_before_reset=1 matches torch)
  const G = "fc.0.gru", hid = P.get(`${G}.weight_hh_l0`).shape[1], inp = P.get(`${G}.weight_ih_l0`).shape[1];
  const reorder = (d, rows, cols) => { const o = new Float32Array(d.length), blk = hid * cols, src = [1, 0, 2]; for (let k = 0; k < 3; k++) o.set(d.subarray(src[k] * blk, (src[k] + 1) * blk), k * blk); return o; };
  const both = (name, cols) => { const o = new Float32Array(2 * 3 * hid * cols); [`${G}.${name}_l0`, `${G}.${name}_l0_reverse`].forEach((k, d) => o.set(reorder(P.get(k).data, 3 * hid, cols), d * 3 * hid * cols)); return o; };
  const W = param(g, "gru.W", [2, 3 * hid, inp], both("weight_ih", inp)), R = param(g, "gru.R", [2, 3 * hid, hid], both("weight_hh", hid));
  const Bv = new Float32Array(2 * 6 * hid); [0, 1].forEach((d) => { const sfx = d ? "_reverse" : ""; Bv.set(reorder(P.get(`${G}.bias_ih_l0${sfx}`).data, 3 * hid, 1), d * 6 * hid); Bv.set(reorder(P.get(`${G}.bias_hh_l0${sfx}`).data, 3 * hid, 1), d * 6 * hid + 3 * hid); });
  const B = param(g, "gru.B", [2, 6 * hid], Bv);
  const seq = g.Transpose(x, { perm: [1, 0, 2] }); // [T,1,384]
  const Y = g.nodeN("GRU", [seq, W, R, B], { hidden_size: hid, direction: "bidirectional", linear_before_reset: 1 }, 2)[0]; // [T,2,1,hid]
  x = g.Reshape(g.Transpose(Y, { perm: [0, 2, 1, 3] }), g.i64([1, -1, 2 * hid])); // [1,T,512]
  const lw = P.get("fc.1.weight"), lb = P.get("fc.1.bias"), [o, i] = lw.shape, wt = new Float32Array(o * i);
  for (let a = 0; a < o; a++) for (let c = 0; c < i; c++) wt[c * o + a] = lw.data[a * i + c];
  x = g.Sigmoid(g.Add(g.MatMul(x, param(g, "fc.1.Wt", [i, o], wt)), param(g, "fc.1.B", lb.shape, lb.data)));
  g.output(g.Identity(x), "float32", [1, "T", 360]);
  return { onnx: g.toBytes() };
}
