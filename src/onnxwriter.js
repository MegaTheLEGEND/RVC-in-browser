// Tiny ONNX writer: protobuf encoder + a graph-builder DSL. No dependencies, runs in browsers and Node.
//
//   const g = new Graph("synth");
//   const x = g.input("x", "float32", [1, "T", 192]);
//   const w = g.f32("w", [192, 192], weights);
//   const y = g.Relu(g.MatMul(x, w));              // any ONNX op: g.<OpType>(input, input, ..., {attrs})
//   g.output(y, "float32", [1, "T", 192]);
//   const bytes = g.toBytes();                      // Uint8Array ModelProto (opset 17, ir 8)
//
// Positional args are node inputs (strings = tensor names, null = omitted optional input). A trailing plain object is the
// attribute map. Numbers that are integers become INT attributes, others FLOAT; wrap with F(x) to force float
// (e.g. alpha: F(1)). Arrays of integers become INTS. Use g.nodeN(op, inputs, attrs, nOut) for multi-output ops.

export const DT = { float32: 1, uint8: 2, int8: 3, int32: 6, int64: 7, bool: 9, float16: 10, float64: 11 };
export class F { constructor(v) { this.v = v; } }

// ----------------------------------------------------------------------------- protobuf
class Msg {
  constructor() { this.parts = []; this.len = 0; }
  _push(u8) { this.parts.push(u8); this.len += u8.length; }
  varint(v) { // v: Number | BigInt (non-negative or negative int64)
    let b = BigInt.asUintN(64, BigInt(v)); const out = [];
    while (b > 0x7fn) { out.push(Number(b & 0x7fn) | 0x80); b >>= 7n; }
    out.push(Number(b)); this._push(Uint8Array.from(out)); return this;
  }
  tag(field, wire) { return this.varint((field << 3) | wire); }
  int(field, v) { return this.tag(field, 0).varint(v); }
  f32(field, v) { const u = new Uint8Array(4); new DataView(u.buffer).setFloat32(0, v, true); this.tag(field, 5); this._push(u); return this; }
  bytes(field, u8) { this.tag(field, 2).varint(u8.length); this._push(u8); return this; }
  str(field, s) { return this.bytes(field, new TextEncoder().encode(s)); }
  msg(field, m) { this.tag(field, 2).varint(m.len); for (const p of m.parts) this._push(p); return this; }
  packedInts(field, arr) { const m = new Msg(); for (const v of arr) m.varint(v); return this.msg(field, m); }
  toBytes() { const out = new Uint8Array(this.len); let o = 0; for (const p of this.parts) { out.set(p, o); o += p.length; } return out; }
}

const toU8 = (dtype, data) => {
  if (data instanceof Uint8Array && dtype !== "uint8" && dtype !== "bool") return data;
  const ctor = { float32: Float32Array, float64: Float64Array, int32: Int32Array, int64: BigInt64Array, int8: Int8Array, uint8: Uint8Array, bool: Uint8Array, float16: Uint16Array }[dtype];
  let arr = data;
  if (!(arr instanceof ctor)) arr = dtype === "int64" ? BigInt64Array.from(Array.from(data, (x) => BigInt(x))) : ctor.from(data);
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
};

function tensorMsg(name, dtype, dims, data) {
  const m = new Msg();
  if (dims.length) m.packedInts(1, dims);
  m.int(2, DT[dtype]);
  if (name) m.str(8, name);
  m.bytes(9, toU8(dtype, data));
  return m;
}

function attrMsg(name, v) {
  const m = new Msg(); m.str(1, name);
  if (v instanceof F) { if (Array.isArray(v.v)) { for (const x of v.v) m.f32(7, x); m.int(20, 6); } else { m.f32(2, v.v); m.int(20, 1); } }
  else if (typeof v === "number") { if (Number.isInteger(v)) { m.int(3, v); m.int(20, 2); } else { m.f32(2, v); m.int(20, 1); } }
  else if (typeof v === "string") { m.bytes(4, new TextEncoder().encode(v)); m.int(20, 3); }
  else if (Array.isArray(v)) {
    if (v.every(Number.isInteger)) { for (const x of v) m.int(8, x); m.int(20, 7); }
    else { for (const x of v) m.f32(7, x); m.int(20, 6); }
  } else if (v && v.__tensor) { m.msg(5, tensorMsg("", v.__tensor.dtype, v.__tensor.dims, v.__tensor.data)); m.int(20, 4); }
  else throw new Error(`bad attribute ${name}=${v}`);
  return m;
}

function valueInfoMsg(name, dtype, shape) {
  const t = new Msg(); t.int(1, DT[dtype]);
  if (shape) {
    const s = new Msg();
    for (const d of shape) { const dm = new Msg(); if (typeof d === "string") dm.str(2, d); else dm.int(1, d); s.msg(1, dm); }
    t.msg(2, s);
  }
  const tp = new Msg(); tp.msg(1, t);
  const vi = new Msg(); vi.str(1, name); vi.msg(2, tp); return vi;
}

// ----------------------------------------------------------------------------- graph builder
export class Graph {
  constructor(name = "graph", { opset = 17 } = {}) {
    this.name = name; this.opset = opset; this.nodes = []; this.inits = []; this.inputs = []; this.outputs = []; this._n = 0; this._cache = new Map(); this.initNames = new Set();
    return new Proxy(this, { get: (t, p, r) => (p in t || typeof p !== "string" || !/^[A-Z]/.test(p)) ? Reflect.get(t, p, r) : (...a) => t._call(p, a) });
  }
  uid(p = "t") { return `${p}_${this._n++}`; }
  input(name, dtype, shape) { this.inputs.push(valueInfoMsg(name, dtype, shape)); return name; }
  output(name, dtype, shape) { this.outputs.push(valueInfoMsg(name, dtype, shape)); return name; }
  /** Add an initializer. data: typed array (or array). Returns its name. */
  tensor(name, dtype, dims, data) {
    name = name || this.uid("c");
    if (this.initNames.has(name)) throw new Error(`duplicate initializer ${name}`);
    this.initNames.add(name); this.inits.push(tensorMsg(name, dtype, dims, data)); return name;
  }
  f32(name, dims, data) { return this.tensor(name, "float32", dims, data); }
  /** cached constants */
  i64(list) { const a = Array.isArray(list) ? list : [list]; const k = "i64:" + a.join(","); if (!this._cache.has(k)) this._cache.set(k, this.tensor(null, "int64", [a.length], a)); return this._cache.get(k); }
  i64s(v) { const k = "i64s:" + v; if (!this._cache.has(k)) this._cache.set(k, this.tensor(null, "int64", [], [v])); return this._cache.get(k); }
  c(v, dtype = "float32") { const k = `s:${dtype}:${v}`; if (!this._cache.has(k)) this._cache.set(k, this.tensor(null, dtype, [], dtype === "int64" ? [v] : [v])); return this._cache.get(k); }
  _call(op, args) {
    let attrs = {};
    const last = args[args.length - 1];
    if (last && typeof last === "object" && !(last instanceof F) && !Array.isArray(last)) { attrs = last; args = args.slice(0, -1); }
    return this.nodeN(op, args, attrs, 1);
  }
  nodeN(op, ins, attrs = {}, nOut = 1) {
    const outs = Array.from({ length: nOut }, () => this.uid(op.toLowerCase()));
    const m = new Msg();
    for (const i of ins) m.str(1, i == null ? "" : i);
    for (const o of outs) m.str(2, o);
    m.str(3, this.uid("n_" + op)); m.str(4, op);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined) m.msg(5, attrMsg(k, v));
    this.nodes.push(m);
    return nOut === 1 ? outs[0] : outs;
  }
  toBytes() {
    const gm = new Msg();
    for (const n of this.nodes) gm.msg(1, n);
    gm.str(2, this.name);
    for (const t of this.inits) gm.msg(5, t);
    for (const i of this.inputs) gm.msg(11, i);
    for (const o of this.outputs) gm.msg(12, o);
    const model = new Msg();
    model.int(1, 8); model.str(2, "rvc-web");
    const os = new Msg(); os.str(1, ""); os.int(2, this.opset); model.msg(8, os);
    model.msg(7, gm);
    return model.toBytes();
  }
}
