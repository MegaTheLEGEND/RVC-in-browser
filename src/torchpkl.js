// Read PyTorch checkpoints (.pth/.pt/.bin zip format with a pickle, and .safetensors) in plain JS — no Python, no torch.
// Works in browsers (DecompressionStream) and Node >= 18.
//
//   const ck = await loadTorch(arrayBuffer);        // -> plain object graph; tensors are `TorchTensor`
//   ck.weight["enc_p.emb_phone.weight"].f32()       // Float32Array (fp16/bf16/f64 converted)
//   ck.weight[...].shape                            // [out, in]
//
// Unknown pickled classes become inert `PyObj` stubs, so checkpoints that reference numpy / argparse / etc. still load.

const te = new TextDecoder("utf-8");

export function halfToFloat(h) {
  const e = (h >> 10) & 0x1f, m = h & 0x3ff; let v;
  if (e === 0) v = m * 2 ** -24; else if (e === 31) v = m ? NaN : Infinity; else v = (1 + m / 1024) * 2 ** (e - 15);
  return h & 0x8000 ? -v : v;
}

const DTYPES = {
  FloatStorage: ["float32", 4], HalfStorage: ["float16", 2], BFloat16Storage: ["bfloat16", 2], DoubleStorage: ["float64", 8],
  LongStorage: ["int64", 8], IntStorage: ["int32", 4], ShortStorage: ["int16", 2], CharStorage: ["int8", 1], ByteStorage: ["uint8", 1], BoolStorage: ["bool", 1],
};

export class PyObj { constructor(cls, args) { this.__class = cls; this.__args = args; } __setstate__(s) { this.__state = s; } }
export class StorageRef { constructor(dtype, size, key, numel, getBytes) { this.dtype = dtype; this.size = size; this.key = key; this.numel = numel; this.getBytes = getBytes; } }

export class TorchTensor {
  constructor(storage, offset, shape, stride) { this.storage = storage; this.offset = offset; this.shape = shape; this.stride = stride; this.dtype = storage.dtype; }
  get numel() { return this.shape.reduce((a, b) => a * b, 1); }
  /** Converts to a contiguous Float32Array (fp16 / bf16 / f64 / ints are converted). */
  f32() {
    const n = this.numel, { dtype, size } = this.storage, bytes = this.storage.getBytes();
    const contiguous = (() => { let s = 1; for (let i = this.shape.length - 1; i >= 0; i--) { if (this.shape[i] !== 1 && this.stride[i] !== s) return false; s *= this.shape[i]; } return true; })();
    let src;
    const bo = bytes.byteOffset;
    const view = (Ctor) => ((bo % Ctor.BYTES_PER_ELEMENT === 0) ? new Ctor(bytes.buffer, bo, Math.floor(bytes.byteLength / Ctor.BYTES_PER_ELEMENT)) : new Ctor(bytes.slice().buffer));
    switch (dtype) {
      case "float32": src = view(Float32Array); break;
      case "float64": src = view(Float64Array); break;
      case "float16": { const u = view(Uint16Array); src = { length: u.length, at: (i) => halfToFloat(u[i]) }; break; }
      case "bfloat16": { const u = view(Uint16Array), t = new Uint32Array(1), f = new Float32Array(t.buffer); src = { length: u.length, at: (i) => { t[0] = u[i] << 16; return f[0]; } }; break; }
      case "int64": { const b = view(BigInt64Array); src = { length: b.length, at: (i) => Number(b[i]) }; break; }
      case "int32": src = view(Int32Array); break; case "int16": src = view(Int16Array); break;
      case "int8": src = view(Int8Array); break; case "uint8": case "bool": src = view(Uint8Array); break;
      default: throw new Error(`unsupported tensor dtype ${dtype}`);
    }
    const get = src.at && typeof src.at === "function" && !ArrayBuffer.isView(src) ? (i) => src.at(i) : (i) => src[i];
    const out = new Float32Array(n);
    if (contiguous) {
      if (src instanceof Float32Array) out.set(src.subarray(this.offset, this.offset + n));
      else for (let i = 0; i < n; i++) out[i] = get(this.offset + i);
    } else { // generic strided gather
      const nd = this.shape.length, idx = new Array(nd).fill(0);
      for (let i = 0; i < n; i++) {
        let p = this.offset; for (let d = 0; d < nd; d++) p += idx[d] * this.stride[d];
        out[i] = get(p);
        for (let d = nd - 1; d >= 0; d--) { if (++idx[d] < this.shape[d]) break; idx[d] = 0; }
      }
    }
    return out;
  }
}

// ----------------------------------------------------------------------------- zip
async function inflateRaw(u8) {
  const ds = new DecompressionStream("deflate-raw");
  const stream = new Blob([u8]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Minimal zip reader. Returns Map<name, {method, csize, usize, dataStart}>. Supports zip64 sizes/offsets. */
export function readZipDirectory(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 22 - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("not a zip file");
  let count = dv.getUint16(eocd + 10, true), cdSize = dv.getUint32(eocd + 12, true), cdOff = dv.getUint32(eocd + 16, true);
  if (cdOff === 0xffffffff || count === 0xffff) { // zip64
    const loc = eocd - 20;
    if (loc >= 0 && dv.getUint32(loc, true) === 0x07064b50) {
      const z = Number(dv.getBigUint64(loc + 8, true));
      count = Number(dv.getBigUint64(z + 32, true)); cdSize = Number(dv.getBigUint64(z + 40, true)); cdOff = Number(dv.getBigUint64(z + 48, true));
    }
  }
  const entries = new Map(); let p = cdOff;
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("bad zip central directory");
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true), off = dv.getUint32(p + 42, true);
    const nl = dv.getUint16(p + 28, true), el = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true);
    const name = te.decode(u8.subarray(p + 46, p + 46 + nl));
    if (csize === 0xffffffff || usize === 0xffffffff || off === 0xffffffff) {
      let q = p + 46 + nl; const end = q + el;
      while (q + 4 <= end) {
        const id = dv.getUint16(q, true), sz = dv.getUint16(q + 2, true);
        if (id === 1) { let r = q + 4; if (usize === 0xffffffff) { usize = Number(dv.getBigUint64(r, true)); r += 8; } if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(r, true)); r += 8; } if (off === 0xffffffff) { off = Number(dv.getBigUint64(r, true)); } }
        q += 4 + sz;
      }
    }
    const lnl = dv.getUint16(off + 26, true), lel = dv.getUint16(off + 28, true);
    entries.set(name, { method, csize, usize, dataStart: off + 30 + lnl + lel });
    p += 46 + nl + el + cl;
  }
  return entries;
}

// ----------------------------------------------------------------------------- pickle (protocols 0-5, the subset torch emits)
const MARK = Symbol("mark");
function unpickle(buf, { findClass, persistentLoad }) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const stack = [], memo = new Map(); let pos = 0, memoN = 0;
  const popMark = () => { let i = stack.length - 1; while (stack[i] !== MARK) i--; const items = stack.splice(i + 1); stack.pop(); return items; };
  const readLine = () => { let e = pos; while (buf[e] !== 10) e++; const s = te.decode(buf.subarray(pos, e)); pos = e + 1; return s; };
  const readLong = (n) => { let v = 0n; for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(buf[pos + i]); if (n && buf[pos + n - 1] & 0x80) v -= 1n << BigInt(8 * n); pos += n; return Math.abs(Number(v)) <= Number.MAX_SAFE_INTEGER ? Number(v) : v; };
  for (;;) {
    const op = buf[pos++];
    switch (op) {
      case 0x80: pos++; break; // PROTO
      case 0x95: pos += 8; break; // FRAME
      case 0x2e: return stack.pop(); // STOP
      case 0x28: stack.push(MARK); break; // MARK
      case 0x30: stack.pop(); break; // POP
      case 0x31: popMark(); break; // POP_MARK
      case 0x32: stack.push(stack[stack.length - 1]); break; // DUP
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; case 0x89: stack.push(false); break;
      case 0x49: { const s = readLine(); stack.push(s === "00" ? false : s === "01" ? true : Number(s)); break; } // INT
      case 0x4a: stack.push(dv.getInt32(pos, true)); pos += 4; break; // BININT
      case 0x4b: stack.push(buf[pos++]); break; // BININT1
      case 0x4d: stack.push(dv.getUint16(pos, true)); pos += 2; break; // BININT2
      case 0x4c: { const s = readLine().replace(/L$/, ""); stack.push(Number(s)); break; } // LONG
      case 0x8a: { const n = buf[pos++]; stack.push(readLong(n)); break; } // LONG1
      case 0x8b: { const n = dv.getUint32(pos, true); pos += 4; stack.push(readLong(n)); break; } // LONG4
      case 0x46: stack.push(Number(readLine())); break; // FLOAT
      case 0x47: stack.push(dv.getFloat64(pos, false)); pos += 8; break; // BINFLOAT
      case 0x53: { const s = readLine(); stack.push(s.replace(/^['"]|['"]$/g, "")); break; } // STRING
      case 0x54: { const n = dv.getInt32(pos, true); pos += 4; stack.push(te.decode(buf.subarray(pos, pos + n))); pos += n; break; } // BINSTRING
      case 0x55: { const n = buf[pos++]; stack.push(te.decode(buf.subarray(pos, pos + n))); pos += n; break; } // SHORT_BINSTRING
      case 0x56: stack.push(readLine()); break; // UNICODE
      case 0x58: { const n = dv.getUint32(pos, true); pos += 4; stack.push(te.decode(buf.subarray(pos, pos + n))); pos += n; break; } // BINUNICODE
      case 0x8c: { const n = buf[pos++]; stack.push(te.decode(buf.subarray(pos, pos + n))); pos += n; break; } // SHORT_BINUNICODE
      case 0x8d: { const n = Number(dv.getBigUint64(pos, true)); pos += 8; stack.push(te.decode(buf.subarray(pos, pos + n))); pos += n; break; } // BINUNICODE8
      case 0x42: { const n = dv.getUint32(pos, true); pos += 4; stack.push(buf.slice(pos, pos + n)); pos += n; break; } // BINBYTES
      case 0x43: { const n = buf[pos++]; stack.push(buf.slice(pos, pos + n)); pos += n; break; } // SHORT_BINBYTES
      case 0x8e: { const n = Number(dv.getBigUint64(pos, true)); pos += 8; stack.push(buf.slice(pos, pos + n)); pos += n; break; } // BINBYTES8
      case 0x96: { const n = Number(dv.getBigUint64(pos, true)); pos += 8; stack.push(buf.slice(pos, pos + n)); pos += n; break; } // BYTEARRAY8
      case 0x29: stack.push([]); stack[stack.length - 1].__tuple = true; break; // EMPTY_TUPLE
      case 0x74: { const it = popMark(); it.__tuple = true; stack.push(it); break; } // TUPLE
      case 0x85: { const a = [stack.pop()]; a.__tuple = true; stack.push(a); break; }
      case 0x86: { const b = stack.pop(), a = stack.pop(), t = [a, b]; t.__tuple = true; stack.push(t); break; }
      case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(), t = [a, b, c]; t.__tuple = true; stack.push(t); break; }
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x6c: stack.push(popMark()); break; // LIST
      case 0x61: { const v = stack.pop(); stack[stack.length - 1].push(v); break; } // APPEND
      case 0x65: { const it = popMark(); const l = stack[stack.length - 1]; for (const v of it) l.push(v); break; } // APPENDS
      case 0x7d: stack.push(new Map()); break; // EMPTY_DICT  (Map keeps non-string keys; converted later)
      case 0x64: { const it = popMark(); const m = new Map(); for (let i = 0; i < it.length; i += 2) m.set(it[i], it[i + 1]); stack.push(m); break; } // DICT
      case 0x73: { const v = stack.pop(), k = stack.pop(); setItem(stack[stack.length - 1], k, v); break; } // SETITEM
      case 0x75: { const it = popMark(); const d = stack[stack.length - 1]; for (let i = 0; i < it.length; i += 2) setItem(d, it[i], it[i + 1]); break; } // SETITEMS
      case 0x8f: stack.push(new Set()); break; // EMPTY_SET
      case 0x90: { const it = popMark(); const s = stack[stack.length - 1]; for (const v of it) s.add(v); break; } // ADDITEMS
      case 0x91: { const it = popMark(); stack.push(new Set(it)); break; } // FROZENSET
      case 0x71: memo.set(buf[pos++], stack[stack.length - 1]); break; // BINPUT
      case 0x72: memo.set(dv.getUint32(pos, true), stack[stack.length - 1]); pos += 4; break; // LONG_BINPUT
      case 0x94: memo.set(memoN++, stack[stack.length - 1]); break; // MEMOIZE
      case 0x70: memo.set(Number(readLine()), stack[stack.length - 1]); break; // PUT
      case 0x68: stack.push(memo.get(buf[pos++])); break; // BINGET
      case 0x6a: stack.push(memo.get(dv.getUint32(pos, true))); pos += 4; break; // LONG_BINGET
      case 0x67: stack.push(memo.get(Number(readLine()))); break; // GET
      case 0x63: { const m = readLine(), n = readLine(); stack.push(findClass(m, n)); break; } // GLOBAL
      case 0x93: { const n = stack.pop(), m = stack.pop(); stack.push(findClass(m, n)); break; } // STACK_GLOBAL
      case 0x52: { const args = stack.pop(), f = stack.pop(); stack.push(callObj(f, args)); break; } // REDUCE
      case 0x81: { const args = stack.pop(), c = stack.pop(); stack.push(callObj(c, args)); break; } // NEWOBJ
      case 0x92: { const kw = stack.pop(), args = stack.pop(), c = stack.pop(); stack.push(callObj(c, args)); break; } // NEWOBJ_EX
      case 0x62: { const st = stack.pop(), o = stack[stack.length - 1]; build(o, st); break; } // BUILD
      case 0x51: stack.push(persistentLoad(stack.pop())); break; // BINPERSID
      case 0x50: stack.push(persistentLoad(readLine())); break; // PERSID
      case 0x6f: case 0x69: case 0x82: case 0x83: case 0x84: default:
        throw new Error(`unsupported pickle opcode 0x${op.toString(16)} at ${pos - 1}`);
    }
  }
}
function setItem(d, k, v) { if (d instanceof Map) d.set(k, v); else if (d && typeof d === "object") d[k] = v; }
function callObj(f, args) {
  if (typeof f === "function") return f(...(args || []));
  if (f instanceof StubClass) return f.construct(args);
  return new PyObj(String(f), args);
}
function build(o, st) {
  if (o && typeof o.__setstate__ === "function") return o.__setstate__(st);
  if (o instanceof Map && st instanceof Map) for (const [k, v] of st) o.set(k, v);
  else if (o && typeof o === "object") Object.assign(o, st instanceof Map ? Object.fromEntries(st) : { __state: st });
}
class StubClass { constructor(name) { this.name = name; } construct(args) { return new PyObj(this.name, args); } }

function normalize(x, seen = new Map()) { // Map -> plain object (string keys) / Map (other keys); tuples -> arrays
  if (x instanceof Map) {
    if (seen.has(x)) return seen.get(x);
    const allStr = [...x.keys()].every((k) => typeof k === "string");
    if (!allStr) { const m = new Map(); seen.set(x, m); for (const [k, v] of x) m.set(k, normalize(v, seen)); return m; }
    const o = {}; seen.set(x, o); for (const [k, v] of x) o[k] = normalize(v, seen); return o;
  }
  if (Array.isArray(x)) { if (seen.has(x)) return seen.get(x); const a = []; seen.set(x, a); for (const v of x) a.push(normalize(v, seen)); return a; }
  if (x instanceof PyObj && x.__state && typeof x.__state === "object") { for (const k of Object.keys(x)) { /* keep */ } }
  return x;
}

// ----------------------------------------------------------------------------- torch zip checkpoint
export async function loadTorchZip(u8) {
  const entries = readZipDirectory(u8);
  const names = [...entries.keys()];
  const pklName = names.find((n) => /(^|\/)data\.pkl$/.test(n));
  if (!pklName) throw new Error("zip has no data.pkl — not a PyTorch checkpoint");
  const prefix = pklName.slice(0, pklName.length - "data.pkl".length);
  const read = async (name) => {
    const e = entries.get(name); if (!e) throw new Error(`missing ${name}`);
    const raw = u8.subarray(e.dataStart, e.dataStart + e.csize);
    return e.method === 0 ? raw : e.method === 8 ? inflateRaw(raw) : (() => { throw new Error(`zip method ${e.method} unsupported`); })();
  };
  const pkl = await read(pklName);
  // storages: loaded lazily on first use, but getBytes() must be synchronous -> pre-resolve the ones that are referenced
  const wanted = new Map(); // key -> StorageRef
  const findClass = (mod, name) => {
    if (mod === "collections" && name === "OrderedDict") return (arg) => { const m = new Map(); if (Array.isArray(arg)) for (const [k, v] of arg) m.set(k, v); return m; };
    if (mod === "torch._utils" && name === "_rebuild_tensor_v2") return (storage, offset, size, stride) => new TorchTensor(storage, offset, [...size], [...stride]);
    if (mod === "torch._utils" && name === "_rebuild_tensor") return (storage, offset, size, stride) => new TorchTensor(storage, offset, [...size], [...stride]);
    if (mod === "torch._utils" && name === "_rebuild_parameter") return (data) => data;
    if (mod === "torch._utils" && name === "_rebuild_parameter_with_state") return (data) => data;
    if (mod === "torch.serialization" && name === "_get_layout") return (n) => n;
    if (mod === "torch" && DTYPES[name]) return { __storageType: name };
    if (mod === "torch" && /Storage$/.test(name)) return { __storageType: name };
    if (mod === "torch" && /^(float|half|bfloat|double|int|long|bool|uint|short|complex)/.test(name)) return { __dtype: name };
    if (mod === "builtins" && name === "set") return (a) => new Set(a && a[0] ? a[0] : []);
    if (mod === "builtins" && name === "list") return (a) => (a ? [...a] : []);
    if (mod === "builtins" && name === "dict") return (a) => new Map(a || []);
    if (mod === "builtins" && name === "tuple") return (a) => (a ? [...a] : []);
    if (mod === "_codecs" && name === "encode") return (s) => s;
    return new StubClass(`${mod}.${name}`);
  };
  const persistentLoad = (pid) => {
    if (!Array.isArray(pid) || pid[0] !== "storage") throw new Error("unsupported persistent id");
    const [, stype, key, , numel] = pid;
    const tn = stype && stype.__storageType, d = DTYPES[tn];
    if (!d) throw new Error(`unsupported storage type ${tn}`);
    const ref = new StorageRef(d[0], d[1], String(key), numel, () => { if (!ref._bytes) throw new Error("storage not loaded"); return ref._bytes; });
    wanted.set(ref.key, ref);
    return ref;
  };
  const root = unpickle(pkl, { findClass, persistentLoad });
  for (const ref of wanted.values()) ref._bytes = await read(`${prefix}data/${ref.key}`);
  return normalize(root);
}

// ----------------------------------------------------------------------------- safetensors
const ST = { F32: ["float32", 4], F16: ["float16", 2], BF16: ["bfloat16", 2], F64: ["float64", 8], I64: ["int64", 8], I32: ["int32", 4], I16: ["int16", 2], I8: ["int8", 1], U8: ["uint8", 1], BOOL: ["bool", 1] };
export function loadSafetensors(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const hl = Number(dv.getBigUint64(0, true)), hdr = JSON.parse(te.decode(u8.subarray(8, 8 + hl))), base = 8 + hl, out = {};
  for (const [k, v] of Object.entries(hdr)) {
    if (k === "__metadata__") continue;
    const [dtype, size] = ST[v.dtype] || (() => { throw new Error(`safetensors dtype ${v.dtype}`); })();
    const bytes = u8.subarray(base + v.data_offsets[0], base + v.data_offsets[1]);
    const ref = new StorageRef(dtype, size, k, bytes.length / size, () => bytes);
    let s = 1; const stride = new Array(v.shape.length); for (let i = v.shape.length - 1; i >= 0; i--) { stride[i] = s; s *= v.shape[i]; }
    out[k] = new TorchTensor(ref, 0, v.shape, stride);
  }
  return out;
}

/** Detects format by content: zip → torch checkpoint; otherwise safetensors. Accepts ArrayBuffer | Uint8Array. */
export async function loadTorch(input) {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (u8[0] === 0x50 && u8[1] === 0x4b) return loadTorchZip(u8);
  if (u8[0] === 0x80) throw new Error("legacy (pre-1.6, non-zip) torch checkpoints are not supported — re-save with a recent torch.save");
  try { return loadSafetensors(u8); } catch (e) { throw new Error(`unrecognised checkpoint format (${e.message})`); }
}
