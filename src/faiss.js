// Direct reader for faiss `.index` files (IndexIVFFlat — what RVC trains — and plain IndexFlat), plus RVC's retrieval blend.
// Search matches faiss: nearest `nprobe` coarse cells (value stored in the file, RVC saves 1), exact squared-L2 inside them, top-8.
const fourcc = (s) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24);

class R {
  constructor(buf) { this.u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf); this.dv = new DataView(this.u8.buffer, this.u8.byteOffset, this.u8.byteLength); this.p = 0; }
  u32() { const v = this.dv.getUint32(this.p, true); this.p += 4; return v; }
  i32() { const v = this.dv.getInt32(this.p, true); this.p += 4; return v; }
  u8_() { return this.u8[this.p++]; }
  i64() { const v = Number(this.dv.getBigInt64(this.p, true)); this.p += 8; return v; }
  f32() { const v = this.dv.getFloat32(this.p, true); this.p += 4; return v; }
  /** copy n float32 values */
  floats(n) { const o = new Float32Array(n); new Uint8Array(o.buffer).set(this.u8.subarray(this.p, this.p + n * 4)); this.p += n * 4; return o; }
  i64s(n) { const o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = this.i64(); return o; }
}

function readHeader(r) {
  const d = r.i32(), ntotal = r.i64(); r.i64(); r.i64(); r.u8_(); const metric = r.i32(); if (metric > 1) r.f32();
  return { d, ntotal, metric };
}

function readFlat(r, name) { // after fourcc
  const h = readHeader(r); const n = r.i64(); // vector length: bytes (new faiss) or floats (old faiss)
  const count = h.ntotal * h.d, vec = n === count * 4 ? r.floats(count) : n === count ? r.floats(count) : (() => { throw new Error(`${name}: unexpected flat storage size`); })();
  return { ...h, vectors: vec };
}

function readInvLists(r, d) {
  const tag = r.u32(); if (tag !== fourcc("ilar")) throw new Error("unsupported inverted-list type (only ArrayInvertedLists)");
  const nlist = r.i64(), codeSize = r.i64(); if (codeSize !== d * 4) throw new Error("index is not a Flat (float32) IVF index");
  const type = r.u32(), sizes = new Array(nlist).fill(0);
  if (type === fourcc("full")) { const n = r.i64(); for (let i = 0; i < n; i++) sizes[i] = r.i64(); }
  else if (type === fourcc("sprs")) { const n = r.i64(); for (let i = 0; i < n; i += 2) { const li = r.i64(), sz = r.i64(); sizes[li] = sz; } }
  else throw new Error("unknown inverted-list layout");
  const lists = [];
  for (let i = 0; i < nlist; i++) { const n = sizes[i]; if (!n) { lists.push({ n: 0, vecs: new Float32Array(0), ids: new Float64Array(0) }); continue; } const vecs = r.floats(n * d); const ids = r.i64s(n); lists.push({ n, vecs, ids }); }
  return lists;
}

export class FaissIndex {
  constructor(buf) {
    const r = new R(buf), tag = r.u32();
    if (tag === fourcc("IwFl") || tag === fourcc("IvFl")) { // IVFFlat
      const h = readHeader(r); this.dim = h.d; this.ntotal = h.ntotal;
      const nlist = r.i64(), nprobe = r.i64();
      const qtag = r.u32(); if (qtag !== fourcc("IxF2") && qtag !== fourcc("IxFl")) throw new Error("only flat-L2 coarse quantizers are supported");
      const q = readFlat(r, "quantizer");
      this.centroids = q.vectors; this.nlist = nlist; this.nprobe = Math.max(1, Math.min(nprobe, nlist));
      // direct map (ignored) then invlists
      r.u8_(); const nm = r.i64(); r.p += nm * 8; // dm.array
      this.lists = readInvLists(r, h.d);
    } else if (tag === fourcc("IxF2") || tag === fourcc("IxFl")) {
      const f = readFlat(r, "flat"); this.dim = f.d; this.ntotal = f.ntotal; this.nlist = 1; this.nprobe = 1;
      this.centroids = new Float32Array(f.d); this.lists = [{ n: f.ntotal, vecs: f.vectors, ids: Float64Array.from({ length: f.ntotal }, (_, i) => i) }];
    } else throw new Error(`unsupported faiss index type "${String.fromCharCode(tag & 255, (tag >> 8) & 255, (tag >> 16) & 255, (tag >> 24) & 255)}" — RVC indexes are IVF*,Flat`);
  }

  /** Nearest-k squared-L2 within the nprobe closest cells. Returns {d: Float64Array(k), v: Int32Array of (list,row) packed as two arrays}. */
  _search(q, k) {
    const { dim, nlist, nprobe, centroids, lists } = this;
    const cd = new Float64Array(nlist);
    for (let c = 0; c < nlist; c++) { let s = 0; const o = c * dim; for (let j = 0; j < dim; j++) { const t = q[j] - centroids[o + j]; s += t * t; } cd[c] = s; }
    const order = nprobe === 1 ? [cd.indexOf(Math.min(...cd))] : Array.from(cd.keys()).sort((a, b) => cd[a] - cd[b]).slice(0, nprobe);
    const bestD = new Float64Array(k).fill(Infinity), bestL = new Int32Array(k).fill(-1), bestR = new Int32Array(k).fill(-1);
    for (const c of order) {
      const { n, vecs } = lists[c];
      for (let i = 0; i < n; i++) {
        let s = 0; const o = i * dim; for (let j = 0; j < dim; j++) { const t = q[j] - vecs[o + j]; s += t * t; }
        if (s < bestD[k - 1]) { let p = k - 1; while (p > 0 && bestD[p - 1] > s) { bestD[p] = bestD[p - 1]; bestL[p] = bestL[p - 1]; bestR[p] = bestR[p - 1]; p--; } bestD[p] = s; bestL[p] = c; bestR[p] = i; }
      }
    }
    return { bestD, bestL, bestR };
  }

  /** RVC blend: weights (1/score)^2 over the top-8, normalised; returns [T*dim]. Missing neighbours get weight 0 (faiss -1 ids). */
  blend(feats, T, K = 8) {
    const { dim } = this, out = new Float32Array(T * dim);
    for (let t = 0; t < T; t++) {
      const q = feats.subarray(t * dim, (t + 1) * dim), { bestD, bestL, bestR } = this._search(q, K);
      let ws = 0; const w = new Float64Array(K);
      for (let m = 0; m < K; m++) { if (bestL[m] < 0) continue; const s = Math.max(bestD[m], 1e-12); w[m] = (1 / s) ** 2; ws += w[m]; }
      if (!ws) { out.set(q, t * dim); continue; }
      for (let m = 0; m < K; m++) { if (bestL[m] < 0) continue; const wm = w[m] / ws, v = this.lists[bestL[m]].vecs, o = bestR[m] * dim; for (let j = 0; j < dim; j++) out[t * dim + j] += wm * v[o + j]; }
    }
    return out;
  }
}
