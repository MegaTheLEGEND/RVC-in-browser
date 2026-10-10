import { Rng } from "./dsp.js";

/** Wraps an exported RVC generator. Noise is supplied from JS (rnd / snd) so the graph is deterministic and WebGPU-friendly. */
export class Synth {
  constructor(ort, session, meta, seed = 1234) {
    this.ort = ort; this.session = session; this.meta = meta; this.rng = new Rng(seed);
    this.ifF0 = meta.f0 !== 0 && meta.f0 !== false; this.tgtSr = meta.sr; this.upp = meta.upp; this.inter = meta.inter_channels; this.phoneDim = meta.phone_dim;
  }
  /** feats: Float32Array [T*C]; pitch: Int32Array/Array [T] (coarse 1..255); pitchf: Float32Array [T]. Returns Float32Array [(T-skip)*upp]. */
  async run({ feats, T, pitch, pitchf, sid, skip = 0 }) {
    const { ort } = this, f = {};
    f.phone = new ort.Tensor("float32", feats, [1, T, this.phoneDim]);
    f.sid = new ort.Tensor("int64", BigInt64Array.from([BigInt(sid)]), [1]);
    f.rnd = new ort.Tensor("float32", this.rng.fillNormal(new Float32Array(this.inter * T)), [1, this.inter, T]);
    if (this.ifF0) {
      const pi = new BigInt64Array(T); for (let i = 0; i < T; i++) pi[i] = BigInt(pitch[i]);
      f.pitch = new ort.Tensor("int64", pi, [1, T]);
      f.pitchf = new ort.Tensor("float32", Float32Array.from(pitchf.subarray(0, T)), [1, T]);
      f.snd = new ort.Tensor("float32", this.rng.fillNormal(new Float32Array(T * this.upp)), [1, T * this.upp, 1]);
    }
    f.skip = new ort.Tensor("int64", BigInt64Array.from([BigInt(skip)]), [1]);
    const feeds = {}; for (const n of this.session.inputNames) { if (!(n in f)) throw new Error(`voice model expects unknown input "${n}"`); feeds[n] = f[n]; }
    const out = await this.session.run(feeds);
    return Float32Array.from(out[this.session.outputNames[0]].data);
  }
}
