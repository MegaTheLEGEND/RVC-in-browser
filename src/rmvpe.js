import { rmvpeLogMel } from "./dsp.js";

const CENTS = (() => { // np.pad(20*arange(360)+1997.3794084376191, (4,4))
  const c = new Float64Array(368);
  for (let i = 0; i < 360; i++) c[i + 4] = 20 * i + 1997.3794084376191;
  return c;
})();

/** Port of RMVPE.decode/to_local_average_cents. hidden: Float32Array [nFrames*360]. Returns Float32Array f0 (Hz, 0 = unvoiced). */
export function decodeRmvpe(hidden, nFrames, thred = 0.03) {
  const f0 = new Float32Array(nFrames);
  const sal = new Float64Array(368);
  for (let t = 0; t < nFrames; t++) {
    sal.fill(0);
    let best = 0, maxv = -Infinity;
    for (let k = 0; k < 360; k++) { const v = hidden[t * 360 + k]; sal[k + 4] = v; if (v > maxv) { maxv = v; best = k; } }
    // window [best, best+8] in padded coords == center-4 .. center+4
    let ps = 0, ws = 0;
    for (let k = best; k < best + 9; k++) { ps += sal[k] * CENTS[k]; ws += sal[k]; }
    let cents = ps / ws;
    if (maxv <= thred) cents = 0;
    let f = 10 * Math.pow(2, cents / 1200);
    f0[t] = f === 10 ? 0 : f;
  }
  return f0;
}

export class Rmvpe {
  constructor(ort, session) { this.ort = ort; this.session = session; }
  async infer(audio16k, thred = 0.03) {
    const { data, nFrames } = rmvpeLogMel(audio16k);
    const nPad = 32 * Math.floor((nFrames - 1) / 32 + 1) - nFrames;
    const T = nFrames + nPad;
    const mel = new Float32Array(128 * T); // zero padded on the right (constant 0, like F.pad)
    for (let m = 0; m < 128; m++) mel.set(data.subarray(m * nFrames, (m + 1) * nFrames), m * T);
    const input = new this.ort.Tensor("float32", mel, [1, 128, T]);
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const hid = out[this.session.outputNames[0]]; // [1, T, 360]
    return decodeRmvpe(hid.data.subarray(0, nFrames * 360), nFrames, thred);
  }
}
