// Faithful JS port of infer/vc/pipeline.py (Pipeline.pipeline / vc / get_f0 post-processing).
import { butterHighpass, filtfilt, padReflect, resample, changeRms } from "./dsp.js";
import { CHUNK } from "./config.js";

const WINDOW = 160, SR = 16000;
const F0_MIN = 50, F0_MAX = 1100;
const MEL_MIN = 1127 * Math.log(1 + F0_MIN / 700), MEL_MAX = 1127 * Math.log(1 + F0_MAX / 700);
let _hp = null;

const rint = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; }; // numpy round-half-even

/** get_f0 post-processing: fill unvoiced by interpolation (np.interp semantics), pitch shift, coarse quantisation. */
export function postprocessF0(f0in, upKey) {
  const f0 = Float32Array.from(f0in);
  const voiced = []; for (let i = 0; i < f0.length; i++) if (f0[i] !== 0) voiced.push(i);
  if (voiced.length > 0 && voiced.length < f0.length) {
    let j = 0;
    for (let i = 0; i < f0.length; i++) {
      if (f0[i] !== 0) continue;
      while (j + 1 < voiced.length && voiced[j + 1] < i) j++;
      if (i < voiced[0]) f0[i] = f0[voiced[0]];
      else if (i > voiced[voiced.length - 1]) f0[i] = f0[voiced[voiced.length - 1]];
      else { const a = voiced[j], b = voiced[j + 1]; f0[i] = f0[a] + ((f0[b] - f0[a]) * (i - a)) / (b - a); }
    }
  }
  const mul = Math.pow(2, upKey / 12);
  const pitchf = new Float32Array(f0.length), coarse = new Int32Array(f0.length);
  for (let i = 0; i < f0.length; i++) {
    const v = f0[i] * mul; pitchf[i] = v;
    let m = 1127 * Math.log(1 + v / 700);
    if (m > 0) m = ((m - MEL_MIN) * 254) / (MEL_MAX - MEL_MIN) + 1;
    if (m <= 1) m = 1; if (m > 255) m = 255;
    coarse[i] = rint(m);
  }
  return { pitch: coarse, pitchf };
}

function upsample2(data, T, C) { const o = new Float32Array(T * 2 * C); for (let t = 0; t < T; t++) { const src = data.subarray(t * C, (t + 1) * C); o.set(src, 2 * t * C); o.set(src, (2 * t + 1) * C); } return o; }

/** Splits long audio at quiet points (opt_ts in the Python code). */
export function findCutPoints(audioPad, audioLen, { tMax, tCenter, tQuery }) {
  const opt = [];
  if (audioPad.length <= tMax) return opt;
  const P = new Float64Array(audioPad.length + 1);
  for (let i = 0; i < audioPad.length; i++) P[i + 1] = P[i] + Math.abs(audioPad[i]);
  const asum = (k) => P[k + WINDOW] - P[k];
  for (let t = tCenter; t < audioLen; t += tCenter) {
    const lo = t - tQuery, hi = Math.min(t + tQuery, audioLen);
    let best = Infinity, bi = 0;
    for (let k = lo; k < hi; k++) { const v = asum(k); if (v < best) { best = v; bi = k - lo; } }
    opt.push(lo + bi);
  }
  return opt;
}

/**
 * @param audio16k Float32Array mono @16 kHz
 * @param be {hubert, f0 (or null), synth, index (or null)}
 * @param p  {sid, f0UpKey, indexRate, resampleSr, rmsMixRate, protect}
 * @returns {{samples: Int16Array, sampleRate: number, timings: object}}
 */
export async function convert(audio16k, be, p, onProgress = () => {}) {
  const { synth } = be, tgtSr = synth.tgtSr, ifF0 = synth.ifF0 && be.f0;
  const xPad = p.xPad ?? CHUNK.xPad, tPad = Math.round(SR * xPad), tPadTgt = Math.round(tgtSr * xPad), tPad2 = tPad * 2;
  const tQuery = SR * CHUNK.xQuery, tCenter = SR * CHUNK.xCenter, tMax = SR * CHUNK.xMax;
  const timings = { hubert: 0, f0: 0, synth: 0 };
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  let audio = Float32Array.from(audio16k);
  let mx = 0; for (let i = 0; i < audio.length; i++) mx = Math.max(mx, Math.abs(audio[i]));
  if (mx / 0.95 > 1) { const s = mx / 0.95; for (let i = 0; i < audio.length; i++) audio[i] /= s; }

  onProgress("filter", 0);
  if (!_hp) _hp = butterHighpass(5, 48, SR);
  audio = Float32Array.from(filtfilt(_hp.b, _hp.a, audio));
  const audioPadW = padReflect(audio, WINDOW >> 1, WINDOW >> 1);
  const optTs = findCutPoints(audioPadW, audio.length, { tMax, tCenter, tQuery });

  const audioPad = padReflect(audio, tPad, tPad);
  const pLen = Math.floor(audioPad.length / WINDOW);
  let pitch = null, pitchf = null;
  if (ifF0) {
    onProgress("f0", 0); const t0 = now();
    const raw = await be.f0.infer(audioPad);
    const pp = postprocessF0(raw, p.f0UpKey);
    pitch = new Int32Array(pLen); pitchf = new Float32Array(pLen);
    for (let i = 0; i < pLen; i++) { const j = Math.min(i, pp.pitch.length - 1); pitch[i] = pp.pitch[j]; pitchf[i] = pp.pitchf[j]; }
    timings.f0 += now() - t0;
  }

  const vc = async (chunk, pit, pitf) => {
    const t0 = now();
    const h = await be.hubert.run(chunk);
    let { data: feats, T, C } = h;
    timings.hubert += now() - t0;
    const useProtect = p.protect < 0.5 && pit && pitf;
    const feats0 = useProtect ? Float32Array.from(feats) : null;
    if (be.index && p.indexRate !== 0) {
      const blended = be.index.blend(feats, T);
      const f = new Float32Array(feats.length);
      for (let i = 0; i < f.length; i++) f[i] = blended[i] * p.indexRate + (1 - p.indexRate) * feats[i];
      feats = f;
    }
    feats = upsample2(feats, T, C);
    const feats0u = useProtect ? upsample2(feats0, T, C) : null;
    T *= 2;
    let pl = Math.floor(chunk.length / WINDOW);
    if (T < pl) { pl = T; if (pit) { pit = pit.subarray(0, pl); pitf = pitf.subarray(0, pl); } }
    if (T > pl) { feats = feats.subarray(0, pl * C); T = pl; }
    if (useProtect) {
      for (let t = 0; t < pl; t++) { const w = pitf[t] >= 1 ? 1 : p.protect; for (let c = 0; c < C; c++) { const i = t * C + c; feats[i] = feats[i] * w + feats0u[i] * (1 - w); } }
    }
    const t1 = now();
    // realtime: generate only the last (keep + pad + 6 warm-up) frames; the encoder/flow still see the whole window
    const skip = p.keep ? Math.max(0, pl - Math.ceil(p.keep / WINDOW) - (tPad / WINDOW | 0) - 6) : 0;
    const out = await be.synth.run({ feats, T: pl, pitch: pit, pitchf: pitf, sid: p.sid, skip });
    out.skipped = skip;
    timings.synth += now() - t1;
    return out;
  };

  const trim = (o) => o.subarray(o.skipped ? 0 : tPadTgt, o.length - tPadTgt);
  const sl = (a, s, e) => (a ? a.subarray(s, e) : null);
  const parts = []; let s = 0, t = null;
  for (let n = 0; n < optTs.length; n++) {
    t = Math.floor(optTs[n] / WINDOW) * WINDOW;
    onProgress("convert", n / (optTs.length + 1));
    parts.push(trim(await vc(audioPad.subarray(s, t + tPad2 + WINDOW), sl(pitch, Math.floor(s / WINDOW), Math.floor((t + tPad2) / WINDOW)), sl(pitchf, Math.floor(s / WINDOW), Math.floor((t + tPad2) / WINDOW)))));
    s = t;
  }
  const t0i = t === null ? 0 : t;
  onProgress("convert", optTs.length / (optTs.length + 1));
  parts.push(trim(await vc(audioPad.subarray(t0i), sl(pitch, Math.floor(t0i / WINDOW)), sl(pitchf, Math.floor(t0i / WINDOW)))));

  let total = 0; for (const x of parts) total += x.length;
  let out = new Float32Array(total); { let o = 0; for (const x of parts) { out.set(x, o); o += x.length; } }

  if (p.rmsMixRate !== 1) out = changeRms(audio, SR, out, tgtSr, p.rmsMixRate);
  let outSr = tgtSr;
  if (tgtSr !== p.resampleSr && p.resampleSr >= 16000) { onProgress("resample", 0); out = resample(out, tgtSr, p.resampleSr); outSr = p.resampleSr; }

  if (p.raw) return { samples: out, sampleRate: outSr, timings };
  let am = 0; for (let i = 0; i < out.length; i++) am = Math.max(am, Math.abs(out[i]));
  am /= 0.99; let scale = 32768; if (am > 1) scale /= am;
  const samples = new Int16Array(out.length);
  for (let i = 0; i < out.length; i++) { const v = Math.trunc(out[i] * scale); samples[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v; }
  onProgress("done", 1);
  return { samples, sampleRate: outSr, timings };
}
