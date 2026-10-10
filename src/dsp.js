// Pure-JS DSP primitives used by the RVC browser pipeline.
// Everything here mirrors the numpy/scipy/librosa behaviour used by the Python code
// (verified by tests/test_dsp.mjs against reference values produced with numpy/scipy).

// ---------------------------------------------------------------- FFT
const _twCache = new Map();
function twiddles(n) {
  let t = _twCache.get(n);
  if (!t) {
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = -Math.sin((2 * Math.PI * i) / n); }
    const rev = new Uint32Array(n);
    const bits = Math.log2(n) | 0;
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); rev[i] = r; }
    t = { cos, sin, rev };
    _twCache.set(n, t);
  }
  return t;
}

/** In-place radix-2 complex FFT (forward). n must be a power of two. */
export function fft(re, im) {
  const n = re.length;
  const { cos, sin, rev } = twiddles(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0, tw = 0; k < half; k++, tw += step) {
        const a = start + k, b = a + half;
        const tr = re[b] * cos[tw] - im[b] * sin[tw];
        const ti = re[b] * sin[tw] + im[b] * cos[tw];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

// ---------------------------------------------------------------- padding helpers
/** numpy.pad(mode='reflect') for 1-D arrays (no edge repeat). */
export function padReflect(x, left, right) {
  const n = x.length;
  const out = new (x.constructor)(n + left + right);
  out.set(x, left);
  const refl = (i) => { // index into x for position i (can be out of range)
    const period = 2 * (n - 1);
    if (n === 1) return 0;
    let m = ((i % period) + period) % period;
    return m < n ? m : period - m;
  };
  for (let i = 0; i < left; i++) out[i] = x[refl(i - left)];
  for (let i = 0; i < right; i++) out[left + n + i] = x[refl(n + i)];
  return out;
}

// ---------------------------------------------------------------- STFT / mel (RMVPE front-end)
export function hannPeriodic(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** torch.stft(center=True, pad_mode='reflect') magnitude. Returns {data, nBins, nFrames}, data[bin*nFrames+frame]. */
export function stftMagnitude(audio, nFft = 1024, hop = 160, winLength = 1024) {
  const x = padReflect(audio, nFft >> 1, nFft >> 1);
  const nFrames = 1 + Math.floor((x.length - nFft) / hop);
  const nBins = (nFft >> 1) + 1;
  const win = hannPeriodic(winLength);
  const off = (nFft - winLength) >> 1; // torch centres a shorter window inside n_fft
  const out = new Float32Array(nBins * nFrames);
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  for (let f = 0; f < nFrames; f++) {
    re.fill(0); im.fill(0);
    const s = f * hop;
    for (let i = 0; i < winLength; i++) re[off + i] = x[s + off + i] * win[i];
    fft(re, im);
    for (let k = 0; k < nBins; k++) out[k * nFrames + f] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
  }
  return { data: out, nBins, nFrames };
}

const hz2melHtk = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hzHtk = (m) => 700 * (10 ** (m / 2595) - 1);

/** librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True) with norm='slaney'. Returns Float32Array [nMels*nBins]. */
export function melFilterbank(sr, nFft, nMels, fmin, fmax) {
  const nBins = (nFft >> 1) + 1;
  const fftFreqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * sr) / nFft;
  const mmin = hz2melHtk(fmin), mmax = hz2melHtk(fmax);
  const melF = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) melF[i] = mel2hzHtk(mmin + ((mmax - mmin) * i) / (nMels + 1));
  const w = new Float32Array(nMels * nBins);
  for (let i = 0; i < nMels; i++) {
    const enorm = 2 / (melF[i + 2] - melF[i]);
    for (let k = 0; k < nBins; k++) {
      const lower = (fftFreqs[k] - melF[i]) / (melF[i + 1] - melF[i]);
      const upper = (melF[i + 2] - fftFreqs[k]) / (melF[i + 2] - melF[i + 1]);
      w[i * nBins + k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
  }
  return w;
}

let _rmvpeBasis = null;
/** RMVPE log-mel: MelSpectrogram(128, 16000, win 1024, hop 160, fmin 30, fmax 8000, clamp 1e-5). Output [128 * nFrames]. */
export function rmvpeLogMel(audio16k) {
  if (!_rmvpeBasis) _rmvpeBasis = melFilterbank(16000, 1024, 128, 30, 8000);
  const { data, nBins, nFrames } = stftMagnitude(audio16k, 1024, 160, 1024);
  const out = new Float32Array(128 * nFrames);
  for (let m = 0; m < 128; m++) {
    const row = _rmvpeBasis.subarray(m * nBins, (m + 1) * nBins);
    // sparse: find nonzero range once per row
    let lo = 0, hi = nBins - 1;
    while (lo < nBins && row[lo] === 0) lo++;
    while (hi > lo && row[hi] === 0) hi--;
    for (let f = 0; f < nFrames; f++) {
      let acc = 0;
      for (let k = lo; k <= hi; k++) acc += row[k] * data[k * nFrames + f];
      out[m * nFrames + f] = Math.log(Math.max(acc, 1e-5));
    }
  }
  return { data: out, nFrames };
}

// ---------------------------------------------------------------- Butterworth + filtfilt (scipy-compatible)
/** scipy.signal.butter(N, Wn, 'high', fs=fs) -> {b, a} (float64). Bilinear transform with prewarp. */
export function butterHighpass(order, fc, fs) {
  const fsd = 2; // scipy uses fs=2 normalisation internally for digital design
  const wn = (2 * fc) / fs; // normalised 0..1
  const warped = 2 * fsd * Math.tan((Math.PI * wn) / fsd);
  // analog lowpass prototype poles (unit cutoff)
  let poles = [];
  for (let m = -order + 1; m < order; m += 2) {
    const ang = (Math.PI * m) / (2 * order);
    poles.push([-Math.cos(ang), -Math.sin(ang)]); // -exp(j*pi*m/(2N))... conj-symmetric set
  }
  // lowpass -> highpass: s -> wo/s ; poles p -> wo/p ; zeros at 0 (order of them); gain k = k * prod(-z)/prod(-p) with z=0 => k/prod(-p)
  const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  let prodNegP = [1, 0];
  for (const p of poles) prodNegP = cmul(prodNegP, [-p[0], -p[1]]);
  const kHp = cdiv([1, 0], prodNegP)[0]; // real
  const polesHp = poles.map((p) => cdiv([warped, 0], p));
  // bilinear transform (fs2 = 2*fsd)
  const fs2 = 2 * fsd;
  const zerosA = new Array(order).fill(0); // analog zeros at 0
  const zD = zerosA.map(() => [(fs2 + 0) / (fs2 - 0), 0]); // -> 1
  const pD = polesHp.map((p) => cdiv([fs2 + p[0], p[1]], [fs2 - p[0], -p[1]]));
  // degree difference = 0 for highpass (zeros == poles count)
  let kD = kHp;
  let num = [1, 0], den = [1, 0];
  // gain: k * prod(fs2 - z) / prod(fs2 - p)
  num = [1, 0]; den = [1, 0];
  for (const _ of zerosA) num = cmul(num, [fs2, 0]);
  for (const p of polesHp) den = cmul(den, [fs2 - p[0], -p[1]]);
  const kz = cdiv(num, den);
  kD = kHp * kz[0];
  const poly = (roots) => { // coefficients of prod (x - r), real part
    let c = [[1, 0]];
    for (const r of roots) {
      const nc = new Array(c.length + 1).fill(null).map(() => [0, 0]);
      for (let i = 0; i < c.length; i++) {
        nc[i][0] += c[i][0]; nc[i][1] += c[i][1];
        const t = cmul(c[i], r);
        nc[i + 1][0] -= t[0]; nc[i + 1][1] -= t[1];
      }
      c = nc;
    }
    return c.map((v) => v[0]);
  };
  const b = poly(zD).map((v) => v * kD);
  const a = poly(pD);
  return { b, a };
}

/** scipy.signal.lfilter(b, a, x, zi=...) in float64, direct form II transposed. Returns y; zf in-place into zi if provided. */
export function lfilter(b, a, x, zi) {
  const n = Math.max(a.length, b.length);
  const bb = new Float64Array(n), aa = new Float64Array(n);
  const a0 = a[0];
  for (let i = 0; i < b.length; i++) bb[i] = b[i] / a0;
  for (let i = 0; i < a.length; i++) aa[i] = a[i] / a0;
  const z = new Float64Array(n - 1);
  if (zi) z.set(zi);
  const y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = bb[0] * xi + (n > 1 ? z[0] : 0);
    for (let j = 0; j < n - 2; j++) z[j] = z[j + 1] + bb[j + 1] * xi - aa[j + 1] * yi;
    if (n > 1) z[n - 2] = bb[n - 1] * xi - aa[n - 1] * yi;
    y[i] = yi;
  }
  return y;
}

function solveLinear(A, B) { // Gaussian elimination with partial pivoting; A is n x n (array of arrays)
  const n = B.length;
  const M = A.map((r, i) => [...r, B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** scipy.signal.lfilter_zi */
export function lfilterZi(b, a) {
  const n = Math.max(a.length, b.length);
  const bb = Array.from({ length: n }, (_, i) => (i < b.length ? b[i] / a[0] : 0));
  const aa = Array.from({ length: n }, (_, i) => (i < a.length ? a[i] / a[0] : 0));
  const m = n - 1;
  // IminusA = I - companion(a).T ; companion first row = -a[1:], subdiagonal ones
  const comp = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let j = 0; j < m; j++) comp[0][j] = -aa[j + 1];
  for (let i = 1; i < m; i++) comp[i][i - 1] = 1;
  const IminusA = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => (i === j ? 1 : 0) - comp[j][i]));
  const B = Array.from({ length: m }, (_, i) => bb[i + 1] - aa[i + 1] * bb[0]);
  return solveLinear(IminusA, B);
}

/** scipy.signal.filtfilt(b, a, x) with default padtype='odd', padlen=3*max(len(a),len(b)). Returns Float64Array. */
export function filtfilt(b, a, x) {
  const padlen = 3 * Math.max(a.length, b.length);
  const n = x.length;
  const ext = new Float64Array(n + 2 * padlen);
  for (let i = 0; i < padlen; i++) ext[i] = 2 * x[0] - x[padlen - i];
  for (let i = 0; i < n; i++) ext[padlen + i] = x[i];
  for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  const zi = lfilterZi(b, a);
  let y = lfilter(b, a, ext, zi.map((v) => v * ext[0]));
  y.reverse();
  y = lfilter(b, a, y, zi.map((v) => v * y[0]));
  y.reverse();
  return y.subarray(padlen, padlen + n);
}

// ---------------------------------------------------------------- resampling (windowed-sinc polyphase)
function gcd(a, b) { while (b) [a, b] = [b, a % b]; return a; }
function bessel0(x) { let s = 1, t = 1; for (let k = 1; k < 50; k++) { t *= (x / (2 * k)) ** 2; s += t; if (t < 1e-12 * s) break; } return s; }

/** Band-limited resampler (Kaiser-windowed sinc, rational ratio). Quality comparable to librosa 'kaiser_best'-ish; not bit-identical to soxr. */
export function resample(x, srIn, srOut, { halfWidth = 24, beta = 8.6, rolloff = 0.945 } = {}) {
  if (srIn === srOut) return x instanceof Float32Array ? x : Float32Array.from(x);
  const g = gcd(srIn, srOut);
  const up = srOut / g, down = srIn / g;
  const cutoff = Math.min(1, up / down) * rolloff; // relative to input Nyquist after upsample by `up`
  const scale = Math.min(1, up / down);
  const taps = Math.ceil(halfWidth / scale); // taps on each side, in input samples
  const outLen = Math.ceil((x.length * up) / down);
  const out = new Float32Array(outLen);
  const i0 = 1 / bessel0(beta);
  const kernel = (t) => { // t in input-sample units
    const u = t / taps;
    if (u <= -1 || u >= 1) return 0;
    const arg = Math.PI * cutoff * t;
    const s = t === 0 ? cutoff : Math.sin(arg) / (Math.PI * t);
    return s * bessel0(beta * Math.sqrt(1 - u * u)) * i0;
  };
  // precompute kernels per output phase
  const phases = new Array(up);
  for (let p = 0; p < up; p++) {
    const frac = (p * down) % up; // output sample p sits at input position (p*down)/up
    const pos0 = Math.floor((p * down) / up);
    const fr = ((p * down) % up) / up;
    const w = new Float64Array(2 * taps);
    let sum = 0;
    for (let k = 0; k < 2 * taps; k++) { const idx = k - taps + 1; w[k] = kernel(idx - fr); sum += w[k]; }
    for (let k = 0; k < 2 * taps; k++) w[k] /= sum; // unity DC gain
    phases[p] = { w, pos0 };
  }
  const n = x.length;
  for (let o = 0; o < outLen; o++) {
    const cycle = Math.floor(o / up), p = o % up;
    const { w, pos0 } = phases[p];
    const base = cycle * down + pos0 - taps + 1;
    let acc = 0;
    if (base >= 0 && base + 2 * taps <= n) { for (let k = 0; k < 2 * taps; k++) acc += w[k] * x[base + k]; }
    else { for (let k = 0; k < 2 * taps; k++) { const i = base + k; if (i >= 0 && i < n) acc += w[k] * x[i]; } }
    out[o] = acc;
  }
  return out;
}

// ---------------------------------------------------------------- misc numerics
/** torch F.interpolate(mode='linear', align_corners=False) on 1-D data. */
export function interpLinear(src, outSize) {
  const n = src.length, out = new Float32Array(outSize);
  const scale = n / outSize;
  for (let i = 0; i < outSize; i++) {
    let s = (i + 0.5) * scale - 0.5; if (s < 0) s = 0;
    const i0 = Math.min(Math.floor(s), n - 1), i1 = Math.min(i0 + 1, n - 1), f = s - i0;
    out[i] = src[i0] * (1 - f) + src[i1] * f;
  }
  return out;
}

/** librosa.feature.rms(y, frame_length, hop_length, center=True, pad_mode='constant') */
export function rmsFrames(y, frameLength, hop) {
  const pad = frameLength >> 1;
  const nFrames = 1 + Math.floor(y.length / hop);
  const out = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    let s = 0;
    const start = f * hop - pad;
    for (let i = 0; i < frameLength; i++) { const j = start + i; if (j >= 0 && j < y.length) s += y[j] * y[j]; }
    out[f] = Math.sqrt(s / frameLength);
  }
  return out;
}

/** infer/vc/pipeline.py change_rms: scale data2 (output) so its loudness envelope follows data1 (input). Mutates data2. */
export function changeRms(data1, sr1, data2, sr2, rate) {
  const r1 = interpLinear(rmsFrames(data1, (sr1 >> 1) * 2, sr1 >> 1), data2.length);
  const r2 = interpLinear(rmsFrames(data2, (sr2 >> 1) * 2, sr2 >> 1), data2.length);
  for (let i = 0; i < data2.length; i++) data2[i] *= Math.pow(r1[i], 1 - rate) * Math.pow(Math.max(r2[i], 1e-6), rate - 1);
  return data2;
}

/** Seeded Gaussian generator (xorshift128 + Box-Muller). */
export class Rng {
  constructor(seed = 0x9e3779b9) { this.s = [seed >>> 0 || 1, 0x243f6a88, 0xb7e15162, 0xdeadbeef]; for (let i = 0; i < 16; i++) this.u32(); this.spare = null; }
  u32() { let t = this.s[3]; const s = this.s[0]; this.s[3] = this.s[2]; this.s[2] = this.s[1]; this.s[1] = s; t ^= t << 11; t ^= t >>> 8; this.s[0] = (t ^ s ^ (s >>> 19)) >>> 0; return this.s[0]; }
  uniform() { return (this.u32() + 0.5) / 4294967296; }
  normal() { if (this.spare !== null) { const v = this.spare; this.spare = null; return v; } const u = this.uniform(), v = this.uniform(); const r = Math.sqrt(-2 * Math.log(u)); this.spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v); }
  fillNormal(arr, std = 1) { for (let i = 0; i < arr.length; i++) arr[i] = this.normal() * std; return arr; }
}
