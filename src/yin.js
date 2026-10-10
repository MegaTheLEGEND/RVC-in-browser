// Lightweight model-free pitch tracker (YIN, de Cheveigné & Kawahara 2002) used as the
// "no model needed" f0 option. Quality is below RMVPE; intended for quick tests / fallback.
export class Yin {
  constructor({ sr = 16000, hop = 160, fmin = 50, fmax = 1100, threshold = 0.15 } = {}) { Object.assign(this, { sr, hop, fmin, fmax, threshold }); }
  async infer(x) {
    const { sr, hop, fmin, fmax, threshold } = this;
    const tauMin = Math.max(2, Math.floor(sr / fmax)), tauMax = Math.ceil(sr / fmin);
    const W = tauMax * 2; // integration window
    const nFrames = 1 + Math.floor(x.length / hop);
    const f0 = new Float32Array(nFrames);
    const d = new Float64Array(tauMax + 2), cm = new Float64Array(tauMax + 2);
    for (let f = 0; f < nFrames; f++) {
      const c = f * hop, s = c - (W >> 1);
      // silence gate
      let e = 0; for (let i = 0; i < W; i++) { const j = s + i; const v = j >= 0 && j < x.length ? x[j] : 0; e += v * v; }
      if (e / W < 1e-6) continue;
      for (let tau = 1; tau <= tauMax + 1; tau++) {
        let sum = 0;
        for (let i = 0; i < W; i++) { const j = s + i, k = j + tau; if (j < 0 || k >= x.length) continue; const df = x[j] - x[k]; sum += df * df; }
        d[tau] = sum;
      }
      cm[1] = 1; let run = 0;
      for (let tau = 1; tau <= tauMax + 1; tau++) { run += d[tau]; cm[tau] = run > 0 ? (d[tau] * tau) / run : 1; }
      let tau = -1;
      for (let t = tauMin; t <= tauMax; t++) { if (cm[t] < threshold) { while (t + 1 <= tauMax && cm[t + 1] < cm[t]) t++; tau = t; break; } }
      if (tau < 0) continue;
      const a = cm[tau - 1], b = cm[tau], g = cm[tau + 1];
      const den = a + g - 2 * b; const shift = den !== 0 ? (a - g) / (2 * den) : 0;
      f0[f] = sr / (tau + shift);
    }
    return f0;
  }
}
