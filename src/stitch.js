/** Overlap-add of consecutive realtime windows. Each window's output is end-aligned with its input; segment k covers
 *  [T_k - block - xf, T_k]; the last `xf` samples are held and cross-faded with the head of the next segment. */
export class Stitcher {
  constructor() { this.hold = null; }
  /** o: window output (end-aligned), blockOut/xfOut in output samples. Returns samples to play (length blockOut, or blockOut-xfOut for the first window). */
  push(o, blockOut, xfOut) {
    const need = blockOut + xfOut;
    if (!this.hold) { // first window: nothing before it
      const n = Math.min(o.length, blockOut); const emit = o.slice(o.length - n, o.length - xfOut);
      this.hold = o.slice(o.length - xfOut); return emit;
    }
    const seg = o.slice(Math.max(0, o.length - need)); // seg.length === need when the window has enough context
    const out = new Float32Array(blockOut);
    for (let i = 0; i < xfOut; i++) { const w = (i + 0.5) / xfOut; out[i] = this.hold[i] * (1 - w) + seg[i] * w; }
    out.set(seg.subarray(xfOut, blockOut), xfOut);
    this.hold = seg.slice(blockOut, blockOut + xfOut); return out;
  }
}
