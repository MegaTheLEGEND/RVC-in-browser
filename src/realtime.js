// Realtime mode: mic -> 16 kHz blocks -> worker (HuBERT+F0+synth on a sliding window) -> crossfade -> speakers.
import { Stitcher } from "./stitch.js";
let CTX = 24000; const MIN_BLOCK = 4000, XF = 640, W = 160; // 1.5 s context, 0.25 s min block, 40 ms crossfade (16 kHz samples)
const WORKLET = `class C extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor("cap",C)`;

export class Realtime {
  constructor({ worker, getParams, getSr, getCtx, onStat, onError }) { Object.assign(this, { worker, getParams, getSr, getCtx, onStat, onError }); this.running = false; }
  async start() {
    if (this.getCtx) CTX = Math.max(4000, Math.round(this.getCtx() * 16000 / W) * W);
    this.inbuf = new Float32Array(0); this.done = 0; this.busy = false; this.st = new Stitcher(); this.id = 0; this.running = true;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    this.inCtx = new AudioContext({ sampleRate: 16000 });
    await this.inCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
    const src = this.inCtx.createMediaStreamSource(this.stream), node = new AudioWorkletNode(this.inCtx, "cap");
    node.port.onmessage = (e) => { const a = e.data, n = new Float32Array(this.inbuf.length + a.length); n.set(this.inbuf); n.set(a, this.inbuf.length); this.inbuf = n; this.kick(); };
    src.connect(node); this.node = node;
    this.outSr = this.getSr(); this.outCtx = new AudioContext({ sampleRate: this.outSr }); this.next = 0;
  }
  kick() {
    if (!this.running || this.busy) return;
    const fresh = Math.floor((this.inbuf.length - this.done) / W) * W; if (fresh < MIN_BLOCK) return;
    const end = this.done + fresh, len = Math.floor(Math.min(end, CTX + fresh) / W) * W, win = this.inbuf.slice(end - len, end);
    this.busy = true; this.cur = { fresh, t: performance.now(), id: ++this.id }; this.done = end;
    this.worker.postMessage({ type: "rt", id: this.id, audio: win, keep: fresh + XF, ...this.getParams() }, [win.buffer]);
  }
  /** called with the worker's rtOut message */
  onOut(m) {
    this.busy = false; if (!this.running) return;
    const per = (this.outSr / 16000) * W, blockOut = (this.cur.fresh / W) * per, xfOut = (XF / W) * per;
    const pcm = this.st.push(m.samples, blockOut, xfOut);
    const buf = this.outCtx.createBuffer(1, pcm.length, this.outSr); buf.copyToChannel(pcm, 0);
    const s = this.outCtx.createBufferSource(); s.buffer = buf; s.connect(this.outCtx.destination);
    const now = this.outCtx.currentTime, at = Math.max(now + 0.03, this.next); s.start(at); this.next = at + buf.duration;
    this.onStat({ procMs: Math.round(m.ms), blockMs: Math.round(this.cur.fresh / 16), lagMs: Math.round((this.next - now) * 1000), t: m.t, realtime: m.ms < (this.cur.fresh / 16) });
    this.kick();
  }
  stop() { this.running = false; this.stream?.getTracks().forEach((t) => t.stop()); this.node?.disconnect(); this.inCtx?.close(); this.outCtx?.close(); }
}
