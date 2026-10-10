/** Decode any browser-supported audio file to mono Float32 at `sr` (default 16 kHz) using the browser's own resampler. */
export async function decodeToMono(arrayBuffer, sr = 16000) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr });
  try {
    const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const n = buf.length, out = new Float32Array(n), ch = buf.numberOfChannels;
    for (let c = 0; c < ch; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
    return out;
  } finally { ctx.close(); }
}

/** 16-bit PCM mono WAV from Int16Array. */
export function encodeWav(int16, sampleRate) {
  const buf = new ArrayBuffer(44 + int16.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + int16.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, int16.length * 2, true);
  new Int16Array(buf, 44).set(int16);
  return new Blob([buf], { type: "audio/wav" });
}
