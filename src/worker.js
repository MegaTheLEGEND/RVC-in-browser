// Inference worker: owns onnxruntime-web sessions so the UI thread never blocks.
import { loadOrt, createSession } from "./ort.js";
import { Hubert } from "./hubert.js";
import { Rmvpe } from "./rmvpe.js";
import { Yin } from "./yin.js";
import { Synth } from "./synth.js";
import { FaissIndex } from "./faiss.js";
import { loadTorch } from "./torchpkl.js";
import { buildSynth } from "./build_synth.js";
import { buildHubert } from "./build_hubert.js";
import { buildRmvpe } from "./build_rmvpe.js";
import { convert } from "./pipeline.js";

const S = { ort: null, flavor: "wasm", hubert: {}, rmvpe: null, synth: null, index: null };
// HuBERT built in JS from HF weights (pytorch_model.bin / .safetensors); built lazily per version (v1 = layer 9 + final_proj, v2 = layer 12)
async function ensureHubert(v) {
  if (S.hubert[v] || !S.hubertCk) return;
  status(`Building HuBERT ${v} graph in JS…`);
  const { onnx } = buildHubert(S.hubertCk, v, S.hubertCfg || {});
  S.hubert[v] = new Hubert(S.ort, await createSession(S.ort, onnx, { webgpu: S.flavor === "webgpu", name: "hubert" }));
  post({ type: "loaded", kind: "hubert", version: v });
}
const post = (m, t) => self.postMessage(m, t || []);
const status = (text) => post({ type: "status", text });

// Messages must run one at a time: ORT-web throws "multiple calls to initWasm()" if two sessions are created concurrently.
let _q = Promise.resolve();
self.onmessage = (ev) => { _q = _q.then(() => handle(ev)); };
async function handle(ev) {
  const m = ev.data;
  try {
    if (m.type === "init") {
      S.flavor = m.webgpu ? "webgpu" : "wasm";
      status(`Loading onnxruntime-web (${S.flavor})…`);
      S.ort = await loadOrt(S.flavor);
      status(`onnxruntime-web ready (threads=${S.ort.env.wasm.numThreads}${crossOriginIsolated ? "" : ", not cross-origin-isolated → single thread"})`);
      post({ type: "ready" });
    } else if (m.type === "load") {
      if (!S.ort) throw new Error("runtime not initialised");
      const useGpu = S.flavor === "webgpu";
      if (m.kind === "hubertWeights") {
        status(`Reading HuBERT weights (${(m.bytes.byteLength / 1e6).toFixed(0)} MB)…`);
        const ck = await loadTorch(m.bytes); S.hubertCk = ck.weight && !ck.weight.byteLength ? ck.weight : ck; S.hubertCfg = m.config || null; S.hubert = {};
        const hasV1 = Object.keys(S.hubertCk).some((k) => k.endsWith("final_proj.weight"));
        if (S.synth) await ensureHubert(S.synth.meta.version);
        else post({ type: "loaded", kind: "hubert", version: hasV1 ? "v1+v2 (built on demand)" : "v2 (built on demand)" });
      } else if (m.kind === "hubert") {
        status(`Loading HuBERT ${m.version} (${(m.bytes.byteLength / 1e6).toFixed(0)} MB)…`);
        const hb = new Hubert(S.ort, await createSession(S.ort, new Uint8Array(m.bytes), { webgpu: useGpu, name: "hubert" }));
        const probe = await hb.run(new Float32Array(8000)); const ver = probe.C === 256 ? "v1" : "v2"; // 256-d = layer 9 + final_proj (v1), 768-d = v2; trust the model, not the file name
        S.hubert[ver] = hb; m.version = ver;
      } else if (m.kind === "rmvpeWeights") {
        status("Reading rmvpe.pt and building the graph in JS…");
        const { onnx } = buildRmvpe(await loadTorch(m.bytes));
        S.rmvpe = new Rmvpe(S.ort, await createSession(S.ort, onnx, { webgpu: false, name: "rmvpe" }));
      } else if (m.kind === "rmvpe") {
        status("Loading RMVPE…");
        S.rmvpe = new Rmvpe(S.ort, await createSession(S.ort, new Uint8Array(m.bytes), { webgpu: false, name: "rmvpe" }));
      } else if (m.kind === "voice") {
        status("Reading .pth and building the voice graph in JS…");
        const ck = await loadTorch(m.bytes);
        const { onnx, meta } = buildSynth(ck);
        const sess = await createSession(S.ort, onnx, { webgpu: false, name: "voice" });
        S.synth = new Synth(S.ort, sess, meta);
        S.index = null; await ensureHubert(meta.version);
        if (m.indexBytes) { status("Reading faiss index…"); S.index = new FaissIndex(m.indexBytes); }
        post({ type: "voiceInfo", meta, index: !!S.index });
      }
      post({ type: "loaded", kind: m.kind === "rmvpeWeights" ? "rmvpe" : m.kind, version: m.version });
    } else if (m.type === "rt") {
      const { synth } = S; if (synth) await ensureHubert(synth.meta.version); const hub = synth && S.hubert[synth.meta.version]; if (!hub) throw new Error(!synth ? "load a voice (.pth) first" : `no HuBERT for this ${synth.meta.version} voice: pick pytorch_model.bin/.safetensors or a hubert onnx under "HuBERT" (the runtime switch clears loaded models)`);
      let f0 = null; if (synth.ifF0) f0 = m.f0Method === "rmvpe" && S.rmvpe ? S.rmvpe : new Yin();
      const t0 = performance.now();
      const r = await convert(m.audio, { hubert: hub, f0, synth, index: S.index }, { ...m.params, keep: m.keep, xPad: 0.1, raw: true, rmsMixRate: 1, resampleSr: 0 }, () => {});
      post({ type: "rtOut", id: m.id, samples: r.samples, sampleRate: r.sampleRate, ms: performance.now() - t0, t: r.timings }, [r.samples.buffer]);
    } else if (m.type === "convert") {
      const { synth } = S; if (!synth) throw new Error("load a voice model first");
      const hub = S.hubert[synth.meta.version]; if (!hub) throw new Error(`this voice needs HuBERT ${synth.meta.version} (hubert_${synth.meta.version}.onnx)`);
      let f0 = null;
      if (synth.ifF0) { if (m.f0Method === "rmvpe") { if (S.rmvpe) f0 = S.rmvpe; else { status("No RMVPE loaded — using YIN pitch instead."); f0 = new Yin(); } } else f0 = new Yin(); }
      const t0 = performance.now();
      const r = await convert(m.audio, { hubert: hub, f0, synth, index: S.index }, m.params, (stage, frac) => post({ type: "progress", stage, frac }));
      status(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s — HuBERT ${(r.timings.hubert / 1000).toFixed(1)} s · F0 ${(r.timings.f0 / 1000).toFixed(1)} s · synth ${(r.timings.synth / 1000).toFixed(1)} s${S.index && m.params.indexRate ? " · index on" : ""}`);
      post({ type: "result", samples: r.samples, sampleRate: r.sampleRate }, [r.samples.buffer]);
    }
  } catch (e) { post({ type: "error", message: String((e && e.message) || e) }); console.error(e); }
}
