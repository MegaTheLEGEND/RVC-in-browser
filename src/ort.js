import { ORT_BASE, ORT_MODULE_WASM, ORT_MODULE_WEBGPU } from "./config.js";

let _ort = null, _flavor = null;
/** Lazily import onnxruntime-web (ESM) inside the worker. flavor: 'wasm' | 'webgpu' */
async function pickBase(file) {
  // Prefer a vendored copy (web/vendor/ort) — same-origin, works offline, no CDN worker/CORS quirks.
  const local = new URL("../vendor/ort", import.meta.url).href;
  try { const r = await fetch(`${local}/${file}`, { method: "HEAD" }); if (r.ok) return local; } catch (_) {}
  return ORT_BASE;
}

export async function loadOrt(flavor = "wasm", base = null) {
  if (_ort && _flavor === flavor) return _ort;
  const file = flavor === "webgpu" ? ORT_MODULE_WEBGPU : ORT_MODULE_WASM;
  base = base || (await pickBase(file));
  const mod = await import(/* @vite-ignore */ `${base}/${file}`);
  const ort = mod.default && mod.default.InferenceSession ? mod.default : mod;
  ort.env.wasm.wasmPaths = `${base}/`;
  const iso = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  ort.env.wasm.numThreads = iso ? Math.max(1, (navigator.hardwareConcurrency || 2) - 1) : 1;
  _ort = ort; _flavor = flavor;
  return ort;
}

/** Create a session from an ArrayBuffer/Uint8Array, preferring WebGPU when requested and falling back to WASM. */
export async function createSession(ort, bytes, { webgpu = false, name = "model" } = {}) {
  const eps = webgpu ? ["webgpu", "wasm"] : ["wasm"];
  try {
    return await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
  } catch (e) {
    if (webgpu) { console.warn(`[${name}] WebGPU session failed (${e}); falling back to WASM`); return await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
    throw e;
  }
}
