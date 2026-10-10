// Central place for URLs / tunables. Edit ORT_BASE to point at a vendored copy (see README) for offline use.
export const ORT_VERSION = "1.22.0"; // pinned (>=2 weeks old); bump deliberately
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;
// Module file names inside ORT_BASE. Verify they exist for the version you pin.
export const ORT_MODULE_WASM = "ort.wasm.min.mjs";
export const ORT_MODULE_WEBGPU = "ort.webgpu.min.mjs";
// Chunking presets from configs/config.py (non-half / low-VRAM preset; seconds)
export const CHUNK = { xPad: 1, xQuery: 6, xCenter: 38, xMax: 41 };
