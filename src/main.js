import { decodeToMono, encodeWav } from "./audio.js";
import { Realtime } from "./realtime.js";
import { cachePut, cacheGet, cacheKeys, cacheDelete } from "./cache.js";

const $ = (id) => document.getElementById(id);
const log = (t, err) => { const l = $("log"), d = document.createElement("div"); if (err) d.className = "err"; d.textContent = t; l.appendChild(d); l.scrollTop = l.scrollHeight; };
const have = { hubert: new Set(), rmvpe: false, voice: null };
let worker = null, ready = null, audio16k = null;

function startWorker() {
  worker?.terminate();
  have.hubert.clear(); have.rmvpe = false; have.voice = null; renderChips();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  ready = new Promise((res) => { worker._res = res; });
  worker.onmessage = onWorker; worker.onerror = (e) => log(`worker error: ${e.message}`, true);
  worker.postMessage({ type: "init", webgpu: $("ep").value === "webgpu" });
}
function onWorker({ data: m }) {
  if (m.type === "ready") worker._res();
  else if (m.type === "status") log(m.text);
  else if (m.type === "voiceInfo") {
    const me = m.meta; have.voice = { version: me.version, f0: me.f0, sr: me.sr, index: m.index };
    $("sid").max = Math.max(0, me.n_spk - 1); $("f0m").disabled = !me.f0; renderChips();
  } else if (m.type === "loaded") { if (m.kind === "hubert") have.hubert.add(m.version); if (m.kind === "rmvpe") have.rmvpe = true; renderChips(); }
  else if (m.type === "rtOut") rt?.onOut(m);
  else if (m.type === "progress") { $("bar").hidden = false; $("bar").value = m.frac; }
  else if (m.type === "result") {
    const blob = encodeWav(m.samples, m.sampleRate), url = URL.createObjectURL(blob);
    $("out-player").src = url; $("out-player").hidden = false; $("dl").href = url; $("dl").hidden = false; $("bar").hidden = true; $("go").disabled = false;
  } else if (m.type === "error") { if (rt?.running) { rt.stop(); $("rt").textContent = "Start realtime"; } log(m.message, true); $("bar").hidden = true; $("go").disabled = !canConvert(); }
}
const canConvert = () => !!(audio16k && have.voice);
function renderChips() {
  const c = [...have.hubert].map((v) => `HuBERT ${v}`); if (have.rmvpe) c.push("RMVPE"); if (have.voice) c.push(`Voice ${have.voice.version}${have.voice.f0 ? "" : " (no f0)"} · ${have.voice.sr / 1000} kHz${have.voice.index ? " · index" : ""}`);
  $("loaded").innerHTML = c.map((t) => `<span class="chip ok">${t}</span>`).join("");
  $("go").disabled = !canConvert();
}

async function sendModel(kind, name, bytes, extra = {}) {
  await ready;
  const copy = bytes.slice(0); // keep the original for the cache; transfer the copy
  worker.postMessage({ type: "load", kind, bytes: copy, ...extra }, [copy]);
}
const hubertVersion = (name) => (/v1/i.test(name) && !/v2/i.test(name) ? "v1" : "v2");

async function loadHubertW(name, bytes, store = true) { if (store) cachePut(`hubertw:${name}`, bytes); const cfg = $("f-hcfg").files[0]; await sendModel("hubertWeights", name, bytes, { config: cfg ? JSON.parse(await cfg.text()) : null }); }
async function loadHubert(name, bytes, store = true) { if (store) cachePut(`hubert:${name}`, bytes); await sendModel("hubert", name, bytes, { version: hubertVersion(name) }); }
async function loadRmvpe(name, bytes, store = true) { const pt = !/\.onnx$/i.test(name); if (store) cachePut(`rmvpe${pt ? "w" : ""}:${name}`, bytes); await sendModel(pt ? "rmvpeWeights" : "rmvpe", name, bytes); }
async function loadVoice(pth, index, store = true) {
  const bytes = await pth.arrayBuffer(), indexBytes = index ? await index.arrayBuffer() : null;
  if (store) cachePut(`voice:${pth.name}`, { bytes, indexBytes, indexName: index && index.name });
  await sendVoice(bytes, indexBytes);
}
async function sendVoice(bytes, indexBytes) {
  await ready;
  if (!have.hubert.size) { const ks = await cacheKeys(); const w = ks.filter((x) => x.startsWith("hubertw:")), o = ks.filter((x) => x.startsWith("hubert:")), list = w.length ? w.slice(0, 1) : o; if (list.length) { for (const k of list) { log(`Loading cached ${k}…`); await loadCached(k); } } else log("No HuBERT loaded yet — add one under HuBERT (pytorch_model.bin / .safetensors / .onnx).", true); } const t = [bytes.slice(0)]; if (indexBytes) t.push(indexBytes.slice(0));
  worker.postMessage({ type: "load", kind: "voice", bytes: t[0], indexBytes: t[1] || null }, t);
}

async function renderCache() {
  const keys = (await cacheKeys().catch(() => [])) || [];
  $("cached").innerHTML = keys.length ? "" : "<span class='chip'>none yet</span>";
  for (const k of keys) {
    const s = document.createElement("span"); s.className = "chip"; s.textContent = k;
    const b = document.createElement("button"); b.textContent = "load"; b.onclick = () => loadCached(k);
    const d = document.createElement("button"); d.textContent = "✕"; d.onclick = async () => { await cacheDelete(k); renderCache(); };
    s.append(b, d); $("cached").appendChild(s);
  }
}
async function loadCached(k) {
  const v = await cacheGet(k), name = k.slice(k.indexOf(":") + 1);
  if (k.startsWith("hubertw:")) await loadHubertW(name, v, false);
  else if (k.startsWith("hubert:")) await loadHubert(name, v, false);
  else if (k.startsWith("rmvpe:") || k.startsWith("rmvpew:")) await loadRmvpe(name, v, false);
  else if (k.startsWith("voice:")) await sendVoice(v.bytes, v.indexBytes);
}

// ---- wiring
$("f-hubert").onchange = async (e) => { const f = e.target.files[0]; if (f && !/\.onnx$/i.test(f.name)) { await loadHubertW(f.name, await f.arrayBuffer()); renderCache(); } else if (f) { await loadHubert(f.name, await f.arrayBuffer()); renderCache(); } };
$("f-rmvpe").onchange = async (e) => { const f = e.target.files[0]; if (f) { await loadRmvpe(f.name, await f.arrayBuffer()); renderCache(); } };
$("f-voice").onchange = async (e) => { const f = e.target.files[0]; if (f) { await loadVoice(f, $("f-index").files[0]); renderCache(); } };
$("f-index").onchange = async (e) => { const p = $("f-voice").files[0]; if (p && e.target.files[0]) await loadVoice(p, e.target.files[0]); };
$("ep").onchange = () => { log("Runtime changed — reload your models."); startWorker(); };
$("f-audio").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const buf = await f.arrayBuffer();
    $("in-player").src = URL.createObjectURL(new Blob([buf])); $("in-player").hidden = false;
    audio16k = await decodeToMono(buf, 16000);
    log(`Audio: ${(audio16k.length / 16000).toFixed(1)} s`); renderChips();
  } catch (err) { log(`Could not decode audio: ${err.message || err}`, true); }
};
for (const [id, o] of [["key", "o-key"], ["idx", "o-idx"], ["prot", "o-prot"], ["rms", "o-rms"]]) $(id).oninput = () => { $(o).textContent = $(id).value; };
$("go").onclick = () => {
  $("go").disabled = true; $("out-player").hidden = true; $("dl").hidden = true; $("bar").value = 0;
  const a = Float32Array.from(audio16k);
  worker.postMessage({
    type: "convert", audio: a, f0Method: $("f0m").value,
    params: { sid: +$("sid").value, f0UpKey: +$("key").value, indexRate: +$("idx").value, protect: +$("prot").value, rmsMixRate: +$("rms").value, resampleSr: +$("rsr").value },
  }, [a.buffer]);
};

let rt = null;
$("rt").onclick = async () => {
  if (rt?.running) { rt.stop(); $("rt").textContent = "Start realtime"; return; }
  if (!have.voice) return log("Load a voice (and HuBERT) first.", true);
  const params = () => ({ f0Method: $("f0m").value, params: { sid: +$("sid").value, f0UpKey: +$("key").value, indexRate: +$("idx").value, protect: +$("prot").value } });
  rt = new Realtime({ worker, getParams: params, getSr: () => have.voice.sr, getCtx: () => +$("rt-ctx").value, onError: (e) => log(String(e), true),
    onStat: (s) => { $("rt-stat").textContent = `processing ${s.procMs} ms / block ${s.blockMs} ms · queued ${s.lagMs} ms ${s.t ? ` (HuBERT ${Math.round(s.t.hubert)} · F0 ${Math.round(s.t.f0)} · synth ${Math.round(s.t.synth)})` : ""} ${s.realtime ? "✓" : "(too slow, blocks growing)"}`; } });
  try { await rt.start(); $("rt").textContent = "Stop realtime"; } catch (e) { log(`Realtime failed: ${e.message || e}`, true); rt = null; }
};
if (!crossOriginIsolated) log("Tip: serve with web/serve.py to enable multi-threaded WASM (cross-origin isolation). Single-threaded still works, just slower.");
startWorker(); renderCache();
