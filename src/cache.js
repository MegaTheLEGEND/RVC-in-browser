// Tiny IndexedDB blob cache so multi-hundred-MB models are picked once, not every visit.
const DB = "rvc-web", STORE = "models";
const open = () => new Promise((res, rej) => { const r = indexedDB.open(DB, 1); r.onupgradeneeded = () => r.result.createObjectStore(STORE); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
const tx = async (mode, fn) => { const db = await open(); return new Promise((res, rej) => { const t = db.transaction(STORE, mode), q = fn(t.objectStore(STORE)); t.oncomplete = () => res(q && q.result); t.onerror = () => rej(t.error); }); };
export const cachePut = (key, value) => tx("readwrite", (s) => s.put(value, key)).catch((e) => console.warn("cache put failed", e));
export const cacheGet = (key) => tx("readonly", (s) => s.get(key));
export const cacheKeys = () => tx("readonly", (s) => s.getAllKeys());
export const cacheDelete = (key) => tx("readwrite", (s) => s.delete(key));
