#!/usr/bin/env python3
"""Static server for the RVC browser app. Adds COOP/COEP headers so onnxruntime-web can use multi-threaded WASM.
Usage:  python web/serve.py [port]     then open http://localhost:8000
(Opening index.html via file:// will NOT work: ES modules and workers need http.)"""
import sys, os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class H(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".onnx": "application/octet-stream"}
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://localhost:{port}"); ThreadingHTTPServer(("", port), H).serve_forever()
