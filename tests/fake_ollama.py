"""Tiny in-process fake of the Ollama HTTP API for integration tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from literature_buddy.retrieval.embeddings import HashingEmbedder

_EMB = HashingEmbedder()


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._json({"models": [{"name": "fake:latest"}]})
                else:
                    self._json({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, **body})
                if self.path == "/api/embed":
                    vecs = [_EMB.embed_query(t).tolist() for t in body["input"]]
                    return self._json({"embeddings": vecs})
                if self.path == "/api/chat":
                    msgs = body["messages"]
                    last = msgs[-1]
                    if last.get("images"):
                        reply = "The chart shows six bars; the blue bars are higher than the grey ones."
                    elif "Standalone query" in last["content"]:
                        reply = "Which dataset and how many samples were used for training"
                    else:
                        reply = "The model was trained on 12,450 samples from the TCGA cohort [S1]. Inference: it is large."
                        if "Figure 1" in last["content"].split("QUESTION:")[-1] and "MODEL-GENERATED" in last["content"]:
                            reply = "**Paper states:** Figure 1 compares Dice scores [S1]. **Inference:** (figure reading) blue bars are higher."
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for i in range(0, len(reply), 10):
                        self.wfile.write((json.dumps({"message": {"content": reply[i:i + 10]}}) + "\n").encode())
                        self.wfile.flush()
                    self.wfile.write((json.dumps({"done": True}) + "\n").encode())
                    return
                self._json({}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
