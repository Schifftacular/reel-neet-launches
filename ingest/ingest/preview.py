"""Live preview: the frame the ingest just read, with every OCR box, its raw text, the
parsed number, which vehicle each slot is routed to, the locked clock and recent events.

Served as an MJPEG page on http://127.0.0.1:<port>/ so it works with opencv-python-headless
(no GUI window needed). Open the URL in any browser while `ingest.run --preview` runs.
"""
from __future__ import annotations

import collections
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>OCR ingest preview</title>
<style>body{margin:0;background:#111;color:#eee;font:14px system-ui}img{width:100%;height:auto;display:block}
p{margin:6px 10px;opacity:.7}</style></head><body><img src="/stream"><p>Live OCR ingest preview. Green box = read OK, amber = read but rejected, red = nothing read.</p></body></html>"""

COL = {"ok": (80, 220, 80), "rej": (0, 180, 255), "none": (60, 60, 230), "text": (255, 200, 80)}


class Preview:
    def __init__(self, port: int = 8090, width: int = 1280):
        self.width = width
        self.jpeg: bytes | None = None
        self.cond = threading.Condition()
        self.events: collections.deque = collections.deque(maxlen=8)
        prev = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            with prev.cond:
                                prev.cond.wait(timeout=5)
                                buf = prev.jpeg
                            if buf is None:
                                continue
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf))
                            self.wfile.write(buf + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        print(f"[preview] open http://127.0.0.1:{port}/", flush=True)

    def note(self, typ: str, data: dict, mt: float | None) -> None:
        if typ in ("milestone", "milestone.confirm"):
            s = f"{typ.upper()} {data.get('id')} t={data.get('t')} {data.get('status', '')}"
        elif typ == "layout.change":
            s = f"LAYOUT {data.get('slot')} -> {data.get('vehicle')} ({data.get('by')})"
        elif typ in ("clock.lock", "clock.state", "clock.jump"):
            s = f"{typ.upper()} " + " ".join(f"{k}={v}" for k, v in data.items())
        else:
            return
        clock = "" if mt is None else f"T{'-' if mt < 0 else '+'}{int(abs(mt)) // 60:02d}:{int(abs(mt)) % 60:02d}  "
        self.events.appendleft(clock + s)

    def draw(self, frame: np.ndarray, regions, raw: dict, parsed: dict, slot_of: dict, header: list[str],
             telemetry: dict) -> None:
        img = frame.copy()
        H, W = img.shape[:2]
        for r in regions:
            x0, y0 = int(r.x * W), int(r.y * H)
            x1, y1 = int((r.x + r.w) * W), int((r.y + r.h) * H)
            text, conf = raw.get(r.name, ("", 0.0))
            val = parsed.get(r.name, "n/a")
            if r.kind == "text":
                col = COL["text"]
            elif not text.strip():
                col = COL["none"]
            elif val is None:
                col = COL["rej"]
            else:
                col = COL["ok"]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            veh = slot_of.get(r.name)
            label = f"{text.strip()[:14]!r} c{conf:.2f}"
            if r.kind == "number":
                label += f" -> {val}" + (f" [{veh}]" if veh else "")
            _label(img, label, (x0, y0 - 8), col)
        # status panel, top-left
        lines = header + [f"{k}: {v}" for k, v in telemetry.items()] + ["", "recent:"] + list(self.events)
        pad, lh = 12, 30
        cv2.rectangle(img, (0, 0), (int(W * 0.42), pad * 2 + lh * len(lines)), (0, 0, 0), -1)
        for i, s in enumerate(lines):
            cv2.putText(img, s, (pad, pad + lh * (i + 1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2,
                        cv2.LINE_AA)
        if self.width and W > self.width:
            img = cv2.resize(img, (self.width, int(H * self.width / W)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()


def _label(img, s, org, col):
    x, y = org
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(img, s, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
