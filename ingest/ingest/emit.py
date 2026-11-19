"""Event envelope + sinks. PROVISIONAL contract v0.1 until the runtime defines the shared one.

Every event is one JSON object per line:
  {"v": "0.1", "seq": 12, "type": "milestone", "source": "ocr-ingest",
   "wall": "2026-11-17T23:14:05.120Z", "media_t": 812.5, "mission_t": 148.2, "data": {...}}
"""
from __future__ import annotations

import json
import socket
import sys
import urllib.request
from datetime import datetime, timezone

CONTRACT_VERSION = "0.1"


class Emitter:
    def __init__(self, jsonl_path: str | None = None, udp: str | None = None, http: str | None = None,
                 stdout: bool = True, quiet_types: set[str] | None = None):
        self.seq = 0
        self.f = open(jsonl_path, "a") if jsonl_path else None
        self.udp = None
        if udp:
            host, port = udp.rsplit(":", 1)
            self.udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
        self.http = http
        self.stdout = stdout
        self.quiet = quiet_types or set()

    def emit(self, type_: str, data: dict, media_t: float | None, mission_t: float | None) -> dict:
        self.seq += 1
        ev = {"v": CONTRACT_VERSION, "seq": self.seq, "type": type_, "source": "ocr-ingest",
              "wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
              "media_t": None if media_t is None else round(media_t, 2),
              "mission_t": None if mission_t is None else round(mission_t, 2), "data": data}
        line = json.dumps(ev, separators=(",", ":"))
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        if self.udp:
            self.udp[0].sendto(line.encode(), self.udp[1])
        if self.http:
            try:
                req = urllib.request.Request(self.http, data=line.encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=0.5).close()
            except Exception as e:  # never let the sink stall ingest
                print(f"[emit] http sink error: {e}", file=sys.stderr)
        if self.stdout and type_ not in self.quiet:
            print(line, flush=True)
        return ev
