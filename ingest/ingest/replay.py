"""Replay a recorded events JSONL into the hub in real time (or faster), re-stamping `wall`.

Handy for rehearsing graphics without a live stream:
  python -m ingest.replay events.jsonl --udp 127.0.0.1:4101 [--speed 1] [--from-mission -120]

Note: the hub reconciles the clock against wall time, so --speed other than 1 will
make it re-anchor T-0 repeatedly. Use 1 for a faithful rehearsal.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--udp", default="127.0.0.1:4101")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--from-mission", type=float, default=None, help="skip events before this mission time")
    a = ap.parse_args(argv)
    host, port = a.udp.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    events = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    if a.from_mission is not None:
        events = [e for e in events if e.get("mission_t") is None or e["mission_t"] >= a.from_mission]
    start_media, start_wall = events[0]["media_t"], time.monotonic()
    for e in events:
        due = start_wall + (e["media_t"] - start_media) / a.speed
        time.sleep(max(0.0, due - time.monotonic()))
        e["wall"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        sock.sendto(json.dumps(e, separators=(",", ":")).encode(), (host, int(port)))
    print(f"replayed {len(events)} events")


if __name__ == "__main__":
    main()
