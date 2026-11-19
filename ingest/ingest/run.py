"""Ingest entry point.

  python -m ingest.run --source ffmpeg --input vod.mp4 --fps 2 --timeline timelines/falcon9_sample.json --jsonl events.jsonl
  python -m ingest.run --source obs --input "SpaceX Webcast" --fps 2 --udp 127.0.0.1:9870
  python -m ingest.run --source images --input "frames/*.png" --timeline ...
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import sources
from .clock import ClockModel
from .emit import Emitter
from .layout import LayoutMapper
from .ocr import Region, parse_clock_loose, parse_field, read_region
from .timeline import TimelineFusion, load
from .vehicle import VehicleTracker, dynamic_pressure_kpa

NUM_CONF = 0.4
WEAK_CONF = 0.1
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Pipeline:
    def __init__(self, profile: dict, emitter: Emitter, timeline: list[dict] | None = None, assume: bool = False):
        self.regions = [Region(**r) for r in profile["regions"]]
        self.layout = LayoutMapper(profile)
        # Vehicle-specific milestone ids, e.g. Starship hot staging: ship ignition (ses1) IS stage_sep.
        self.ms_alias: dict = profile.get("milestone_alias") or {}
        self.ms_drop: set = set(profile.get("milestone_drop") or [])
        self.em = emitter
        self.clock = ClockModel(countdown_floor=profile.get("clock_countdown") == "floor")
        self.veh = VehicleTracker()
        self.fusion = TimelineFusion(timeline, assume) if timeline else None
        self.pool = ThreadPoolExecutor(max_workers=len(self.regions))
        self.last_tick = None
        self.conf_acc = {r.name: [] for r in self.regions}
        self.frames = 0
        self.clock_conf = 0.0
        self.preview = None   # set to a preview.Preview to draw every frame

    def read(self, frame):
        res = list(self.pool.map(lambda r: (r, read_region(frame, r)), self.regions))
        raw = {}
        for r, (text, conf) in res:
            self.conf_acc[r.name].append(conf)
            raw[r.name] = (text, conf)
        return raw

    def process(self, t: float, frame) -> None:
        self.frames += 1
        raw = self.read(frame)
        ctext, cconf = raw.get("clock", ("", 0))
        # No confidence cut-off: on real footage Tesseract scored correct clocks at 0.0-0.35.
        # The parser and the clock model's lock reject junk instead.
        loose = parse_clock_loose(ctext)
        creading = self.clock.resolve(t, *loose) if loose else None
        self.clock_conf = round(cconf, 2) if creading is not None else 0.0
        mt_before = self.clock.mission_time(t)
        for typ, data in self.clock.update(t, creading):
            self.em.emit(typ, data, t, self.clock.mission_time(t))
        mt = self.clock.mission_time(t)

        by_name = {r.name: r for r in self.regions}
        # Number fields keep the 0.4 confidence cut-off. Real-footage A/B (Starship, 1,630 frames):
        # without it, gauge icons read as '2'/'72'/'8' and dropped leading digits gave ~290 wrong
        # speeds; with it, 0-1 wrong numbers at 92-97% read rate. The clock has no cut-off because
        # its parser + lock reject junk (61% -> 98.6% read rate).
        nums = {k: (parse_field(v[0], by_name[k]) if v[1] >= NUM_CONF else None)
                for k, v in raw.items() if by_name[k].kind == "number"}
        # Low-confidence reads (Flight 15 booster speed read right at 0.1-0.37) may only continue
        # an already-acquired track, close to where it is heading. See vehicle.Channel.push.
        weak = {k: (parse_field(v[0], by_name[k]) if WEAK_CONF <= v[1] < NUM_CONF else None)
                for k, v in raw.items() if by_name[k].kind == "number"}
        # Starship moves vehicles between boxes mid-flight; route boxes to booster/ship channels.
        labels = {k: v[0] for k, v in raw.items() if by_name[k].kind == "text"}
        had_stack = "stack" in self.layout.current.values()
        split = []
        for typ, data in self.layout.update(mt, labels):
            self.em.emit(typ, data, t, mt)
        if had_stack and "stack" not in self.layout.current.values():
            split = self.veh.stack_ended(t, mt)
        box_nums = nums
        nums = self.layout.route(nums)
        weak = self.layout.route(weak)
        # Telemetry is meaningless before liftoff; SpaceX shows 0s anyway.
        for typ, data in split + self.veh.update(t, mt, nums, weak):
            if typ in ("milestone", "milestone.confirm"):
                if data["id"] in self.ms_drop:
                    continue
                data = {**data, "id": self.ms_alias.get(data["id"], data["id"])}
            if typ == "milestone" and self.fusion:
                data = self.fusion.on_detection(data)
            elif typ == "milestone":
                data = {**data, "source": "ocr", "status": "detected"}
            self.em.emit(typ, data, t, mt)
        if self.fusion:
            for data in self.fusion.tick(mt, self.clock.state):
                self.em.emit("milestone", data, t, mt)

        # 1 Hz clock tick + telemetry sample
        if self.last_tick is None or t - self.last_tick >= 0.999:
            self.last_tick = t
            self.em.emit("clock.tick", {"state": self.clock.state,
                                        "display": fmt_clock(mt) if mt is not None else None,
                                        "confidence": self.clock_conf}, t, mt)
            s = self.veh.snapshot(t)
            for st in ("s1", "s2"):
                v, h = s[f"{st}_speed"], s[f"{st}_alt"]
                if v is None and h is None:
                    continue
                self.em.emit("telemetry", {"stage": st, "speed_kmh": v, "alt_km": h,
                                           "accel_kmh_s": _r(self.veh.ch[f"{st}_speed"].rate()),
                                           "q_kpa": _r(dynamic_pressure_kpa(v, h)) if v is not None and h is not None else None,
                                           "phase": self.veh.phase[st]}, t, mt)
        if self.preview is not None:
            self._draw_preview(frame, raw, box_nums, creading, mt)
        if self.frames % 20 == 0:
            self.em.emit("ingest.health", {
                "frames": self.frames, "clock_locked": self.clock.offset is not None,
                "region_conf": {k: round(sum(v) / len(v), 2) if v else None for k, v in self.conf_acc.items()}}, t, mt)
            self.conf_acc = {r.name: [] for r in self.regions}


    def _draw_preview(self, frame, raw, box_nums, creading, mt):
        slot_of = {}
        for slot, cfg in self.layout.slots.items():
            veh = self.layout.current.get(slot)
            for f in ("speed", "alt"):
                if cfg.get(f):
                    slot_of[cfg[f]] = veh or "ignored"
        parsed = dict(box_nums)
        parsed["clock"] = creading
        s = self.veh.snapshot(t)
        tel = {}
        for st, name in (("s1", "booster/S1"), ("s2", "ship/S2")):
            v, h = s[f"{st}_speed"], s[f"{st}_alt"]
            tel[name] = f"{'-' if v is None else round(v)} km/h  {'-' if h is None else round(h, 1)} km  {self.veh.phase[st]}"
        header = [f"clock {fmt_clock(mt) if mt is not None else '--'}  [{self.clock.state}]",
                  f"frame {self.frames}"]
        self.preview.draw(frame, self.regions, raw, parsed, slot_of, header, tel)


def _r(x, n=1):
    return None if x is None else round(x, n)


def fmt_clock(mt: float) -> str:
    s = int(round(abs(mt)))
    return f"T{'-' if mt < 0 else '+'} {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ffmpeg", "images", "obs"], required=True)
    ap.add_argument("--input", required=True, help="file/URL/device, glob, or OBS source name")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--realtime", action="store_true",
                    help="ffmpeg source: read the file at playback speed (ffmpeg -re), for hub replay rehearsals")
    ap.add_argument("--preview", nargs="?", const=8090, type=int, metavar="PORT",
                    help="serve a live annotated view at http://127.0.0.1:PORT/ (default 8090)")
    ap.add_argument("--profile", default=os.path.join(HERE, "profiles", "spacex_falcon.json"))
    ap.add_argument("--timeline", help="planned milestones JSON (from the runtime rundown)")
    ap.add_argument("--jsonl", help="append events to this file")
    ap.add_argument("--udp", help="host:port to send each event as a UDP datagram")
    ap.add_argument("--hub", action="store_true", help="send to the runtime hub (UDP 127.0.0.1:4101)")
    ap.add_argument("--assume", action="store_true",
                    help="emit 'assumed' milestones from the timeline yourself (standalone use; the hub does this otherwise)")
    ap.add_argument("--http", help="POST each event to this URL")
    ap.add_argument("--obs-password", default=os.environ.get("OBS_PASSWORD"))
    ap.add_argument("--quiet", default="clock.tick,telemetry", help="types not echoed to stdout")
    a = ap.parse_args(argv)

    with open(a.profile) as f:
        profile = json.load(f)
    tl = load(a.timeline) if a.timeline else None
    if a.hub and not a.udp:
        a.udp = "127.0.0.1:4101"
    em = Emitter(a.jsonl, a.udp, a.http, quiet_types=set(filter(None, a.quiet.split(","))))
    pipe = Pipeline(profile, em, tl, a.assume)
    if a.preview:
        from .preview import Preview
        pipe.preview = Preview(a.preview)
        _emit = em.emit

        def emit_and_note(typ, data, media_t, mission_t):
            pipe.preview.note(typ, data, mission_t)
            return _emit(typ, data, media_t, mission_t)
        em.emit = emit_and_note

    if a.source == "ffmpeg":
        src = sources.ffmpeg_source(a.input, fps=a.fps, realtime=a.realtime)
    elif a.source == "images":
        src = sources.image_dir_source(a.input, fps=a.fps)
    else:
        src = sources.obs_source(a.input, password=a.obs_password, fps=a.fps)
    for t, frame in src:
        pipe.process(t, frame)


if __name__ == "__main__":
    main()
