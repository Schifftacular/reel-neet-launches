"""Calibrate overlay regions against a real frame.

  python -m ingest.calibrate frame.png [--profile profiles/spacex_falcon.json] [--out calib/]

Grab a frame from a recent Falcon 9 webcast VOD during ascent (both stages' numbers
visible), e.g.:  ffmpeg -ss 00:25:00 -i vod.mp4 -frames:v 1 frame.png
Writes calib/boxes.png (boxes drawn), one crop + preprocessed crop per region, and
prints what OCR read. Nudge the profile numbers until every field reads cleanly.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

from .ocr import Region, parse_clock_loose, parse_field, preprocess, read_region
from .run import HERE


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--profile", default=os.path.join(HERE, "profiles", "spacex_falcon.json"))
    ap.add_argument("--out", default="calib")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    frame = cv2.imread(a.frame)
    if frame.shape[1] != 1920:
        frame = cv2.resize(frame, (1920, 1080))
    prof = json.load(open(a.profile))
    vis = frame.copy()
    for rd in prof["regions"]:
        r = Region(**rd)
        H, W = frame.shape[:2]
        p0 = (int(r.x * W), int(r.y * H))
        p1 = (int((r.x + r.w) * W), int((r.y + r.h) * H))
        cv2.rectangle(vis, p0, p1, (0, 255, 0), 2)
        cv2.putText(vis, r.name, (p0[0], p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(a.out, f"{r.name}_crop.png"), r.crop(frame))
        cv2.imwrite(os.path.join(a.out, f"{r.name}_bw.png"), preprocess(r.crop(frame)))
        text, conf = read_region(frame, r)
        parsed = parse_clock_loose(text) if r.kind == "clock" else parse_field(text, r)
        print(f"{r.name:10s} conf={conf:.2f} text={text!r:24s} parsed={parsed}")
    cv2.imwrite(os.path.join(a.out, "boxes.png"), vis)
    print(f"wrote {a.out}/boxes.png")


if __name__ == "__main__":
    main()
