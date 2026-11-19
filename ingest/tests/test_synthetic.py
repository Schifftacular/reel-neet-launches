"""End-to-end run over synthetic frames.  python -m tests.test_synthetic [--fps 1] [--out events.jsonl]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.emit import Emitter  # noqa: E402
from ingest.run import HERE, Pipeline  # noqa: E402
from ingest.timeline import load  # noqa: E402
from tests.synth import scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", default="synthetic_events.jsonl")
    ap.add_argument("--timeline")
    ap.add_argument("--assume", action="store_true")
    ap.add_argument("--save-frame", help="write the frame at mission T+120 here (for calibrate.py)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    prof = json.load(open(os.path.join(HERE, "profiles", "spacex_falcon.json")))
    tl = load(a.timeline or os.path.join(HERE, "timelines", "falcon9_sample.json"))
    em = Emitter(a.out, quiet_types={"clock.tick", "telemetry", "ingest.health"})
    pipe = Pipeline(prof, em, tl, a.assume)
    clock_err, n_clock = [], 0
    t0 = time.time()
    for media_t, frame, truth in scenario(fps=a.fps):
        if a.save_frame and abs(truth - 120) < 0.5:
            import cv2
            cv2.imwrite(a.save_frame, frame)
        pipe.process(media_t, frame)
        est = pipe.clock.mission_time(media_t)
        if est is not None:
            clock_err.append(abs(est - truth))
        n_clock += 1
    el = time.time() - t0
    errs = sorted(clock_err)
    print(f"\nframes={n_clock} elapsed={el:.0f}s ({n_clock / el:.1f} fps)", file=sys.stderr)
    print(f"clock |err|: median={errs[len(errs) // 2]:.2f}s p95={errs[int(len(errs) * .95)]:.2f}s "
          f"max={errs[-1]:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
