"""Frame sources. Each yields (t_media_seconds, BGR ndarray).

Recommended live setup: read frames straight out of OBS (the program feed you are
already decoding) instead of opening a second copy of the stream. Use either:
  * OBS virtual camera  -> `ffmpeg` source with a v4l2/dshow/avfoundation device, or
  * OBS WebSocket       -> `obs` source (GetSourceScreenshot on the webcast source).
For testing, point the `ffmpeg` source at a recorded VOD file.
"""
from __future__ import annotations

import base64
import glob
import os
import subprocess
import time
from typing import Iterator

import cv2
import numpy as np


def ffmpeg_source(url: str, fps: float = 2.0, width: int = 1920, height: int = 1080,
                  input_args: list[str] | None = None, realtime: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    """Anything ffmpeg can open: a file, an HLS URL, a capture device, `streamlink ... -O |` via '-'."""
    cmd = ["ffmpeg", "-loglevel", "error"]
    if realtime:
        cmd += ["-re"]
    cmd += (input_args or []) + ["-i", url, "-vf", f"fps={fps},scale={width}:{height}",
                                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        proc.kill()


def image_dir_source(pattern: str, fps: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
    """Sorted image files, e.g. 'frames/*.png'. Timestamp from a '_t<secs>' suffix if present, else index/fps."""
    for i, path in enumerate(sorted(glob.glob(pattern))):
        stem = os.path.splitext(os.path.basename(path))[0]
        t = i / fps
        if "_t" in stem:
            try:
                t = float(stem.rsplit("_t", 1)[1])
            except ValueError:
                pass
        yield t, cv2.imread(path)


def obs_source(source_name: str, host: str = "localhost", port: int = 4455, password: str | None = None,
               fps: float = 2.0, width: int = 1920) -> Iterator[tuple[float, np.ndarray]]:
    """Grab frames from an OBS source via obs-websocket v5 (pip install obsws-python)."""
    import obsws_python as obs  # optional dependency

    cl = obs.ReqClient(host=host, port=port, password=password or "")
    t0 = time.monotonic()
    period = 1.0 / fps
    while True:
        tick = time.monotonic()
        r = cl.get_source_screenshot(source_name, "jpg", width, None, 90)
        b64 = r.image_data.split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        yield tick - t0, img
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
