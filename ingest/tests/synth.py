"""Synthetic webcast frames: a generic launch overlay (not SpaceX artwork) laid out to
match profiles/spacex_falcon.json, driven by a simulated Falcon-like flight profile.

Includes a countdown hold, JPEG artifacts, noise, and occasional garbled frames so
the clock model and telemetry filters get exercised.
"""
from __future__ import annotations

import io
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_CLOCK = ImageFont.truetype(FONT, 54)
F_NUM = ImageFont.truetype(FONT, 38)
F_LAB = ImageFont.truetype(FONT, 18)


def q(v, h):
    return 0.5 * 1.225 * math.exp(-h / 8.5) * (v / 3.6) ** 2 / 1000


def profile(t: float) -> dict:
    """Returns s1/s2 speed (km/h) and altitude (km) at mission time t."""
    if t <= 0:
        return dict(s1v=0, s1h=0, s2v=0, s2h=0)
    # stage 1
    if t <= 147:
        v1 = 8200 * (t / 147) ** 1.6
        h1 = 68 * (t / 147) ** 2.0
    elif t <= 375:   # ballistic coast: climbs to ~125 km apogee near T+260, then falls
        x = (t - 147) / (375 - 147)
        v1 = 8200 - 1400 * math.sin(math.pi * x * 0.9) + 300 * x
        h1 = 68 + 57 * math.sin(math.pi * x * 0.8) - 18 * x
        h1 = max(h1, 0)
    elif t <= 395:   # entry burn
        x = (t - 375) / 20
        v1 = 8100 - 2700 * x
        h1 = 52 - 12 * x
    elif t <= 455:   # aero descent: strong drag (~-73 km/h/s)
        x = (t - 395) / 60
        v1 = 5400 - 4400 * x
        h1 = 40 - 32 * x
    elif t <= 480:   # near-terminal velocity, barely slowing
        x = (t - 455) / 25
        v1 = 1000 - 100 * x
        h1 = 8 - 5 * x
    elif t <= 505:   # landing burn
        x = (t - 480) / 25
        v1 = 900 * (1 - x)
        h1 = 3 * (1 - x) ** 2
    else:
        v1, h1 = 0, 0
    # stage 2: rides along until sep, coasts, burns from T+158 to SECO at T+520
    if t <= 151:
        v2, h2 = (v1, h1) if t <= 147 else (8200 - 2 * (t - 147), profile(min(t, 147))["s1h"] + 1.2 * (t - 147))
    elif t <= 158:
        v2, h2 = 8190 - 2 * (t - 151), 72 + (t - 151)
    elif t <= 520:
        x = (t - 158) / (520 - 158)
        v2 = 8180 + (27300 - 8180) * (0.45 * x + 0.55 * x ** 2)   # ~24 km/h/s at ignition, rising
        h2 = 79 + 130 * math.sin(math.pi / 2 * x)
    else:
        v2, h2 = 27300, 209 + 0.05 * (t - 520)
    return dict(s1v=v1, s1h=h1, s2v=v2, s2h=h2)


def clock_text(mt: float) -> str:
    s = int(math.floor(abs(mt) + 1e-6)) if mt >= 0 else int(math.ceil(abs(mt) - 1e-6))
    return f"T{'-' if mt < 0 else '+'} {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def background(mt: float) -> Image.Image:
    # cheap moving "video": vertical gradient with a bright plume blob
    y = np.linspace(0, 1, H)[:, None]
    base = np.concatenate([(30 + 90 * y) * np.ones((1, W))] * 3).reshape(3, H, W).transpose(1, 2, 0)
    base[..., 2] *= 0.6
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    cx, cy = W // 2 + int(80 * math.sin(mt / 9)), int(H * 0.45)
    d.ellipse([cx - 60, cy - 160, cx + 60, cy + 160], fill=(255, 230, 190))
    return img


def render(mt: float, garble: bool = False) -> np.ndarray:
    p = profile(mt)
    img = background(mt)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, int(H * 0.86), W, int(H * 0.965)], fill=(8, 10, 14, 215))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)

    def num(v, nd=0):
        return f"{v:,.{nd}f}"

    ct = clock_text(mt)
    fields = [
        (0.40, 0.886, ct, F_CLOCK),
        (0.065, 0.887, num(p["s1v"]), F_NUM), (0.205, 0.887, num(p["s1h"], 1 if p["s1h"] < 100 else 0), F_NUM),
        (0.715, 0.887, num(p["s2v"]), F_NUM), (0.855, 0.887, num(p["s2h"], 1 if p["s2h"] < 100 else 0), F_NUM),
    ]
    for x, y, text, font in fields:
        if garble and random.random() < 0.5:
            text = "".join(random.choice("0123456789,") for _ in text)
        d.text((int(x * W), int(y * H)), text, font=font, fill=(240, 240, 240))
    for x, lab in [(0.065, "STAGE 1  SPEED KM/H"), (0.205, "ALTITUDE KM"), (0.715, "STAGE 2  SPEED KM/H"),
                   (0.855, "ALTITUDE KM"), (0.465, "MISSION CLOCK")]:
        d.text((int(x * W), int(0.935 * H)), lab, font=F_LAB, fill=(150, 160, 170))

    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    arr = np.clip(arr.astype(np.int16) + np.random.normal(0, 6, arr.shape).astype(np.int16), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 55])  # stream-like compression
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def scenario(fps: float = 1.0, start: float = -60, end: float = 560, hold_at: float = -40, hold_len: float = 20,
             garble_every: int = 37, seed: int = 7):
    """Yields (media_t, frame, truth_mission_t). Countdown holds at `hold_at` for `hold_len` media seconds."""
    random.seed(seed)
    np.random.seed(seed)
    media_t, i = 0.0, 0
    dt = 1.0 / fps
    while True:
        if media_t < (hold_at - start):
            mt = start + media_t
        elif media_t < (hold_at - start) + hold_len:
            mt = hold_at
        else:
            mt = start + media_t - hold_len
        if mt > end:
            break
        yield media_t, render(mt, garble=(garble_every and i % garble_every == garble_every - 1)), mt
        media_t += dt
        i += 1
