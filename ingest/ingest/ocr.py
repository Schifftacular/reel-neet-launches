"""Region cropping, preprocessing and OCR for webcast overlay fields."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

# Tesseract's OpenMP threads fight each other when several regions OCR in parallel:
# one thread per call is ~100x faster here (0.14 s vs 14 s per frame on 4 cores).
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

DIGITS = "0123456789"


@dataclass
class Region:
    name: str
    x: float  # normalized 0..1 of frame width
    y: float
    w: float
    h: float
    kind: str  # "clock" | "number" | "text"
    whitelist: str | None = None
    integer: bool = False   # True: '.' and ',' are both thousands separators (speed). False: decimal allowed (altitude)
    decimal_below: float | None = None  # overlay always shows a decimal under this value (SpaceX altitude: 100 km)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        x0, y0 = int(self.x * W), int(self.y * H)
        x1, y1 = int((self.x + self.w) * W), int((self.y + self.h) * H)
        return frame[y0:y1, x0:x1]


def preprocess(crop: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Overlay text is light on dark. Return black text on white, upscaled."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Make text black: if most pixels are black (dark background), invert.
    if np.mean(bw) < 127:
        bw = 255 - bw
    return cv2.copyMakeBorder(bw, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def read_region(frame: np.ndarray, r: Region) -> tuple[str, float]:
    """Return (text, confidence 0..1). Retries with thinner strokes when the first read looks bad:
    Tesseract misreads heavy bold digits (a bold '8' comes back as '§' or is dropped).
    Real-footage finding: Tesseract often scores a perfectly read clock at conf 0, so a read
    that parses is preferred over one that only has higher confidence."""
    base = preprocess(r.crop(frame))
    best: tuple[str, float] | None = None
    best_ok = False
    for img in (base, cv2.dilate(base, np.ones((3, 3), np.uint8))):
        text, conf = _tess(img, r)
        ok = _parses(text, r)
        if best is None or (ok and not best_ok) or (ok == best_ok and conf > best[1]):
            best, best_ok = (text, conf), ok
        if ok and conf >= 0.8 and text[0] not in ",.":
            break
    return best if best is not None else ("", 0.0)


def parse_field(text: str, r: Region) -> float | None:
    """parse_number plus the field's format rules. Real footage: icons read as lone digits ('4')
    where altitude is normally '4.2', so a decimal-less small altitude is rejected."""
    v = parse_number(text, r.integer)
    if v is not None and r.decimal_below is not None and v < r.decimal_below and not any(c in text for c in ".,"):
        return None
    return v


def _parses(text: str, r: Region) -> bool:
    if not text:
        return False
    if r.kind == "clock":
        return parse_clock_loose(text) is not None
    if r.kind == "text":
        return sum(c.isalpha() for c in text) >= 3
    return parse_field(text, r) is not None


def _tess(img: np.ndarray, r: Region) -> tuple[str, float]:
    wl = r.whitelist
    if wl is None:
        # No whitelist for the clock: Tesseract 5's LSTM reports conf=0 for whitelisted
        # mixed-symbol strings like "T+00:02:00". The clock regex validates instead.
        wl = {"number": DIGITS + ",."}.get(r.kind)
    cfg = "--psm 7 --oem 1"
    if wl:
        cfg += f" -c tessedit_char_whitelist={wl.replace(' ', '')}"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    text = " ".join(t for t, _ in words)
    conf = max(c for _, c in words) / 100.0   # min() let one conf-0 word zero a correct read
    return text, conf


# ---------- parsers ----------

_CLOCK = re.compile(r"([+-])?(\d{1,3}):(\d{2}):(\d{2})")
_LOOKALIKE = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1"})


def parse_clock_loose(text: str) -> tuple[str | None, float] | None:
    """-> (sign or None, seconds). Real footage: zeros come back as O/Q and the small '+' is
    often lost ('T 00:05:59', 'T00:10:31'), so the sign is optional here."""
    t = text.replace(" ", "")
    t = t[1:] if t[:1] in "Tt" else t
    t = t.translate(_LOOKALIKE).replace("—", "-").replace("–", "-")
    m = _CLOCK.search(t)
    if not m:
        return None
    sign, h, mi, s = m.groups()
    h = h[-2:]  # a doubled zero ('T+000:02:09') just adds a leading digit
    if int(mi) > 59 or int(s) > 59:
        return None
    return sign, float(int(h) * 3600 + int(mi) * 60 + int(s))


def parse_clock(text: str) -> float | None:
    """'T- 00:01:23' -> -83.0 ; 'T+ 00:02:31' -> 151.0. None if the sign is missing (see parse_clock_loose)."""
    p = parse_clock_loose(text)
    if not p or p[0] is None:
        return None
    return -p[1] if p[0] == "-" else p[1]


def parse_number(text: str, integer: bool = False) -> float | None:
    """OCR regularly swaps ',' and '.', so the field type decides what they mean."""
    t = text.replace(" ", "")
    if integer:
        t = t.replace(",", "").replace(".", "")
    else:
        t = t.replace(",", ".")
        if t.count(".") > 1:
            return None
    m = re.search(r"\d+(\.\d+)?", t)
    return float(m.group(0)) if m else None
