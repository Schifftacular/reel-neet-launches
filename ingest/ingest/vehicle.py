"""Telemetry filtering and vehicle-state/milestone detection from OCR'd stage data."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# Physical plausibility limits used to reject OCR misreads.
MAX_RATE = {"speed": 350.0, "alt": 12.0}      # km/h per s (~10 g), km per s
ACCEL_WIN = 3.0                                 # seconds for derivative fit
ACQUIRE_GAP = 10.0                              # no good read for this long -> re-acquire
STALE_SECS = 3.0                                # snapshot stops reporting a channel this long after its last read
# Digits Tesseract confuses on the SpaceX font. Flight 15: a leading 5 read as 9 on clean crops
# (5932 -> 9932, 5x.x km -> 9x.x km) at confidence 0.87, consistently, for 30 s at a time.
CONFUSABLE = {"5": "96", "9": "5", "8": "306", "3": "8", "6": "85", "0": "8", "1": "7", "7": "1"}


def _alternates(v: float, kind: str):
    s = f"{int(round(v))}" if kind == "speed" else f"{v:.1f}"
    for i, ch in enumerate(s):
        for alt in CONFUSABLE.get(ch, ""):
            if i == 0 and alt == "0":
                continue
            try:
                yield float(s[:i] + alt + s[i + 1:])
            except ValueError:
                pass


@dataclass
class Channel:
    kind: str                     # "speed" | "alt"
    last: tuple[float, float] | None = None
    rejects: list = field(default_factory=list)
    hist: deque = field(default_factory=lambda: deque(maxlen=40))
    corrected: int = 0

    def _ok(self, t: float, v: float, slack: float = 1.0) -> bool:
        lt, lv = self.last
        dt = max(t - lt, 0.25)
        return abs(v - lv) <= slack * MAX_RATE[self.kind] * dt + (5 if self.kind == "speed" else 0.5)

    def _tight(self, pred: float) -> float:
        return max(0.03 * abs(pred), 40.0 if self.kind == "speed" else 1.5)

    def _predict(self, t: float) -> float:
        lt, lv = self.last
        r = self.rate() or 0.0
        return lv + r * (t - lt)

    def push(self, t: float, v: float | None, weak: bool = False) -> float | None:
        """weak=True: a low-confidence read. Used only to continue an acquired channel, and only
        when it sits close to where the channel is heading (it can never acquire or re-acquire)."""
        if v is None:
            return None
        acquired = self.last is not None and t - self.last[0] <= ACQUIRE_GAP
        if weak:
            if not acquired:
                return None
            pred = self._predict(t)
            if abs(v - pred) > self._tight(pred):
                return None
            self.last = (t, v)
            self.hist.append((t, v))
            return v
        if acquired and not self._ok(t, v):
            # A misread digit? Take the one-digit lookalike that fits the track, if exactly one does.
            # Only a lookalike that lands close to the prediction counts: a real jump the gate was
            # wrong about must not be bent into a nearby lookalike.
            pred = self._predict(t)
            fits = sorted((abs(a - pred), a) for a in _alternates(v, self.kind)
                          if abs(a - pred) <= self._tight(pred))
            if fits and (len(fits) == 1 or fits[1][0] > 3 * fits[0][0]):
                v = fits[0][1]
                self.corrected += 1
        if self.last is None or t - self.last[0] > ACQUIRE_GAP:
            # (Re)acquire only on 3 consistent reads. Real footage: gauge icons read as lone
            # digits ('4', '7', '6') while a field was off screen.
            self.rejects = (self.rejects + [(t, v)])[-3:]
            if len(self.rejects) == 3 and _consistent(self.rejects, self.kind) and self.rejects[-1][0] - self.rejects[0][0] <= 4:
                self.rejects.clear()
                self.last = (t, v)
                self.hist.append((t, v))
                return v
            return None
        if self.last is not None:
            lt, lv = self.last
            dt = max(t - lt, 0.25)
            if abs(v - lv) > MAX_RATE[self.kind] * dt + (5 if self.kind == "speed" else 0.5):
                self.rejects.append((t, v))
                self.rejects = self.rejects[-3:]
                # three rejected readings that agree with each other -> we were wrong, re-acquire
                if len(self.rejects) == 3 and _consistent(self.rejects, self.kind):
                    self.rejects.clear()
                else:
                    return None
        self.rejects.clear()
        self.last = (t, v)
        self.hist.append((t, v))
        return v

    def value(self, now: float | None = None) -> float | None:
        if not self.hist:
            return None
        if now is not None and now - self.hist[-1][0] > STALE_SECS:
            return None
        vals = sorted(v for _, v in list(self.hist)[-3:])
        return vals[len(vals) // 2]

    def rate(self) -> float | None:
        """Least-squares slope over the last ACCEL_WIN seconds."""
        if len(self.hist) < 3:
            return None
        t_end = self.hist[-1][0]
        pts = [(t, v) for t, v in self.hist if t >= t_end - ACCEL_WIN]
        if len(pts) < 3:
            return None
        n = len(pts)
        mt = sum(t for t, _ in pts) / n
        mv = sum(v for _, v in pts) / n
        den = sum((t - mt) ** 2 for t, _ in pts)
        return sum((t - mt) * (v - mv) for t, v in pts) / den if den else None


def _consistent(pts, kind) -> bool:
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if abs(v1 - v0) > MAX_RATE[kind] * max(t1 - t0, 0.25) + 5:
            return False
    return True


def dynamic_pressure_kpa(speed_kmh: float, alt_km: float) -> float:
    """q = 1/2 rho v^2 with an exponential atmosphere. Uses ground speed, so it is approximate."""
    rho = 1.225 * math.exp(-alt_km / 8.5)
    v = speed_kmh / 3.6
    return 0.5 * rho * v * v / 1000.0


class VehicleTracker:
    """Turns filtered stage telemetry + mission time into phases and milestone detections."""

    def __init__(self):
        self.ch = {k: Channel(k.split("_")[1]) for k in ("s1_speed", "s1_alt", "s2_speed", "s2_alt")}
        self.phase = {"s1": "prelaunch", "s2": "attached"}
        self.done: dict[str, float] = {}
        self.q_peak = (0.0, None)
        self.q_falling = 0
        self.prev_mt: float | None = None
        self.terminal_seen = False   # booster reached ~terminal velocity after the entry burn
        self.burn_min = 0.0   # most negative accel seen in the current booster burn
        self.mt_of: dict[float, float] = {}   # media t -> mission t, for back-dating

    def _hit(self, out, mid, mt, conf, **ev):
        if mid not in self.done:
            self.done[mid] = mt
            out.append(("milestone", {"id": mid, "t": round(mt, 1), "confidence": conf, "evidence": ev}))

    def _phase(self, out, stage, ph):
        if self.phase[stage] != ph:
            out.append(("vehicle.status", {"stage": stage, "phase": ph, "prev": self.phase[stage]}))
            self.phase[stage] = ph

    def update(self, t: float, mt: float | None, raw: dict[str, float | None],
               weak: dict[str, float | None] | None = None) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        weak = weak or {}
        for k, ch in self.ch.items():
            if raw.get(k) is not None:
                ch.push(t, raw[k])
            elif weak.get(k) is not None:
                ch.push(t, weak[k], weak=True)
        if mt is None:
            return out
        v1, h1 = self.ch["s1_speed"].value(), self.ch["s1_alt"].value()
        v2 = self.ch["s2_speed"].value()
        a1, a2 = self.ch["s1_speed"].rate(), self.ch["s2_speed"].rate()
        d = self.done

        # Liftoff: the clock crossing zero is authoritative; telemetry raises confidence.
        if self.prev_mt is not None and self.prev_mt < 0 <= mt:
            self._hit(out, "liftoff", 0.0, 0.8, by="clock")
            self._phase(out, "s1", "ascent")
        if "liftoff" in d and "liftoff_confirmed" not in d and v1 and v1 > 15 and mt < 30:
            d["liftoff_confirmed"] = mt
            # Separate type, not a second "milestone": the hub re-anchors T-0 on every liftoff milestone.
            out.append(("milestone.confirm", {"id": "liftoff", "t": d["liftoff"], "confidence": 0.98,
                                              "evidence": {"by": "clock+s1_speed", "s1_speed": v1}}))

        if v1 is not None and h1 is not None and "liftoff" in d and "meco" not in d:
            q = dynamic_pressure_kpa(v1, h1)
            if q > self.q_peak[0]:
                self.q_peak, self.q_falling = (q, mt), 0
            elif q < 0.97 * self.q_peak[0] and self.q_peak[0] > 15:
                self.q_falling += 1
                if self.q_falling >= 3:
                    self._hit(out, "maxq", self.q_peak[1], 0.6, q_kpa=round(self.q_peak[0], 1))

        if a1 is not None and mt > 60:
            if "liftoff" in d and "meco" not in d and v1 and v1 > 3000 and a1 < 0:
                # back-date to the speed peak: that is when the engines actually cut off
                tp, vp = max(self.ch["s1_speed"].hist, key=lambda p: p[1])
                self._hit(out, "meco", self._mt(tp, t, mt), 0.85, s1_speed=vp, s1_accel=round(a1, 1))
                self._phase(out, "s1", "coast")
            elif "meco" in d and h1 is not None:
                # Booster burns show up as hard deceleration. Which burn it is depends on
                # whether the booster is still climbing (boostback; RTLS/some ASDS) or falling.
                hr = self.ch["s1_alt"].rate() or 0.0
                ph = self.phase["s1"]
                burning = ph in ("boostback", "entry_burn", "landing_burn")
                if burning:
                    self.burn_min = min(self.burn_min, a1)
                if ph == "descent" and h1 < 15 and a1 > -20:
                    self.terminal_seen = True
                if not burning and a1 < -30:
                    if a1 > -60 and not (hr < 0 and h1 < 8 and self.terminal_seen):
                        pass  # gentle decel is aero drag/coast, not a burn (landing burn is the exception)
                    elif hr >= 0 and "boostback_start" not in d and "entry_start" not in d:
                        self._hit(out, "boostback_start", mt - 1.0, 0.8, s1_accel=round(a1, 1), s1_alt=h1)
                        self._phase(out, "s1", "boostback")
                        self.burn_min = a1
                    elif hr < 0 and h1 < 80 and "entry_start" not in d:
                        self._hit(out, "entry_start", mt - 1.0, 0.75, s1_accel=round(a1, 1), s1_alt=h1)
                        self._phase(out, "s1", "entry_burn")
                        self.burn_min = a1
                    elif hr < 0 and h1 < 8 and self.terminal_seen and "landing_start" not in d:
                        self._hit(out, "landing_start", mt - 1.0, 0.75, s1_accel=round(a1, 1), s1_alt=h1)
                        self._phase(out, "s1", "landing_burn")
                # Burn over once deceleration has eased to half its peak, or by 45 km/h/s. A fixed threshold
                # fails after the entry burn because aero drag alone is still -50 km/h/s or so.
                elif burning and (a1 > 0.5 * self.burn_min or a1 - self.burn_min > 45) and ph != "landing_burn":
                    self.burn_min = 0.0
                    end = {"boostback": "boostback_end", "entry_burn": "entry_end"}[ph]
                    self._hit(out, end, mt - 1.0, 0.7, s1_accel=round(a1, 1), s1_alt=h1)
                    self._phase(out, "s1", "coast" if ph == "boostback" else "descent")
        if "landing_start" in d and "landing" not in d and h1 is not None and v1 is not None and h1 < 0.2 and v1 < 30:
            self._hit(out, "landing", mt, 0.9, s1_alt=h1, s1_speed=v1)
            self._phase(out, "s1", "landed")

        if a2 is not None and "meco" in d:
            if "ses1" not in d and a2 > 12:
                # back-date to the slowest S2 speed since MECO (start of the burn)
                since = [p for p in self.ch["s2_speed"].hist if self._mt(p[0], t, mt) >= d["meco"]]
                tp = min(since, key=lambda p: p[1])[0] if since else t
                self._hit(out, "ses1", self._mt(tp, t, mt), 0.85, s2_accel=round(a2, 1))
                self._phase(out, "s2", "burn")
            elif "ses1" in d and "seco1" not in d and v2 and v2 > 20000 and a2 < 5:
                self._hit(out, "seco1", mt - 1.5, 0.85, s2_speed=v2)
                self._phase(out, "s2", "coast")

        self.prev_mt = mt
        return out

    def stack_ended(self, t: float, mt: float | None) -> list[tuple[str, dict]]:
        """Starship hot staging: the overlay stops showing one stacked vehicle and splits it.
        The stack's speed never visibly drops before the ship lights, so MECO is taken as the
        peak of the stack speed, back-dated, when the split happens."""
        out: list[tuple[str, dict]] = []
        h = self.ch["s1_speed"].hist
        if mt is None or "liftoff" not in self.done or "meco" in self.done or not h:
            return out
        tp, vp = max(h, key=lambda p: p[1])
        if vp > 3000:
            self._hit(out, "meco", self._mt(tp, t, mt), 0.75, by="stack_split", s1_speed=vp)
            self._phase(out, "s1", "coast")
        return out

    @staticmethod
    def _mt(tp: float, t: float, mt: float) -> float:
        """Mission time of an earlier media timestamp (clock runs 1:1 in flight)."""
        return mt - (t - tp)

    def snapshot(self, now: float | None = None) -> dict:
        """Current values; a channel with no accepted read for STALE_SECS reports None, so a gauge
        that left the screen (or was replaced by an engine diagram) isn't shown frozen."""
        return {k: ch.value(now) for k, ch in self.ch.items()}
