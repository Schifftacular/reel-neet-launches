"""Mission clock model.

OCR gives a noisy, low-rate reading of the on-screen T-clock. We keep a locked
offset (mission_time = media_time + offset) and only let OCR *correct* it, so a
single misread never moves the clock. A frozen reading means a HOLD; a
consistent new offset (hold resumed, recycle, stream skip) re-locks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

TOL = 1.6          # seconds of disagreement tolerated before a reading is "off-lock"
RELOCK_N = 3       # consecutive agreeing off-lock readings needed to re-lock
HOLD_SECS = 3.5    # reading frozen for this long while counting down -> HOLD
LOST_SECS = 20.0   # no valid reading for this long -> LOST


@dataclass
class ClockModel:
    offset: float | None = None
    state: str = "no_clock"            # no_clock | countdown | hold | flight | lost
    held_value: float | None = None
    last_valid_t: float | None = None
    _off: deque = field(default_factory=lambda: deque(maxlen=RELOCK_N))
    _frozen_since: tuple[float, float] | None = None   # (media_t, value)

    def mission_time(self, t: float) -> float | None:
        if self.state == "hold":
            return self.held_value
        if self.offset is None:
            return None
        return t + self.offset

    def resolve(self, t: float, sign: str | None, mag: float) -> float | None:
        """Turn an OCR read into mission seconds. Real footage loses the small '+' often, so an
        unsigned read takes whichever sign agrees with the locked clock (or is dropped if neither does)."""
        if sign is not None:
            return -mag if sign == "-" else mag
        if self.state == "hold" and self.held_value is not None:
            pred = self.held_value
        elif self.offset is not None:
            pred = t + self.offset
        else:
            return None
        best = min((mag, -mag), key=lambda v: abs(v - pred))
        return best if abs(best - pred) <= 2 * TOL else None

    def update(self, t: float, reading: float | None) -> list[tuple[str, dict]]:
        """Feed one frame. Returns a list of (event_type, data) transitions."""
        ev: list[tuple[str, dict]] = []
        prev_state = self.state

        if reading is None:
            if self.offset is not None and self.last_valid_t is not None and t - self.last_valid_t > LOST_SECS and self.state != "lost":
                self.state = "lost"
            elif self.offset is not None and self.state in ("countdown", "flight"):
                # keep state in step with the model's own time (real run: stayed "countdown" past T+0)
                self.state = "countdown" if t + self.offset < 0 else "flight"
            if self.state != prev_state:
                ev.append(("clock.state", {"state": self.state, "prev": prev_state}))
            return ev
        self.last_valid_t = t

        # --- hold detection: the displayed value is not advancing ---
        if self._frozen_since and abs(reading - self._frozen_since[1]) < 0.5:
            if t - self._frozen_since[0] >= HOLD_SECS and reading < 0 and self.state != "hold":
                self.state, self.held_value = "hold", reading
                ev.append(("clock.state", {"state": "hold", "prev": prev_state, "held_at": reading}))
            elif self.offset is not None and self.state in ("countdown", "flight", "lost"):
                self.state = "countdown" if t + self.offset < 0 else "flight"
                if self.state != prev_state:
                    ev.append(("clock.state", {"state": self.state, "prev": prev_state}))
            return ev
        self._frozen_since = (t, reading)

        if self.state == "hold":
            # clock moving again -> resume; re-lock immediately on this reading
            self.offset = reading - t
            self._off.clear()
            self.held_value = None
            self.state = "countdown" if reading < 0 else "flight"
            ev.append(("clock.state", {"state": self.state, "prev": "hold", "resumed_at": reading}))
            return ev

        if self.offset is None:
            self._off.append(reading - t)
            if len(self._off) == RELOCK_N and max(self._off) - min(self._off) <= TOL:
                self.offset = sorted(self._off)[len(self._off) // 2]
                self._off.clear()
                ev.append(("clock.lock", {"mission_time": round(t + self.offset, 2)}))
        else:
            resid = reading - (t + self.offset)
            if abs(resid) <= TOL:
                self.offset += 0.2 * resid      # gentle correction
                self._off.clear()
            else:
                self._off.append(reading - t)
                if len(self._off) == RELOCK_N and max(self._off) - min(self._off) <= TOL:
                    old = t + self.offset
                    self.offset = sorted(self._off)[len(self._off) // 2]
                    self._off.clear()
                    ev.append(("clock.jump", {"from": round(old, 2), "to": round(t + self.offset, 2)}))

        mt = self.mission_time(t)
        if mt is not None:
            self.state = "countdown" if mt < 0 else "flight"
        if self.state != prev_state:
            ev.append(("clock.state", {"state": self.state, "prev": prev_state}))
        return ev
