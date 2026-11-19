"""Which vehicle each telemetry slot shows right now.

Falcon 9 keeps stage 1 on the left and stage 2 on the right, so its profile can omit
this. Starship does not (Flight 14 webcast):
  T-8 s .. T+2:18   one pair for the stacked vehicle, on the RIGHT
  T+2:19 .. T+3:15  right = ship (ship altitude replaced by an engine diagram), left = icons
  T+3:16 .. T+7:12  left = SUPER HEAVY, right = STARSHIP
  T+7:13 ..         left = STARSHIP, right = attitude ball / globe

Two sources, in order of trust:
  1. the panel label OCR'd above each slot ("SUPER HEAVY", "STARSHIP"), needing the same
     answer on LABEL_CONFIRM frames in a row before it switches;
  2. a mission-time schedule from the profile (taken from the last flight) as fallback.
     When confirmed labels vanish from every slot for LABEL_GONE frames (Flight 15: the booster
     panel left at ~T+7:00, earlier than Flight 14's 7:13), the schedule jumps to its next row,
     so the ship isn't routed into the booster channel for the seconds in between.

Profile keys:
  "slots":    {"left": {"speed": "s1_speed", "alt": "s1_alt", "label": "left_label"}, "right": {...}}
  "labels":   {"booster": ["SUPER HEAVY", "BOOSTER"], "ship": ["STARSHIP", "SHIP"]}
  "schedule": [{"from": null, "to": 139, "left": null, "right": "stack"}, ...]
Vehicles: "booster" -> s1_* channels, "ship" -> s2_*, "stack" -> both, null -> ignore the slot.
"""
from __future__ import annotations

import re

LABEL_CONFIRM = 2
LABEL_GONE = 6      # frames (3 s at 2 fps) with no label on any slot, after labels were confirmed
VEH_PREFIX = {"booster": ["s1"], "ship": ["s2"], "stack": ["s1", "s2"]}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


class LayoutMapper:
    def __init__(self, profile: dict):
        self.slots: dict = profile.get("slots") or {}
        self.labels = {veh: [_norm(x) for x in words] for veh, words in (profile.get("labels") or {}).items()}
        self.schedule: list = profile.get("schedule") or []
        self.active = bool(self.slots)
        self.current: dict[str, str | None] = {}
        self._pending: dict[str, tuple[str | None, int]] = {}
        self.source: dict[str, str] = {}
        self._had_label = False
        self._no_label = 0
        self._skip_to: float | None = None

    def label_regions(self) -> list[str]:
        return [s["label"] for s in self.slots.values() if s.get("label")]

    def _from_label(self, text: str) -> str | None:
        t = _norm(text)
        if not t:
            return None
        # longest keyword first, so "SUPERHEAVY" is not mistaken for "SHIP"-like fragments
        best = None
        for veh, words in self.labels.items():
            for w in words:
                if w and w in t and (best is None or len(w) > best[1]):
                    best = (veh, len(w))
        return best[0] if best else None

    def _from_schedule(self, slot: str, mt: float | None) -> str | None:
        if mt is None:
            return None
        if self._skip_to is not None:
            mt = max(mt, self._skip_to)
        for row in self.schedule:
            lo = row.get("from")
            hi = row.get("to")
            if (lo is None or mt >= lo) and (hi is None or mt < hi):
                return row.get(slot)
        return None

    def update(self, mt: float | None, label_texts: dict[str, str]) -> list[tuple[str, dict]]:
        """Returns layout.change events when a slot's vehicle changes."""
        ev = []
        any_label = any(self._from_label(label_texts.get(c.get("label", ""), "") or "") for c in self.slots.values())
        if any_label:
            self._no_label = 0
        else:
            self._no_label += 1
            if self._had_label and self._no_label >= LABEL_GONE and mt is not None:
                nxt = [r["from"] for r in self.schedule if r.get("from") is not None and r["from"] > mt]
                if nxt:
                    self._skip_to = min(nxt)
                self._had_label = False
        for slot, cfg in self.slots.items():
            veh, src = None, "schedule"
            lab = label_texts.get(cfg.get("label", ""), "")
            seen = self._from_label(lab) if lab else None
            if seen is not None:
                prev, n = self._pending.get(slot, (None, 0))
                n = n + 1 if prev == seen else 1
                self._pending[slot] = (seen, n)
                if n >= LABEL_CONFIRM or self.current.get(slot) == seen:
                    veh, src = seen, "label"
                    self._had_label = True
                else:
                    veh, src = self.current.get(slot), self.source.get(slot, "schedule")
            else:
                self._pending.pop(slot, None)
                veh = self._from_schedule(slot, mt)
            if veh != self.current.get(slot, "unset"):
                if slot in self.current or veh is not None:
                    ev.append(("layout.change", {"slot": slot, "vehicle": veh, "prev": self.current.get(slot),
                                                 "by": src}))
                self.current[slot] = veh
            self.source[slot] = src
        return ev

    def route(self, nums: dict[str, float | None]) -> dict[str, float | None]:
        """Turn per-box readings into per-vehicle channel readings (s1_* = booster, s2_* = ship)."""
        if not self.active:
            return nums
        out: dict[str, float | None] = {k: None for k in ("s1_speed", "s1_alt", "s2_speed", "s2_alt")}
        # Labels were showing and just dropped out: either an OCR miss or the panels are changing.
        # Route nothing until we know; a blank beats putting the ship's speed on the booster.
        if self._had_label and self._no_label > 0:
            return out
        for slot, cfg in self.slots.items():
            veh = self.current.get(slot)
            for prefix in VEH_PREFIX.get(veh, []):
                for field in ("speed", "alt"):
                    box = cfg.get(field)
                    if box and nums.get(box) is not None and out[f"{prefix}_{field}"] is None:
                        out[f"{prefix}_{field}"] = nums[box]
        return out
