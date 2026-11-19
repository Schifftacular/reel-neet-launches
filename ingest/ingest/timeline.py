"""Cross-check OCR detections against the planned timeline from the runtime rundown.

OCR is the primary source. The timeline only:
  * labels detections and reports how far off-nominal they are, and
  * fills in milestones OCR cannot see (stage sep, fairing sep, payload deploy) or
    missed, marked status="assumed" so graphics can still advance.
"""
from __future__ import annotations

import json

TOL = 10.0       # |detected - nominal| within this -> "nominal"
GRACE = 20.0     # seconds past nominal before an undetected milestone is assumed


def load(path: str) -> list[dict]:
    with open(path) as f:
        doc = json.load(f)
    items = doc["milestones"] if isinstance(doc, dict) else doc
    # milestones without a time (e.g. classified payload deploy) cannot be cross-checked
    return sorted((m for m in items if m.get("t") is not None), key=lambda m: m["t"])


class TimelineFusion:
    def __init__(self, milestones: list[dict], assume: bool = False):
        # When publishing to the runtime hub, the hub owns "assumed" marking; keep assume=False.
        self.assume = assume
        self.plan = {m["id"]: m for m in milestones}
        self.emitted: dict[str, str] = {}   # id -> status

    def on_detection(self, d: dict) -> dict:
        m = self.plan.get(d["id"])
        out = {"id": d["id"], "label": m["label"] if m else d["id"], "t": d["t"],
               "source": "ocr", "confidence": d["confidence"], "evidence": d.get("evidence", {})}
        if m:
            delta = d["t"] - m["t"]
            out.update(nominal_t=m["t"], delta=round(delta, 1),
                       status="detected" if abs(delta) <= TOL else "detected_off_nominal")
        else:
            out["status"] = "detected_unplanned"
        if self.emitted.get(d["id"]) == "assumed":
            out["supersedes"] = "assumed"
        if d.get("update"):
            out["update"] = True
        self.emitted[d["id"]] = out["status"]
        return out

    def tick(self, mt: float | None, clock_state: str) -> list[dict]:
        """Emit assumed milestones whose nominal time + grace has passed. Never during a hold/countdown."""
        if not self.assume or mt is None or clock_state != "flight":
            return []
        out = []
        for mid, m in self.plan.items():
            if mid in self.emitted or m["t"] < 0:
                continue
            # milestones OCR can't observe are assumed right at nominal; others get a grace period
            wait = 0.0 if m.get("ocr") is False else GRACE
            if mt >= m["t"] + wait:
                self.emitted[mid] = "assumed"
                out.append({"id": mid, "label": m["label"], "t": m["t"], "nominal_t": m["t"],
                            "source": "timeline", "status": "assumed",
                            "confidence": 0.5 if m.get("ocr") is False else 0.3})
        return out
