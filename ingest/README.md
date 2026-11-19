# Launch OCR ingest

Reads the SpaceX webcast overlay (mission clock, stage 1 and stage 2 speed and altitude) and turns it into a live vehicle-status feed for the launch runtime hub. SpaceX has no public data API, so this is the primary source of truth for the clock and vehicle state. The planned timeline from the runtime rundown is used only as a cross-check.

```
OBS program feed / VOD / HLS ──frames @2 fps──► OCR 5 regions ──► clock model ──┐
                                                                 telemetry filter ─► vehicle tracker ─► milestones ──► hub UDP :4101
                                                     planned timeline (cross-check) ─┘
```

## What it emits

Contract v0.1, shared with the runtime (`../launch-runtime/CONTRACT.md`). One JSON object per line:

| type | when | data |
|---|---|---|
| `clock.tick` | 1 Hz | `state` (countdown, hold, flight, lost, no_clock), `display` ("T- 00:01:23"), `confidence` |
| `clock.lock`, `clock.state`, `clock.jump` | on change | hold detected, resumed, recycle or stream skip |
| `telemetry` | 1 Hz per stage | `stage`, `speed_kmh`, `alt_km`, `accel_kmh_s`, `q_kpa`, `phase` |
| `vehicle.status` | phase change | `stage` s1/s2, `phase` (prelaunch, ascent, coast, boostback, entry_burn, descent, landing_burn, landed; s2: attached, burn, coast) |
| `milestone` | detected | `id`, `label`, `t` (mission seconds), `nominal_t`, `delta`, `status`, `confidence`, `evidence` |
| `milestone.confirm` | later evidence | same id as an earlier milestone with higher `confidence` (e.g. liftoff confirmed by stage 1 speed). Never re-sent as `milestone`, so the hub doesn't re-anchor T-0 twice |
| `ingest.health` | every 10 s | average OCR confidence per region, clock lock |

Milestones it can detect from the numbers: liftoff (clock crossing zero, confirmed by stage 1 speed), Max-Q (peak of estimated dynamic pressure), MECO (stage 1 speed peak), SES-1 (stage 2 starts accelerating), boostback start/end, entry burn start/end, landing burn start, landing, SECO-1. Stage sep, fairing sep and payload deploy can't be seen in the numbers, so the hub marks those as `assumed` at planned time.

## How it stays honest

- **Clock**: the on-screen clock is read every frame, but the published time comes from a locked offset that OCR only nudges. One bad read never moves the clock. A display that stops advancing for 3.5 s during the count is a **hold**; when it moves again it re-locks. Three agreeing reads that disagree with the lock trigger a **jump** (recycle, stream skip).
- **Telemetry**: readings that imply impossible rates (more than about 10 g, or 12 km/s of altitude change) are rejected, unless three in a row agree, which means the filter was wrong. Speed fields treat `.` and `,` as thousands separators because OCR swaps them.

## Install (home machine)

```
# macOS
brew install tesseract ffmpeg
# Ubuntu/Debian
sudo apt install tesseract-ocr ffmpeg
# Windows: install Tesseract from the UB Mannheim build and ffmpeg, add both to PATH

pip install opencv-python-headless pytesseract numpy pillow
pip install obsws-python   # only for the OBS source
```

## Run

Live from OBS (recommended: OCR reads exactly what you put on air, with no second copy of the stream). Enable OBS WebSocket (Tools > WebSocket Server Settings) and use the name of the source carrying the webcast:

```
python -m ingest.run --source obs --input "SpaceX Webcast" --fps 2 --hub \
  --timeline ../launch-runtime/out/ussf-317.timeline.json --jsonl logs/ussf-317-ocr.jsonl
```

From a file, HLS URL or capture device:

```
python -m ingest.run --source ffmpeg --input vod.mp4 --fps 2 --hub --timeline ../launch-runtime/out/ussf-317.timeline.json
```

Rehearse against the hub with a recorded webcast. Add `--realtime` so frames come at playback speed; without it the ingest reads the file as fast as it can and the hub would chase the clock:

```
python -m ingest.run --source ffmpeg --input flight15.mp4 --realtime --fps 2 --hub \
  --profile profiles/spacex_starship.json --timeline ../launch-runtime/out/starship-flight-15.timeline.json
```

Watch it work: `--preview` serves a live view at http://127.0.0.1:8090/ showing each frame the ingest reads, every OCR box (green = read, amber = read but rejected, red = nothing), the parsed values, which vehicle each box is routed to, the locked clock and the latest milestones. Pair it with `--realtime` so it plays at normal speed:

```
python -m ingest.run --source ffmpeg --input flight15_clip.mp4 --realtime --preview --fps 2 \
  --profile profiles/spacex_starship.json --timeline timelines/starship-flight-15.timeline.json
```

Rehearse graphics without a stream by replaying a recorded run into the hub:

```
python -m ingest.replay logs/ussf-317-ocr.jsonl --udp 127.0.0.1:4101
```

## Before Tuesday: calibrate on a real webcast (must do)

The region boxes in `profiles/spacex_falcon.json` are starting guesses and the synthetic test overlay was drawn to match them. The real SpaceX overlay will be in slightly different places, and the font differs.

1. Download a recent Falcon 9 webcast VOD (a Vandenberg droneship mission is closest to USSF-317).
2. Grab a frame during ascent where both stages show numbers: `ffmpeg -ss 00:25:00 -i vod.mp4 -frames:v 1 frame.png`
3. `python -m ingest.calibrate frame.png` prints what each box reads and writes `calib/boxes.png`. Nudge the x, y, w, h numbers until every field reads cleanly.
4. Run the whole VOD through `ingest.run --source ffmpeg` and check that the milestones line up with what you see. That is the first test against real footage; until then, the detection thresholds have only been tuned on synthetic frames.

## Tests

`python -m tests.test_synthetic --fps 2 --timeline ../launch-runtime/out/ussf-317.timeline.json` renders a generic launch overlay (countdown with a 20 s hold, ascent, droneship landing, SECO, JPEG artifacts, noise, a garbled frame every 37) and runs the full pipeline. Last result (2 fps media, 4 cores, 3 fps processing): every field read correctly on 97 to 100% of frames, clock error median 0.00 s and p95 0.05 s (worst 2.9 s, during the 3.5 s it takes to notice a hold). Every milestone landed within 4 s of the simulated truth. `samples/synthetic-ussf317-events.jsonl` is that run's output, and replaying it into the hub gave a re-anchored clock, a detected hold and resume, liftoff, and live telemetry on `/state`.

## Real-footage test (Starship webcast, 2026-11-15)

Run on Mike's iMac against a downloaded Starship flight webcast (1,630 frames at 2 fps, 7 frames/s processing on an M4). Calibrated profile: `profiles/spacex_starship.json`.

- Speed and altitude: 92 to 97% of visible frames read, with no wrong digits. Every miss was a blank.
- Clock: 61% at first, 98.6% after fixes. Causes: correct reads scored at confidence 0 were thrown away, zeros came back as O/Q, and the small "+" was often lost (now inferred from the locked clock). The clock has no confidence cut-off; the parser and the clock lock reject junk. The clock state now flips to flight on the model's own T+0.
- Number fields keep the 0.4 confidence cut-off. An A/B run without it read 98 to 100% of visible frames but produced about 290 wrong speeds (engine diagrams read as '2', '72', '8'; dropped leading digits like 26066 read as 6066). With it: 0 to 1 wrong numbers. A wrong number on air is worse than a missing one.
- The clock model caught both video jump cuts within about 1 to 2.5 s.
- Altitude fields require a decimal below 100 km (`decimal_below`), which caught most icon reads there.
- **The Starship bar rearranges by phase.** At T+7:13 the ship moves into the left boxes where the booster was, and some fields are replaced by engine diagrams for minutes at a time. A fixed box-to-vehicle mapping is wrong for Starship; a per-phase mapping is still to do. The Falcon 9 layout still needs its own check on a Falcon VOD.

## Known gaps

- Overlay milestone text ("MAX-Q", "MECO" on SpaceX's timeline arc) isn't read yet. The timeline already carries `ocr_text` for it, so it is the next cross-check to add.
- When SpaceX cuts to a full-screen camera without the overlay, the clock goes `lost` after 20 s and the hub keeps counting on the last T-0. Telemetry simply pauses.
- Classified payloads: SpaceX usually ends coverage after landing, so SECO-1 and deploy may never be seen.
