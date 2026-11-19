# Reel Neet Launches

Tools for the Reel Neet launch livestream.

- [`ingest/`](ingest/README.md): OCR ingest that reads the SpaceX webcast overlay (mission clock, booster and ship speed and altitude) and sends vehicle status to the launch runtime hub. Profiles for Falcon 9 and Starship.

Quick start on the Mac:

```
cd ingest
python -m ingest.run --source ffmpeg --input ~/Projects/reel-neet-ocr-test/flight15/flight15_clip.mp4 \
  --profile profiles/spacex_starship.json --fps 2 --jsonl logs/flight15.jsonl
```
