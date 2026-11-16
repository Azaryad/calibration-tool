# Light Calibration Tool

Splits long underwater swim videos into per-swim segments using a physical
**white lamp** that flashes at the start and end of each swim — visible in every
camera and recorded as a voltage pulse on CSV channel 1 (still named `buzzer`).
For each swim it exports the trimmed video clips plus the matching force/buzzer
CSVs, aligned and equal-length.

This is a clean rebuild that fixes the failures of the previous attempt.

## Why the previous version failed (and what changed)

| Previous failure | Fix here |
|---|---|
| Thresholded **raw brightness** → sunlit caustics also hit 255, so CamA caught garbage (7 s "segments") | Signal is **locally-bright-pixel area** in the ROI relative to the surrounding ring (`detect.frame_features`) |
| Adaptive threshold added a robust-sigma term → computed **threshold 320 > 255**, so CamB/CamD detected **nothing** | Threshold is **hard-clamped below the observed peak** (`detect.detect_events`) |
| Scattered caustic glints counted as light | Require a **compact blob** (one disc), via connected-component compactness |
| Greedy "events 1&2 = segment 1" positional matching | **Global duration assignment** (Hungarian), order-independent (`matching.align`) |
| Fixed CSV band `0.002–0.0025` (high never reached) | Validated **0.002 crossing** primary + **MAD-gated adaptive** fallback (`csv_utils.detect_pulses`) |

## Validated on the real data (the cameras that previously failed)

- **CamD** (previously 0 events): clean onsets, baseline 0.000, threshold properly
  clamped, compactness ~0.98.
- **CSV**: all 5 swim pairs → exactly 2 pulses, distinct durations.
- **Matcher**: drops a false flash and separates near-duplicate durations
  (31.344 vs 31.401 s).
- **Auto-ROI**: locates the lamp automatically within a few px of the manual ROI.

## Layout

```
lightcalib/
  ffmpeg_utils.py  ffmpeg/ffprobe discovery, metadata, region streaming, export
  csv_utils.py     pairing (by modified time), parsing, adaptive pulse detection
  detect.py        the light detector (bright-disc area + clamped adaptive threshold)
  matching.py      video-segment ↔ CSV-segment duration assignment
  roi_assist.py    auto-locate the lamp for the ROI step
cli_detect.py      offline proof/validation harness (no UI)
cli_job.py         run the full multi-camera pipeline on one folder (no UI)
app.py             Flask app: index → pairs → trim → roi → detect → review → export
```

## Run the app

Windows: double-click `start_light_calibration.bat` in the repo root. Or manually:

```bash
pip install -r requirements.txt          # needs ffmpeg/ffprobe on PATH
python app.py                            # then open http://127.0.0.1:5001
```

## Offline validation harness

```bash
pip install -r requirements.txt          # needs ffmpeg/ffprobe on PATH
python cli_detect.py --csv-dir "<folder of buzzer/force CSVs>"
python cli_detect.py --video "<CamD ...mp4>" --cam CamD --start 540 \
    --csv-dir "<csv folder>"
```

Cameras are static, so one ROI per camera holds for the whole recording. Use
`--cam {CamA,CamB,CamD,Sample}` for the recovered test ROIs, or `--roi X Y W H`.
