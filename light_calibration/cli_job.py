"""Run the full windowed multi-camera pipeline on the real 3-camera test job.

Usage: python cli_job.py "<folder with the 3 GoPro MP4s and the buzzer/force CSVs>"
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

from lightcalib import ffmpeg_utils, csv_utils, pipeline

if len(sys.argv) < 2:
    sys.exit(__doc__)
BASE = Path(sys.argv[1])
CAM_FILES = {
    "CamA": ("CamA-26-06-22_15.41.33-GX010245.MP4", {"x": 867, "y": 981, "w": 63, "h": 44}),
    "CamB": ("CamB-26-06-22_15.41.34-GX010513.MP4", {"x": 1290, "y": 965, "w": 79, "h": 44}),
    "CamD": ("CamD-26-06-22_15.41.34-GX010626.MP4", {"x": 953, "y": 1013, "w": 72, "h": 42}),
}
TRIM = 540.0


def main() -> None:
    cameras = []
    for cid, (fn, roi) in CAM_FILES.items():
        p = BASE / fn
        cameras.append({"id": cid, "path": p, "meta": ffmpeg_utils.probe(p),
                        "roi": roi, "trim_start": TRIM})

    pairs, _ = csv_utils.pair_csv_files(list(BASE.glob("*.csv")))
    csv_segs = []
    for pr in pairs:
        bz = csv_utils.read_sensor_csv(pr["buzzer"])
        pulses = csv_utils.detect_pulses(bz.times, bz.values)
        if len(pulses) >= 2:
            csv_segs.append({"index": pr["index"], "key_str": "".join(pr["key"]),
                             "duration": pulses[1].onset_time - pulses[0].onset_time})
    print("CSV segments:", [(c["key_str"], round(c["duration"], 3)) for c in csv_segs])

    t0 = time.time()
    results, info = pipeline.analyze_job(cameras, csv_segs, progress=lambda m: print("   .", m))
    print(f"\n==== {info}  elapsed={time.time() - t0:.0f}s ====")
    for s in results:
        flag = "CONFIRMED" if s.confirmed else "unconfirmed"
        print(f"\nseg{s.csv_index} {s.csv_key:9s} csv={s.csv_duration:7.3f}s  [{flag}] {s.warnings}")
        for cid, cr in s.cameras.items():
            if cr.start_time is not None and cr.duration is not None:
                body = f"{cr.start_time:8.3f}->{cr.end_time:8.3f} dur={cr.duration:6.3f} d={cr.delta:+.3f}"
            else:
                body = "not found"
            print(f"     {cid}: {cr.source:10s} {cr.confidence:4s}  {body}  {' '.join(cr.warnings)}")


if __name__ == "__main__":
    main()
