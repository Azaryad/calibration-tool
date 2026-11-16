"""Offline proof/validation harness for the light detector.

Runs the detector on a video + ROI and (optionally) prints the CSV segment
durations it should match. Lets us validate detection against the real GoPro
footage -- especially CamB/CamD, which the previous tool failed on -- before any
UI exists.

Example:
  python cli_detect.py --video ".../CamD-...MP4" --cam CamD --start 540 \
      --csv-dir ".../jobs/73728808c217/input"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lightcalib import ffmpeg_utils, csv_utils, detect

# Operator-drawn ROIs recovered from the previous tool's manifest (static cameras).
KNOWN_ROIS = {
    "CamA": {"x": 867, "y": 981, "w": 63, "h": 44},
    "CamD": {"x": 953, "y": 1013, "w": 72, "h": 42},
    "CamB": {"x": 1290, "y": 965, "w": 79, "h": 44},
    "Sample": {"x": 529, "y": 611, "w": 67, "h": 37},
}


def run_video(video: Path, roi: dict, start: float, scan_fps: float,
              duration: float | None, refine: bool) -> None:
    meta = ffmpeg_utils.probe(video)
    print(f"\n=== {video.name} ===")
    print(f"  {meta.width}x{meta.height} {meta.fps:.3f}fps {meta.vcodec} "
          f"dur={meta.duration:.1f}s  ROI={roi}")
    scanned = detect.scan(video, roi, meta, scan_fps=scan_fps, start_sec=start,
                          duration_sec=duration)
    events, diag = detect.detect_events(scanned, scan_fps=scan_fps)
    print(f"  scanned {len(scanned['t'])} frames @ {scan_fps}fps  "
          f"baseline={diag['baseline']:.4f} peak={diag['peak']:.4f} "
          f"threshold={diag['threshold']:.4f}")
    for w in diag["warnings"]:
        print(f"  ! {w}")
    if not events:
        print("  (no events)")
        return
    print(f"  {len(events)} light event(s):")
    for ev in events:
        t = ev["time"]
        if refine and diag["threshold"] == diag["threshold"]:  # not NaN
            t, _ = detect.refine_onset(video, roi, meta, ev["time"], diag["threshold"])
        print(f"    onset={t:8.3f}s  end={ev['end_time']:8.3f}s  "
              f"bright={ev['bright_frac']:.3f} contrast={ev['contrast']:5.1f} "
              f"compact={ev['compactness']:.2f}")
    # crude pairing preview: consecutive on/off -> candidate segment durations
    onsets = [ (detect.refine_onset(video, roi, meta, ev["time"], diag["threshold"])[0]
                if refine and diag["threshold"] == diag["threshold"] else ev["time"])
               for ev in events ]
    if len(onsets) >= 2:
        print("  candidate segment durations (consecutive onsets):")
        for a, b in zip(onsets, onsets[1:]):
            print(f"    {a:8.3f} -> {b:8.3f}  = {b - a:6.3f}s")


def run_csvs(csv_dir: Path) -> None:
    paths = list(csv_dir.glob("*.csv"))
    pairs, warnings = csv_utils.pair_csv_files(paths)
    print(f"\n=== CSV pairs in {csv_dir} (ordered by modified time) ===")
    for w in warnings:
        print(f"  ! {w}")
    for p in pairs:
        bz = csv_utils.read_sensor_csv(p["buzzer"])
        pulses = csv_utils.detect_pulses(bz.times, bz.values)
        dur = (pulses[1].onset_time - pulses[0].onset_time) if len(pulses) >= 2 else None
        dur_s = f"{dur:.3f}s" if dur else "n/a"
        print(f"  [{p['index']}] {p['buzzer'].name:24s} pulses={len(pulses)} "
              f"segment={dur_s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--cam", choices=sorted(KNOWN_ROIS), help="use a known ROI")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--scan-fps", type=float, default=detect.SCAN_FPS)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--csv-dir", type=Path)
    args = ap.parse_args()

    if args.video:
        if args.roi:
            roi = dict(zip("xywh", args.roi))
        elif args.cam:
            roi = KNOWN_ROIS[args.cam]
        else:
            ap.error("provide --roi or --cam with --video")
        run_video(args.video, roi, args.start, args.scan_fps, args.duration,
                  refine=not args.no_refine)
    if args.csv_dir:
        run_csvs(args.csv_dir)


if __name__ == "__main__":
    main()
