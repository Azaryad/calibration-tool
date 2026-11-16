"""Light-event detection.

Signal (per sampled frame, inside the user ROI):
  * ``bright_frac`` -- fraction of ROI pixels brighter than the local
    background ring by a margin. White lamp on -> large; off -> ~0. This is the
    key change from the previous tool, which thresholded raw brightness and so
    drowned in sunlit caustics that also hit 255.
  * ``contrast``    -- ROI top-10% value minus ring top-10% value (secondary).
  * ``compactness`` -- largest-connected-blob / bright pixels. The lamp is one
    disc (~1.0); scattered caustic sparkle is low. Used to reject false frames.

Thresholding on the ``bright_frac`` time series is adaptive AND hard-clamped
below the observed peak, so it can never exceed the maximum score (the bug that
produced threshold=320 > 255 and zero detections on CamB/CamD)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import ffmpeg_utils
from .ffmpeg_utils import VideoMeta


# ----- tunables (all overridable per call) --------------------------------- #
RING_MARGIN_PX = 46          # ring half-width around the ROI (background sample)
BRIGHT_MARGIN = 38.0         # a pixel is "lamp-bright" if V >= ring_hi + this
SAT_CAP = 248.0              # ...but cap the threshold here: under sun glare the ring
                             # is already ~228, so ring_hi+margin would exceed 255 and
                             # the saturated lamp (255) would never register. Capping
                             # makes the SAME signal work in shadow AND sun.
RING_HI_PCT = 92.0           # percentile of ring V treated as "bright background"
THRESH_FRAC = 0.40           # event threshold = baseline + frac*(peak-baseline)
MIN_BRIGHT_FRAC = 0.03       # absolute floor: need >=3% of ROI lit to count
MIN_CONTRAST = 0.04          # if peak bright_frac never clears this -> no light
SCAN_FPS = 8.0
MERGE_GAP_SEC = 0.6
MIN_ON_SEC = 0.12
MIN_SEPARATION_SEC = 4.0
MIN_COMPACTNESS = 0.35       # peak-frame blob must be at least this compact


@dataclass
class LightEvent:
    index: int
    time: float              # onset (leading edge), seconds in video time
    end_time: float
    frame: int
    end_frame: int
    bright_frac: float       # peak bright fraction in the event
    contrast: float          # peak ROI-vs-ring contrast
    compactness: float       # peak blob compactness
    refined: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    roi: dict
    events: list[LightEvent]
    baseline: float
    peak: float
    threshold: float
    scan_fps: float
    frames_scanned: int
    warnings: list[str]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [e.as_dict() for e in self.events]
        return d


# --------------------------------------------------------------------------- #
def _crop_box(roi: dict, meta: VideoMeta, ring: int) -> tuple[int, int, int, int, int, int]:
    """Return (X0, Y0, W, H, rx, ry): the expanded scan crop and the ROI's
    offset inside it. Even dims keep ffmpeg's crop/pixfmt happy."""
    x, y = int(roi["x"]), int(roi["y"])
    w, h = int(roi["w"]), int(roi["h"])
    X0 = max(0, x - ring)
    Y0 = max(0, y - ring)
    X1 = min(meta.width, x + w + ring)
    Y1 = min(meta.height, y + h + ring)
    W = (X1 - X0) & ~1
    H = (Y1 - Y0) & ~1
    return X0, Y0, W, H, x - X0, y - Y0


def _top_mean(values: np.ndarray, frac: float = 0.10) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(round(flat.size * frac)))
    return float(np.partition(flat, flat.size - k)[-k:].mean())


def frame_features(crop_bgr: np.ndarray, rx: int, ry: int, w: int, h: int) -> tuple[float, float, float]:
    """Return (bright_frac, contrast, compactness) for one cropped frame."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    roiV = V[ry:ry + h, rx:rx + w]
    if roiV.size == 0:
        return 0.0, 0.0, 0.0

    ring_mask = np.ones(V.shape, dtype=bool)
    ring_mask[ry:ry + h, rx:rx + w] = False
    ringV = V[ring_mask]
    ring_hi = float(np.percentile(ringV, RING_HI_PCT)) if ringV.size else 0.0

    thresh = min(ring_hi + BRIGHT_MARGIN, SAT_CAP)
    bright = (roiV >= thresh)
    bright_count = int(bright.sum())
    bright_frac = bright_count / roiV.size
    contrast = _top_mean(roiV) - _top_mean(ringV)

    compactness = 0.0
    if bright_count > 0:
        n_labels, labels = cv2.connectedComponents(bright.astype(np.uint8), connectivity=8)
        if n_labels > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0  # background
            compactness = float(sizes.max()) / bright_count
    return float(bright_frac), float(contrast), float(compactness)


def scan(video_path: Path, roi: dict, meta: VideoMeta, *, scan_fps: float = SCAN_FPS,
         start_sec: float = 0.0, duration_sec: float | None = None,
         ffmpeg: str | None = None) -> dict:
    """Coarse scan of the whole (post-trim) range. Returns time + feature arrays."""
    X0, Y0, W, H, rx, ry = _crop_box(roi, meta, RING_MARGIN_PX)
    w, h = int(roi["w"]), int(roi["h"])
    times: list[float] = []
    bfrac: list[float] = []
    contr: list[float] = []
    compc: list[float] = []
    for t, crop in ffmpeg_utils.scan_region(video_path, X0, Y0, W, H, scan_fps,
                                             start_sec, duration_sec, ffmpeg):
        bf, ct, cp = frame_features(crop, rx, ry, w, h)
        times.append(t)
        bfrac.append(bf)
        contr.append(ct)
        compc.append(cp)
    return {
        "t": np.array(times), "bright": np.array(bfrac),
        "contrast": np.array(contr), "compact": np.array(compc),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    out = [runs[0]]
    for s, e in runs[1:]:
        if s - out[-1][1] - 1 <= gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def detect_events(scanned: dict, *, scan_fps: float = SCAN_FPS,
                  merge_gap_sec: float = MERGE_GAP_SEC, min_on_sec: float = MIN_ON_SEC,
                  min_sep_sec: float = MIN_SEPARATION_SEC) -> tuple[list[dict], dict]:
    """Adaptive, peak-clamped thresholding -> candidate light events."""
    scores = scanned["bright"]
    n = len(scores)
    if n == 0:
        return [], {"baseline": 0.0, "peak": 0.0, "threshold": 0.0, "warnings": ["No frames scanned."]}

    baseline = float(np.median(scores))
    k = max(3, int(round(n * 0.005)))
    peak_robust = float(np.mean(np.sort(scores)[-k:]))
    peak = float(scores.max())
    contrast = peak_robust - baseline

    warnings: list[str] = []
    if peak < MIN_CONTRAST or contrast <= 0:
        return [], {"baseline": baseline, "peak": peak, "threshold": float("nan"),
                    "warnings": ["No clear light in this ROI (peak bright-fraction too low)."]}

    threshold = baseline + THRESH_FRAC * contrast
    threshold = min(threshold, 0.9 * peak_robust)      # *** never exceed the peak ***
    threshold = max(threshold, baseline + 0.01, MIN_BRIGHT_FRAC)

    gap = max(1, int(round(merge_gap_sec * scan_fps)))
    min_on = max(1, int(round(min_on_sec * scan_fps)))
    min_sep = max(1, int(round(min_sep_sec * scan_fps)))

    runs = _merge(_runs(scores >= threshold), gap)
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_on]

    events: list[dict] = []
    last_onset = None
    for s, e in runs:
        if last_onset is not None and s - last_onset < min_sep:
            continue
        seg = slice(s, e + 1)
        pk = int(s + int(np.argmax(scores[seg])))
        events.append({
            "onset_idx": s, "end_idx": e,
            "time": float(scanned["t"][s]), "end_time": float(scanned["t"][e]),
            "bright_frac": float(scores[seg].max()),
            "contrast": float(scanned["contrast"][seg].max()),
            "compactness": float(scanned["compact"][pk]),
        })
        last_onset = s

    kept = [ev for ev in events if ev["compactness"] >= MIN_COMPACTNESS]
    if len(kept) < len(events):
        warnings.append(f"Rejected {len(events) - len(kept)} low-compactness (caustic-like) event(s).")
    return kept, {"baseline": baseline, "peak": peak, "threshold": float(threshold),
                  "warnings": warnings}


def refine_onset(video_path: Path, roi: dict, meta: VideoMeta, approx_time: float,
                 threshold: float, *, window_sec: float = 0.6) -> tuple[float, int]:
    """Frame-accurate onset: re-measure at full fps in a small window around the
    coarse hit (cv2 POS_FRAMES seek is exact), return first frame >= threshold."""
    fps = meta.fps or 30.0
    X0, Y0, W, H, rx, ry = _crop_box(roi, meta, RING_MARGIN_PX)
    w, h = int(roi["w"]), int(roi["h"])
    f_start = max(0, int(round((approx_time - window_sec) * fps)))
    f_end = int(round((approx_time + window_sec) * fps))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    fn = f_start
    hit_frame = None
    try:
        while fn <= f_end:
            ok, fr = cap.read()
            if not ok:
                break
            crop = fr[Y0:Y0 + H, X0:X0 + W]
            bf, _, _ = frame_features(crop, rx, ry, w, h)
            if bf >= threshold:
                hit_frame = fn
                break
            fn += 1
    finally:
        cap.release()
    if hit_frame is None:
        return approx_time, int(round(approx_time * fps))
    return hit_frame / fps, hit_frame


def detect_in_video(video_path: Path, roi: dict, meta: VideoMeta, *,
                    scan_fps: float = SCAN_FPS, start_sec: float = 0.0,
                    refine: bool = True, ffmpeg: str | None = None) -> DetectionResult:
    scanned = scan(video_path, roi, meta, scan_fps=scan_fps, start_sec=start_sec, ffmpeg=ffmpeg)
    raw_events, diag = detect_events(scanned, scan_fps=scan_fps)

    events: list[LightEvent] = []
    for i, ev in enumerate(raw_events, 1):
        t, frame = ev["time"], int(round(ev["time"] * (meta.fps or 30.0)))
        refined = False
        if refine and not np.isnan(diag["threshold"]):
            t, frame = refine_onset(video_path, roi, meta, ev["time"], diag["threshold"])
            refined = True
        events.append(LightEvent(
            index=i, time=t, end_time=ev["end_time"], frame=frame,
            end_frame=int(round(ev["end_time"] * (meta.fps or 30.0))),
            bright_frac=ev["bright_frac"], contrast=ev["contrast"],
            compactness=ev["compactness"], refined=refined,
        ))
    return DetectionResult(
        roi={k: int(roi[k]) for k in ("x", "y", "w", "h")},
        events=events, baseline=diag["baseline"], peak=diag["peak"],
        threshold=diag["threshold"], scan_fps=scan_fps,
        frames_scanned=len(scanned["t"]), warnings=diag["warnings"],
    )
