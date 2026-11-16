"""Auto-locate the calibration lamp so the user only has to confirm/nudge the
ROI instead of hunting for an OFF lamp in a 27-minute clip.

Idea: the lamp is a COMPACT, WHITE-SATURATED disc that appears in a FIXED spot
during brief flashes. So over a short search window we look for pixels that go
white-saturated (high value, low colour saturation), accumulate how often each
does, and pick the most compact, blob-like cluster in the lower part of the
frame (the lamp sits near the pool floor). Moving swimmers and drifting caustics
don't form a single fixed compact blob, so they score low."""
from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np

from . import ffmpeg_utils
from .ffmpeg_utils import VideoMeta

WHITE_V = 235          # "bright" value channel
WHITE_S = 70           # "white" = low colour saturation
LOWER_FRACTION = 0.45  # only search the lower (1 - this) of the frame


def _scan_full(video_path: Path, meta: VideoMeta, scan_fps: float, down_w: int,
               start_sec: float, duration_sec: float, ffmpeg: str) -> tuple[np.ndarray, float]:
    """Stream downscaled full frames; return a white-saturation hit-count map."""
    scale = down_w / meta.width
    dh = int(round(meta.height * scale)) & ~1
    vf = f"fps={scan_fps},scale={down_w}:{dh}"
    cmd = [ffmpeg, "-v", "error"]
    if ffmpeg_utils.HWACCEL:
        cmd += ["-hwaccel", ffmpeg_utils.HWACCEL]
    cmd += ["-ss", f"{start_sec:.3f}", "-i", str(video_path),
            "-t", f"{duration_sec:.3f}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = down_w * dh * 3
    counts = np.zeros((dh, down_w), dtype=np.int32)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            img = np.frombuffer(buf, np.uint8).reshape(dh, down_w, 3)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white = (hsv[:, :, 2] >= WHITE_V) & (hsv[:, :, 1] <= WHITE_S)
            counts += white.astype(np.int32)
    finally:
        proc.stdout.close()
        proc.wait()
    return counts, scale


def locate_light(video_path: Path, meta: VideoMeta, *, start_sec: float = 0.0,
                 duration_sec: float = 120.0, scan_fps: float = 6.0, down_w: int = 640,
                 ffmpeg: str | None = None) -> dict | None:
    """Return a best-guess ROI ``{x,y,w,h}`` in full-res coords, or None."""
    ffmpeg = ffmpeg or ffmpeg_utils.locate_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found.")
    duration_sec = min(duration_sec, max(1.0, meta.duration - start_sec))
    counts, scale = _scan_full(video_path, meta, scan_fps, down_w, start_sec, duration_sec, ffmpeg)
    dh, dw = counts.shape

    # restrict to the lower part of the frame (lamp sits near the floor)
    y_min = int(dh * LOWER_FRACTION)
    region = counts.copy()
    region[:y_min, :] = 0
    if region.max() < 2:                       # never saw a convincing white flash
        return None

    mask = (region >= max(2, int(region.max() * 0.5))).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_score, best = -1.0, None
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        if area < 3:
            continue
        fill = area / float(w * h)             # disc-like blobs fill their bbox well
        hits = int(region[labels == lbl].max())
        score = hits * (0.4 + 0.6 * fill)
        if score > best_score:
            best_score, best = score, (x, y, w, h)
    if best is None:
        return None

    x, y, w, h = best
    pad_w, pad_h = max(4, int(w * 0.4)), max(4, int(h * 0.4))
    fx = int((x - pad_w) / scale)
    fy = int((y - pad_h) / scale)
    fw = int((w + 2 * pad_w) / scale)
    fh = int((h + 2 * pad_h) / scale)
    fx = max(0, min(fx, meta.width - 2))
    fy = max(0, min(fy, meta.height - 2))
    fw = max(2, min(fw, meta.width - fx))
    fh = max(2, min(fh, meta.height - fy))
    return {"x": fx, "y": fy, "w": fw, "h": fh}
