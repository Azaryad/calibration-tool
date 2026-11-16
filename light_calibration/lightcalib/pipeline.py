"""Multi-camera analysis with the windowed cross-camera strategy.

Cameras start within ~1 s of each other, so once we know a flash time from ONE
camera we only need to LOOK at a +/-3 s window in the others instead of scanning
27 minutes three times. Concretely:

  1. Pick the CLEAREST camera (darkest ROI background) as the lead -- detection
     is easiest there, and a clear camera rarely misses a real flash.
  2. Full-scan the lead once; match its events to the CSV segments.
  3. If the lead still left CSV segments unmatched, full-scan the next-clearest
     camera and fill those in (a lead can miss a swim that was occluded for it).
  4. For every other camera, CONFIRM each known flash time with a fallback
     ladder: +/-3 s window (level 1) -> +/-12 s (level 2) -> full scan (level 3).
     Needing level 2/3 means the prediction was off or the camera is struggling,
     so that camera/segment is flagged as a probable problem.
  5. A segment is high-confidence when >=2 cameras agree on its timing.

This also handles the sun case: a clear lead camera finds the flash; the sunlit
camera only has to confirm a KNOWN time, where the saturation-step signal
(see detect.py, SAT_CAP) is reliable even under glare.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import detect, matching
from .ffmpeg_utils import VideoMeta

CONFIRM_BRIGHT = 0.20        # a window confirmation must be at least this bright
CONFIRM_COMPACT = 0.55
WINDOW_LADDER = [(1, 3.0, 12.0), (2, 12.0, 10.0)]   # (level, half_width_s, scan_fps)
AGREE_TOL = 0.5             # cross-camera start agreement (s); clean swims agree to ~0.05s,
                            # so this rejects a camera that locked onto a different event


@dataclass
class CameraResult:
    camera_id: str
    start_time: float | None = None
    end_time: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    duration: float | None = None
    delta: float | None = None          # video duration - csv duration
    source: str = "none"                # "lead" | "window:L1" | "window:L2" | "fullscan" | "none"
    confidence: str = "none"            # high | low | none
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentResult:
    csv_index: int
    csv_key: str
    csv_duration: float
    cameras: dict[str, CameraResult] = field(default_factory=dict)
    confirmed: bool = False             # >=2 cameras agree
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cameras"] = {k: v.as_dict() for k, v in self.cameras.items()}
        return d


# --------------------------------------------------------------------------- #
def probe_condition(video_path: Path, roi: dict, meta: VideoMeta, samples: int = 8) -> dict:
    """Quick ROI-background probe -> clearness (lower ring_hi = clearer) + sun flag."""
    import cv2
    x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
    cap = cv2.VideoCapture(str(video_path))
    fps = meta.fps or 30.0
    his = []
    t0 = max(0.0, meta.duration * 0.1)
    for i in range(samples):
        t = t0 + (meta.duration * 0.8) * i / max(1, samples - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        m = detect.RING_MARGIN_PX
        Y0, Y1 = max(0, y - m), min(fr.shape[0], y + h + m)
        X0, X1 = max(0, x - m), min(fr.shape[1], x + w + m)
        V = cv2.cvtColor(fr[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
        mask = np.ones(V.shape, bool); mask[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
        his.append(float(np.percentile(V[mask], detect.RING_HI_PCT)))
    cap.release()
    ring_hi = float(np.median(his)) if his else 0.0
    return {"ring_hi": ring_hi, "condition": "sun" if ring_hi >= 215 else "clear"}


def _event_strength(ev: dict) -> float:
    return ev["bright_frac"] * max(ev["compactness"], 0.3)


def strongest_event(video_path: Path, roi: dict, meta: VideoMeta, t0: float, t1: float,
                    scan_fps: float) -> dict | None:
    scanned = detect.scan(video_path, roi, meta, scan_fps=scan_fps,
                          start_sec=max(0.0, t0), duration_sec=max(0.5, t1 - t0))
    events, diag = detect.detect_events(scanned, scan_fps=scan_fps)
    events = [e for e in events if e["bright_frac"] >= CONFIRM_BRIGHT and e["compactness"] >= CONFIRM_COMPACT]
    if not events:
        return None
    best = max(events, key=_event_strength)
    if not np.isnan(diag["threshold"]):
        t, frame = detect.refine_onset(video_path, roi, meta, best["time"], diag["threshold"])
        best = {**best, "time": t, "frame": frame}
    return best


def confirm_flash(video_path: Path, roi: dict, meta: VideoMeta, t_expected: float) -> tuple[dict | None, int]:
    """Find the flash near ``t_expected`` using the fallback ladder. Returns
    (event_or_None, level) where level 1 = found in +/-3 s, 2 = +/-12 s,
    3 = needed a full scan, 0 = not found at all."""
    for level, half, fps in WINDOW_LADDER:
        ev = strongest_event(video_path, roi, meta, t_expected - half, t_expected + half, fps)
        if ev is not None:
            return ev, level
    # level 3: full scan, take the strongest event nearest the expected time
    det = detect.detect_in_video(video_path, roi, meta, refine=True)
    cands = [e.as_dict() for e in det.events if e.bright_frac >= CONFIRM_BRIGHT]
    if cands:
        best = min(cands, key=lambda e: abs(e["time"] - t_expected))
        if abs(best["time"] - t_expected) < 30.0:
            return best, 3
    return None, 0


def _confidence(ev: dict | None, level: int) -> str:
    if ev is None:
        return "none"
    strong = ev["bright_frac"] >= matching.STRONG_BRIGHT and ev["compactness"] >= matching.STRONG_COMPACT
    return "high" if (strong and level == 1) else "low"


def _ev_dict(e) -> dict:
    return {"time": e.time, "end_time": e.end_time, "bright_frac": e.bright_frac,
            "compactness": e.compactness, "strength": _event_strength(e.as_dict())}


def _apply_consensus(seg: SegmentResult) -> None:
    """Confirm on the largest set of cameras whose start times cluster within
    AGREE_TOL; downgrade and flag any camera that locked onto a different event."""
    cam_starts = [(cid, cr.start_time) for cid, cr in seg.cameras.items() if cr.start_time is not None]
    if len(cam_starts) < 2:
        return
    best_members: list[str] = []
    for _, t0 in cam_starts:
        members = [cid for cid, t in cam_starts if abs(t - t0) <= AGREE_TOL]
        if len(members) > len(best_members):
            best_members = members
    if len(best_members) >= 2:
        seg.confirmed = True
        for cid, cr in seg.cameras.items():
            if cr.start_time is not None and cid not in best_members:
                cr.confidence = "low"
                cr.warnings.append("Disagrees with the other cameras; likely a wrong event.")
                seg.warnings.append(f"{cid} timing disagrees with the consensus; verify or drop it.")
    else:
        spread = max(t for _, t in cam_starts) - min(t for _, t in cam_starts)
        seg.warnings.append(f"Cameras disagree on the start flash (spread {spread:.2f}s); "
                            f"verify this segment in review before exporting.")


def analyze_job(cameras: list[dict], csv_segs: list[dict], *, scan_fps: float = detect.SCAN_FPS,
                progress=lambda m: None) -> tuple[list[SegmentResult], dict]:
    """``cameras``: list of {id, path (Path), meta (VideoMeta), roi, trim_start}.
    ``csv_segs``: list of {index, key_str, duration}."""
    for cam in cameras:
        cam.setdefault("trim_start", 0.0)
        cam["cond"] = probe_condition(cam["path"], cam["roi"], cam["meta"])
        progress(f"{cam['id']}: {cam['cond']['condition']} (bg {cam['cond']['ring_hi']:.0f})")
    order = sorted(cameras, key=lambda c: c["cond"]["ring_hi"])   # clearest first

    # ---- ONE full scan of the clearest camera, then window the rest -------- #
    lead = order[0]
    progress(f"lead = {lead['id']} (full scan)")
    det = detect.detect_in_video(lead["path"], lead["roi"], lead["meta"],
                                 scan_fps=scan_fps, start_sec=lead["trim_start"], refine=False)
    matches, _ = matching.match_by_pairs([_ev_dict(e) for e in det.events], csv_segs)

    def _refine_pair(cam, t0, t1, thr):
        a, _ = detect.refine_onset(cam["path"], cam["roi"], cam["meta"], t0, thr)
        b, _ = detect.refine_onset(cam["path"], cam["roi"], cam["meta"], t1, thr)
        return a, b

    def _build(c, seed):
        """Seed = (start, end, found_by, confidence). The found_by camera uses the
        seed directly; every other camera CONFIRMS it within a window."""
        t_start, t_end, found_by, base_conf = seed
        seg = SegmentResult(csv_index=c["index"], csv_key=c.get("key_str", str(c["index"])),
                            csv_duration=c["duration"])
        for cam in cameras:
            cr = CameraResult(camera_id=cam["id"])
            if cam["id"] == found_by:
                cr.start_time, cr.end_time = t_start, t_end
                cr.source = "lead" if cam["id"] == lead["id"] else "fullscan"
                cr.confidence = base_conf
            else:
                progress(f"seg{c['index']} confirm in {cam['id']} (~{t_start:.0f}s)")
                s_ev, s_lvl = confirm_flash(cam["path"], cam["roi"], cam["meta"], t_start)
                e_ev, e_lvl = confirm_flash(cam["path"], cam["roi"], cam["meta"], t_end)
                if s_ev and e_ev:
                    cr.start_time, cr.end_time = s_ev["time"], e_ev["time"]
                    lvl = max(s_lvl, e_lvl)
                    cr.source = {1: "window:L1", 2: "window:L2", 3: "fullscan"}[lvl]
                    cr.confidence = _confidence(s_ev, lvl)
                    if lvl >= 2:
                        cr.warnings.append(f"Needed fallback level {lvl} (prediction off / hard camera).")
                else:
                    cr.warnings.append("Flash not confirmed in window.")
            if cr.start_time is not None and cr.end_time is not None:
                cr.duration = cr.end_time - cr.start_time
                cr.delta = cr.duration - c["duration"]
                cr.start_frame = int(round(cr.start_time * (cam["meta"].fps or 30.0)))
                cr.end_frame = int(round(cr.end_time * (cam["meta"].fps or 30.0)))
            seg.cameras[cam["id"]] = cr
        _apply_consensus(seg)
        return seg

    # Seed EVERY matched segment from the lead (high OR low) -- a low-confidence
    # lead match is usually still at roughly the right time, and the windowed
    # confirmation in the other cameras decides whether it holds.
    csv_by_idx = {c["index"]: c for c in csv_segs}
    seg_by_idx: dict[int, SegmentResult] = {}
    for m in matches:
        if m.video is not None:
            a, b = _refine_pair(lead, m.video.start_time, m.video.end_time, det.threshold)
            seg_by_idx[m.csv_index] = _build(csv_by_idx[m.csv_index], (a, b, lead["id"], m.confidence))
    for c in csv_segs:
        seg_by_idx.setdefault(c["index"], SegmentResult(
            csv_index=c["index"], csv_key=c.get("key_str", str(c["index"])), csv_duration=c["duration"],
            warnings=["No event pair matched this segment on the lead camera."]))

    # Escalate ONLY the still-unconfirmed segments: full-scan the next-clearest
    # camera and rebuild just those. A clean job is exactly ONE full scan +
    # windows; extra full scans happen only where the cameras don't yet agree.
    full_scans = 1
    for second in order[1:]:
        unconfirmed = [i for i, s in seg_by_idx.items() if not s.confirmed]
        if not unconfirmed:
            break
        need = [csv_by_idx[i] for i in unconfirmed]
        progress(f"{len(need)} unconfirmed -> full scan {second['id']}")
        full_scans += 1
        det2 = detect.detect_in_video(second["path"], second["roi"], second["meta"],
                                      scan_fps=scan_fps, start_sec=second["trim_start"], refine=False)
        m2, _ = matching.match_by_pairs([_ev_dict(e) for e in det2.events], need)
        for m in m2:
            if m.video is None:
                continue
            a, b = _refine_pair(second, m.video.start_time, m.video.end_time, det2.threshold)
            cand = _build(csv_by_idx[m.csv_index], (a, b, second["id"], m.confidence))
            old = seg_by_idx[m.csv_index]
            n_old = sum(1 for cr in old.cameras.values() if cr.start_time is not None)
            n_new = sum(1 for cr in cand.cameras.values() if cr.start_time is not None)
            if cand.confirmed or (not old.confirmed and n_new > n_old):
                seg_by_idx[m.csv_index] = cand

    results = [seg_by_idx[c["index"]] for c in csv_segs]
    info = {"lead": lead["id"], "segments": len(csv_segs), "full_scans": full_scans,
            "confirmed": sum(1 for s in results if s.confirmed)}
    return results, info
