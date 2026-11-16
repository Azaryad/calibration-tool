"""Match detected video light-events to CSV swim segments by duration.

The previous tool assigned events positionally (events 1&2 = segment 1, ...),
which breaks on a single false flash or a missed event. And we can't assume the
CSV file order (by modified time) equals the video's chronological order -- in
the real data the first filmed swim corresponds to the *last* CSV by mtime. So
matching is purely by duration:

  1. Pair consecutive video onsets into candidate segments whose span is
     swim-sized (drops isolated false flashes between swims).
  2. Globally assign candidate video segments to CSV segments by closest
     duration (Hungarian / optimal assignment), each used at most once. CSV
     segments left unassigned are reported as missing for that camera; extra
     video segments are reported as skipped. Near-equal CSV durations are
     flagged as ambiguous so the reviewer can double-check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

INF = float("inf")
AMBIGUOUS_GAP = 0.15        # CSV durations closer than this are hard to tell apart


@dataclass
class VideoSegment:
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


STRONG_BRIGHT = 0.30        # endpoint looks like the real saturated lamp disc
STRONG_COMPACT = 0.78       # brightness is the main discriminator; compactness is a softer guard


@dataclass
class Match:
    csv_index: int
    csv_key: str
    csv_duration: float
    video: VideoSegment | None = None        # None -> no video pair matched
    delta: float | None = None               # video_duration - csv_duration
    confidence: str = "none"                 # high | low | none
    warnings: list[str] = field(default_factory=list)


def pair_events_to_segments(onsets: list[float], *, min_swim: float = 15.0,
                            max_swim: float = 70.0) -> tuple[list[VideoSegment], int]:
    """Greedily pair consecutive onsets into swim-sized segments. Returns the
    segments and the count of dropped (unpaired/false) onsets."""
    segs: list[VideoSegment] = []
    dropped = 0
    i = 0
    n = len(onsets)
    while i < n - 1:
        dur = onsets[i + 1] - onsets[i]
        if min_swim <= dur <= max_swim:
            segs.append(VideoSegment(onsets[i], onsets[i + 1]))
            i += 2
        else:
            dropped += 1                      # onset i is not a swim start; skip it
            i += 1
    if i == n - 1:
        dropped += 1
    return segs, dropped


def align(video_segs: list[VideoSegment], csv_segs: list[dict], *,
          tol: float = 2.5) -> tuple[list[Match], list[VideoSegment]]:
    """Global least-duration-error assignment. ``csv_segs`` items need
    ``index``, ``duration`` (and optionally ``key_str``). Returns one Match per
    CSV segment (in input order) plus the list of extra/unassigned video
    segments."""
    matches = [Match(csv_index=c["index"], csv_key=c.get("key_str", str(c["index"])),
                     csv_duration=c["duration"]) for c in csv_segs]
    m, n = len(video_segs), len(csv_segs)
    used_video: set[int] = set()
    if m and n:
        cost = np.full((m, n), 1e6)
        for i, v in enumerate(video_segs):
            for j, c in enumerate(csv_segs):
                d = abs(v.duration - c["duration"])
                if d <= tol:
                    cost[i, j] = d
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < 1e6:
                matches[j].video = video_segs[i]
                matches[j].delta = video_segs[i].duration - csv_segs[j]["duration"]
                used_video.add(i)

    # ambiguity flag: CSV durations that sit within AMBIGUOUS_GAP of another
    durs = [c["duration"] for c in csv_segs]
    for j, mtch in enumerate(matches):
        if mtch.video is None:
            mtch.warnings.append("No video event pair matched this segment.")
            continue
        near = [k for k in range(n) if k != j and abs(durs[k] - durs[j]) < AMBIGUOUS_GAP]
        if near and abs(mtch.delta) > AMBIGUOUS_GAP / 2:
            mtch.warnings.append(
                f"Duration close to segment(s) {[csv_segs[k]['index'] for k in near]}; verify assignment.")
    extras = [v for i, v in enumerate(video_segs) if i not in used_video]
    return matches, extras


def match_camera(onsets: list[float], csv_segs: list[dict], *, tol: float = 2.5,
                 min_swim: float = 15.0, max_swim: float = 70.0) -> tuple[list[Match], dict]:
    segs, dropped = pair_events_to_segments(sorted(onsets), min_swim=min_swim, max_swim=max_swim)
    matches, extras = align(segs, csv_segs, tol=tol)
    info = {"video_segments": len(segs), "dropped_onsets": dropped, "extra_segments": len(extras),
            "matched": sum(1 for m in matches if m.video is not None)}
    return matches, info


def match_by_pairs(events: list[dict], csv_segs: list[dict], *, tol: float = 2.0,
                   min_strength: float = 0.0) -> tuple[list[Match], dict]:
    """Robust matcher for messy recordings (many false flashes, multi-press
    clusters). Instead of pairing consecutive events, consider EVERY pair of
    events whose span matches some CSV segment's duration, then greedily accept
    the best (smallest duration error, strongest endpoints) pairs without
    reusing any event or CSV segment.

    ``events`` items need ``time`` and a ``strength`` in [0,1] (e.g. bright_frac
    clipped); pass ``compactness`` too if available for the strength blend."""
    ev = [e for e in sorted(events, key=lambda e: e["time"]) if e.get("strength", 1.0) >= min_strength]
    n = len(ev)

    candidates: list[tuple[float, int, int, int]] = []   # (cost, i, j, csv_pos)
    for i in range(n):
        for j in range(i + 1, n):
            dur = ev[j]["time"] - ev[i]["time"]
            for cpos, c in enumerate(csv_segs):
                if abs(dur - c["duration"]) <= tol:
                    strength = ev[i].get("strength", 1.0) + ev[j].get("strength", 1.0)
                    cost = abs(dur - c["duration"]) - 0.25 * strength   # prefer strong, exact
                    candidates.append((cost, i, j, cpos))
    candidates.sort(key=lambda t: t[0])

    matches = [Match(csv_index=c["index"], csv_key=c.get("key_str", str(c["index"])),
                     csv_duration=c["duration"]) for c in csv_segs]
    def is_strong(e: dict) -> bool:
        return e.get("bright_frac", 1.0) >= STRONG_BRIGHT and e.get("compactness", 1.0) >= STRONG_COMPACT

    used_ev: set[int] = set()
    used_csv: set[int] = set()
    chosen_pairs = 0
    for cost, i, j, cpos in candidates:
        if cpos in used_csv or i in used_ev or j in used_ev:
            continue
        v = VideoSegment(ev[i]["time"], ev[j]["time"])
        matches[cpos].video = v
        matches[cpos].delta = v.duration - csv_segs[cpos]["duration"]
        matches[cpos].confidence = "high" if (is_strong(ev[i]) and is_strong(ev[j])) else "low"
        used_csv.add(cpos); used_ev.add(i); used_ev.add(j)
        chosen_pairs += 1

    durs = [c["duration"] for c in csv_segs]
    for pos, mtch in enumerate(matches):
        if mtch.video is None:
            mtch.warnings.append("No event pair matched this segment's duration.")
            continue
        if mtch.confidence == "low":
            mtch.warnings.append("Matched on faint events (possible swimmer/caustic); "
                                 "confirm against another camera.")
        near = [k for k in range(len(durs)) if k != pos and abs(durs[k] - durs[pos]) < AMBIGUOUS_GAP]
        if near and abs(mtch.delta) > AMBIGUOUS_GAP / 2:
            mtch.warnings.append(
                f"Duration close to segment(s) {[csv_segs[k]['index'] for k in near]}; verify.")
    high = sum(1 for m in matches if m.confidence == "high")
    info = {"events": n, "matched": chosen_pairs, "high_confidence": high,
            "unmatched_segments": len(csv_segs) - chosen_pairs}
    return matches, info
