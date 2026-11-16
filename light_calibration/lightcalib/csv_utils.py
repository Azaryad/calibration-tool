"""Phidget CSV handling: pairing buzzer/force files (ordered by modified time),
tolerant parsing, adaptive light-pulse detection, and timestamp-mask trimming.

The voltage pulse marks the light on/off. The documented band 0.002-0.0025 fits
the samples, but the high bound is never actually reached (peaks ~0.00223), so we
threshold adaptively per file (baseline + fraction of the peak) with the absolute
band only as a sanity guard, and report sub-sample interpolated onsets."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SensorData:
    path: Path
    raw: pd.DataFrame
    times: np.ndarray
    values: np.ndarray


@dataclass
class Pulse:
    index: int
    onset_time: float        # interpolated rising-edge crossing (seconds, epoch)
    end_time: float
    peak_value: float

    def as_dict(self) -> dict:
        return asdict(self)


# ----- pairing ------------------------------------------------------------- #
def natural_key(value: str) -> list[object]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value.casefold())]


def classify_csv(path: Path) -> str | None:
    name = path.name.casefold()
    if "force" in name:
        return "force"
    if "buzzer" in name:
        return "buzzer"
    return None


def pair_key(path: Path) -> tuple[str, str]:
    stem = path.stem.casefold()
    m = re.search(r"^(?P<prefix>.*?)[._\-\s]*(force|buzzer)\s*(?P<num>\w*)$", stem)
    if not m:
        return re.sub(r"[^a-z0-9]+", "", stem), ""
    return re.sub(r"[^a-z0-9]+", "", m.group("prefix")), (m.group("num") or "")


def pair_csv_files(paths: Iterable[Path]) -> tuple[list[dict], list[str]]:
    """Pair name.buzzerN <-> name.forceN; order pairs by original modified time
    (NOT alphabetically -- alphabetical order can swap test order)."""
    warnings: list[str] = []
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    ignored: list[str] = []
    for path in paths:
        kind = classify_csv(path)
        if kind is None:
            ignored.append(path.name)
            continue
        grouped.setdefault(pair_key(path), {})[kind] = path
    if ignored:
        warnings.append("Ignored (no force/buzzer in name): " + ", ".join(ignored))

    pairs: list[dict] = []
    for key, group in grouped.items():
        if "force" in group and "buzzer" in group:
            try:
                mtime = min(group["force"].stat().st_mtime, group["buzzer"].stat().st_mtime)
            except OSError:
                mtime = 0.0
            pairs.append({"key": key, "force": group["force"], "buzzer": group["buzzer"], "mtime": mtime})
        else:
            have = "buzzer" if "buzzer" in group else "force"
            warnings.append(f"Unpaired {have} file for '{key[0]}{key[1]}'.")
    pairs.sort(key=lambda p: (p["mtime"], natural_key(p["buzzer"].name)))
    for i, p in enumerate(pairs, 1):
        p["index"] = i
    return pairs, warnings


# ----- parsing ------------------------------------------------------------- #
def parse_timestamp(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().strip('"').replace(",", ".")
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            s = float(parts[-1]); m = int(parts[-2]); hh = int(parts[-3]) if len(parts) == 3 else 0
            return hh * 3600 + m * 60 + s
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    return np.nan if pd.isna(parsed) else parsed.to_pydatetime().timestamp()


def read_sensor_csv(path: Path) -> SensorData:
    best = None
    for header in (0, None):
        try:
            df = pd.read_csv(path, comment="#", sep=None, engine="python",
                             header=header, encoding="utf-8-sig")
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        times = np.array([parse_timestamp(v) for v in df.iloc[:, 0]], dtype=float)
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times) & np.isfinite(values)
        if best is None or int(valid.sum()) > best[0]:
            best = (int(valid.sum()), df.loc[valid].copy(), times[valid], values[valid])
    if not best or best[0] == 0:
        raise ValueError(f"Could not read sensor CSV {path.name}")
    _, raw, times, values = best
    order = np.argsort(times, kind="stable")
    return SensorData(path=path, raw=raw.iloc[order].copy(), times=times[order], values=values[order])


# ----- pulse detection ----------------------------------------------------- #
def _extract_pulses(times: np.ndarray, values: np.ndarray, thr: float,
                    min_separation_sec: float) -> list[Pulse]:
    """Rising-edge crossings of ``thr`` with sub-sample onset interpolation,
    hysteresis on the falling edge, and a minimum-separation guard."""
    n = len(values)
    pulses: list[Pulse] = []
    i = 1
    while i < n:
        if values[i] >= thr and values[i - 1] < thr:
            v0, v1 = values[i - 1], values[i]
            alpha = (thr - v0) / (v1 - v0) if v1 != v0 else 0.0
            onset = float(times[i - 1] + alpha * (times[i] - times[i - 1]))
            j = i
            while j < n and values[j] >= thr * 0.85:   # hysteresis on the way down
                j += 1
            peak_v = float(np.nanmax(values[i:j])) if j > i else float(values[i])
            pulses.append(Pulse(index=len(pulses) + 1, onset_time=onset,
                                 end_time=float(times[min(j, n - 1)]), peak_value=peak_v))
            guard = onset + min_separation_sec
            while i < n and times[i] < guard:
                i += 1
            continue
        i += 1
    return pulses


def detect_pulses(times: np.ndarray, values: np.ndarray, *,
                  abs_low: float = 0.002, abs_high: float = 0.0025,
                  frac: float = 0.45, min_separation_sec: float = 4.0) -> list[Pulse]:
    """Rising-edge pulse detection. Primary threshold is the documented absolute
    crossing (0.002), which is validated on the real data; an adaptive,
    MAD-gated fallback handles files whose pulses sit at a different scale.

    The earlier version estimated noise with a high percentile of the deviation,
    which the pulse samples themselves inflated -> nothing ever cleared the gate.
    MAD (median absolute deviation) is immune to the ~2% pulse outliers."""
    n = len(values)
    if n == 0:
        return []
    baseline = float(np.median(values))
    mad = float(np.median(np.abs(values - baseline)))
    sigma = 1.4826 * mad                     # robust, pulse-immune noise estimate
    peak = float(np.percentile(values, 99.9))

    candidates: list[float] = []
    if peak > abs_low > baseline:            # documented band fits -> use it
        candidates.append(abs_low)
    if sigma > 0 and peak - baseline > 5 * sigma:   # adaptive fallback
        adaptive = baseline + max(frac * (peak - baseline), 4 * sigma)
        candidates.append(min(adaptive, baseline + 0.85 * (peak - baseline)))

    if not candidates:
        return []
    for thr in candidates:                   # prefer the first that finds >=2 pulses
        pulses = _extract_pulses(times, values, thr, min_separation_sec)
        if len(pulses) >= 2:
            return pulses
    return _extract_pulses(times, values, candidates[0], min_separation_sec)


def sample_stats(times: np.ndarray) -> dict:
    if len(times) < 2:
        return {"samples": int(len(times)), "median_dt": None, "duration": 0.0,
                "warnings": ["CSV has fewer than two samples."]}
    diffs = np.diff(times)
    pos = diffs[diffs > 0]
    warnings: list[str] = []
    median_dt = float(np.median(pos)) if len(pos) else None
    if median_dt and float(np.max(diffs)) > median_dt * 10 and float(np.max(diffs)) > 0.05:
        warnings.append(f"Large timestamp gap: {float(np.max(diffs)):.4f}s.")
    if np.any(diffs <= 0):
        warnings.append("Duplicate or decreasing timestamps present.")
    return {"samples": int(len(times)), "median_dt": median_dt,
            "duration": float(times[-1] - times[0]), "warnings": warnings}


def trim_csv_to_window(input_path: Path, output_path: Path,
                       start_time: float, end_time: float) -> int:
    data = read_sensor_csv(input_path)
    mask = (data.times >= start_time) & (data.times <= end_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.raw.loc[mask].to_csv(output_path, index=False)
    return int(mask.sum())
