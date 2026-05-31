"""
audio_analysis.py — Research-grade audio analysis for swimmer elastic-band
experiment calibration.

Buzzer onset detection pipeline
────────────────────────────────
Primary  (when librosa is installed):
    Spectral-flux onset-strength envelope → onset detection with
    backtracking.  ``backtrack=True`` moves each detected onset to the
    preceding local minimum in the strength envelope — the true leading
    edge of the transient, not its energy peak or centre.

Fallback (no librosa required):
    Short-time energy envelope (5 ms hop / 20 ms window) → rising-edge
    threshold crossing with sub-frame linear interpolation.

Both paths report the ONSET (leading edge) of each buzz pulse, not the
centre.  This matches the moment the listener hears the buzzer, correcting
the systematic timing bias in the previous implementation where the
returned timestamp was the midpoint of the event window.

Install librosa for best results:
    pip install librosa soundfile
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

try:
    import librosa as _librosa
    _LIBROSA_AVAILABLE = True
    logger.info(
        "librosa %s available — spectral-flux onset detection active",
        _librosa.__version__,
    )
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.info(
        "librosa not installed — energy-envelope onset detection active "
        "(pip install librosa for best results)"
    )


# ── Constants ──────────────────────────────────────────────────────────────────

AUDIO_SAMPLE_RATE: int = 22050
"""Video audio extraction sample rate (Hz).

22 050 Hz gives ≈ 45 µs per sample (vs. 62 µs at 16 000 Hz), improving
temporal resolution of buzzer onset estimates by ~25 %.  Still fully above
the Nyquist for typical buzzer frequencies (< 5 kHz).
"""


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def _load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return ``(data_float64, sample_rate)``.

    * Normalises integer PCM (int16 / int32) to the range [-1, 1].
    * Mixes multi-channel audio to mono by averaging.
    """
    sr, raw = wavfile.read(wav_path)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    data = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        data /= float(np.iinfo(raw.dtype).max)
    return data, int(sr)


def bandpass_filter(
    data: np.ndarray,
    sr: int,
    lowcut: float,
    highcut: float,
    order: int = 5,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (scipy ``filtfilt``, order 5).

    Corner frequencies are clamped to safe Nyquist fractions so the filter
    is numerically stable regardless of the caller's input.
    """
    nyq = sr / 2.0
    low = float(np.clip(lowcut / nyq, 1e-4, 0.9999))
    high = float(np.clip(highcut / nyq, 1e-4, 0.9999))
    if high <= low:
        high = min(low + 1e-4, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data.astype(np.float64))


# ── Onset detection: librosa spectral-flux path ────────────────────────────────

def _detect_onsets_librosa(
    data: np.ndarray,
    sr: int,
    min_separation: float,
    lowcut: float | None,
    highcut: float | None,
) -> list[float]:
    """Spectral-flux onset detection with onset backtracking (librosa).

    ``center=False`` enforces causal framing: the onset-strength value at
    time *t* is derived only from audio up to *t*, with no look-ahead.
    ``backtrack=True`` then walks each detected peak back to the preceding
    local minimum of the strength envelope — the true onset instant.

    hop_length = 256 at 22 050 Hz → ≈ 11.6 ms per frame.
    """
    y = data.astype(np.float32)

    if lowcut and highcut and highcut > lowcut:
        y = bandpass_filter(y, sr, lowcut, highcut).astype(np.float32)

    hop = 256

    onset_env = _librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop,
        aggregate=np.median,   # robust to momentary noise spikes
        center=False,          # causal — no look-ahead
        fmin=float(lowcut)  if lowcut  else 20.0,
        fmax=float(highcut) if highcut else float(sr / 2.0),
    )

    wait_frames = max(2, int(min_separation * sr / hop))

    onset_frames = _librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop,
        backtrack=True,    # ← key: retrace to true leading edge
        pre_max=2,
        post_max=2,
        pre_avg=8,         # asymmetric: more past context than future
        post_avg=2,
        delta=0.07,
        wait=wait_frames,
        units="frames",
    )

    raw_times = _librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    distinct: list[float] = []
    for t in sorted(float(x) for x in raw_times):
        if not distinct or (t - distinct[-1]) >= min_separation:
            distinct.append(t)

    return distinct


# ── Onset detection: energy-envelope fallback ──────────────────────────────────

def _detect_onsets_energy(
    data: np.ndarray,
    sr: int,
    min_separation: float,
    lowcut: float | None,
    highcut: float | None,
) -> list[float]:
    """Short-time energy onset detection — no external dependencies required.

    5 ms hop / 20 ms analysis window.  Reports the sub-frame linearly
    interpolated time at which the squared-energy crosses the detection
    threshold on the rising edge.  Uses 10× the median energy as the
    threshold so the detector is robust to varying background levels.
    """
    y = data.copy().astype(np.float64)

    if lowcut and highcut and highcut > lowcut:
        y = bandpass_filter(y, sr, lowcut, highcut)

    peak = float(np.max(np.abs(y)))
    if peak < 1e-9:
        return []
    y /= peak

    hop_s = max(1, int(0.005 * sr))   # 5 ms hop
    win_s = max(2, int(0.020 * sr))   # 20 ms window
    n_fr  = max(1, (len(y) - win_s) // hop_s + 1)

    try:
        energy = np.mean(
            np.lib.stride_tricks.as_strided(
                y,
                shape=(n_fr, win_s),
                strides=(y.strides[0] * hop_s, y.strides[0]),
            ) ** 2,
            axis=1,
        )
    except Exception:
        energy = np.array([
            np.mean(y[i * hop_s: i * hop_s + win_s] ** 2)
            for i in range(n_fr)
        ])

    t_fr  = np.arange(n_fr) * hop_s / sr
    med_e = float(np.median(energy))
    if med_e < 1e-14:
        return []

    threshold  = max(10.0 * med_e, 1e-6)
    min_dur_fr = max(1, int(0.05 * sr / hop_s))   # event must last ≥ 50 ms

    onsets: list[float] = []
    in_ev  = False
    ev_s   = 0
    ev_len = 0

    for i, e in enumerate(energy):
        if e > threshold and not in_ev:
            in_ev  = True
            ev_s   = i
            ev_len = 1
        elif e > threshold and in_ev:
            ev_len += 1
        elif e <= threshold and in_ev:
            in_ev = False
            if ev_len >= min_dur_fr:
                # Sub-frame interpolated crossing time on the rising edge
                if ev_s > 0:
                    prev_e = energy[ev_s - 1]
                    curr_e = energy[ev_s]
                    frac   = (threshold - prev_e) / max(curr_e - prev_e, 1e-20)
                    t_on   = t_fr[ev_s - 1] + frac * (t_fr[ev_s] - t_fr[ev_s - 1])
                else:
                    t_on = t_fr[ev_s]
                onsets.append(float(t_on))

    if in_ev and ev_len >= min_dur_fr:
        onsets.append(float(t_fr[ev_s]))

    distinct: list[float] = []
    for t in onsets:
        if not distinct or (t - distinct[-1]) >= min_separation:
            distinct.append(t)

    return distinct


# ── Public API: video-audio buzzer onset detection ─────────────────────────────

def detect_buzzer_onsets(
    wav_path: str,
    min_separation: float = 3.0,
    lowcut: float | None = None,
    highcut: float | None = None,
) -> tuple[float | None, float | None]:
    """Detect up to two buzzer onset times from a mono WAV file.

    Parameters
    ----------
    wav_path:
        Path to a mono WAV file (any sample rate; produced by ffmpeg extraction).
    min_separation:
        Minimum seconds between distinct buzzer events.
    lowcut, highcut:
        Optional bandpass limits (Hz) to isolate the buzzer frequency band
        before onset detection.  Applied before both the librosa and energy
        detection paths.

    Returns
    -------
    (t_buzz1, t_buzz2) — seconds from the start of the audio file.
    Either value is ``None`` if fewer events were detected.

    Notes
    -----
    Uses librosa spectral-flux + backtracking when librosa is installed;
    otherwise uses the energy-envelope fallback.  Both paths report the
    *leading edge* (onset) of each buzz pulse — the moment it becomes
    audible — rather than the event centre used in the previous
    implementation.
    """
    try:
        data, sr = _load_wav(wav_path)
    except Exception as exc:
        raise ValueError(f"Cannot read audio '{wav_path}': {exc}") from exc

    if _LIBROSA_AVAILABLE:
        try:
            onsets = _detect_onsets_librosa(data, sr, min_separation, lowcut, highcut)
            logger.debug("librosa onsets: %s", onsets)
        except Exception as exc:
            logger.warning(
                "librosa onset detection failed (%s) — using energy fallback", exc
            )
            onsets = _detect_onsets_energy(data, sr, min_separation, lowcut, highcut)
    else:
        onsets = _detect_onsets_energy(data, sr, min_separation, lowcut, highcut)
        logger.debug("energy onsets: %s", onsets)

    return (
        onsets[0] if len(onsets) >= 1 else None,
        onsets[1] if len(onsets) >= 2 else None,
    )


# ── Public API: CH1 CSV buzzer onset detection ─────────────────────────────────

def detect_buzzer_csv_onset(
    times: np.ndarray,
    voltages: np.ndarray,
    min_separation: float = 2.0,
) -> tuple:
    """Detect buzzer events in the Phidget CH1 voltage signal.

    Preserves the global-median / MAD baseline approach from the previous
    implementation (robust to sustained pulses), but replaces cluster-centre
    reporting with *interpolated rising-edge crossing* times.

    Given Phidget sample intervals of ~1 ms, linear interpolation between
    adjacent samples gives sub-millisecond onset precision.

    Parameters
    ----------
    times:       1-D float64 timestamps (seconds, epoch or relative).
    voltages:    1-D float64 CH1 voltage samples.
    min_separation: minimum seconds between distinct events.

    Returns
    -------
    (t_buzz1, t_buzz2, t_buzz3, info_dict)
    t_buzz2 and t_buzz3 may be None.
    """
    duration_s = float(times[-1] - times[0])
    global_med = float(np.median(voltages))
    global_mad = float(np.median(np.abs(voltages - global_med)))
    residual   = np.abs(voltages - global_med)

    info: dict = {
        "duration_s":      round(duration_s, 2),
        "global_median":   round(global_med, 8),
        "global_mad":      global_mad,
        "residual_max":    float(residual.max()),
        "method":          "onset_interpolated",
        "multiplier_used": None,
        "n_events_found":  0,
    }

    noise = global_mad if global_mad > 1e-12 else (float(np.std(voltages)) or 1e-8)

    def _onsets(threshold: float, sep: float) -> list[float]:
        """Sub-sample interpolated rising-edge crossing times."""
        result: list[float] = []
        in_ev = False

        for i in range(1, len(residual)):
            r_prev, r_curr = float(residual[i - 1]), float(residual[i])
            if r_curr > threshold and not in_ev:
                # Linearly interpolate the exact crossing point
                if r_curr > r_prev:
                    frac = (threshold - r_prev) / (r_curr - r_prev + 1e-30)
                    t_on = float(times[i - 1]) + frac * (float(times[i]) - float(times[i - 1]))
                else:
                    t_on = float(times[i])
                result.append(t_on)
                in_ev = True
            elif r_curr <= threshold and in_ev:
                in_ev = False

        distinct: list[float] = []
        for t in result:
            if not distinct or (t - distinct[-1]) >= sep:
                distinct.append(t)
        return distinct

    for mult in (50, 20, 10, 5, 3):
        thr    = mult * noise
        onsets = _onsets(thr, min_separation)
        if onsets:
            info["multiplier_used"] = mult
            info["n_events_found"]  = len(onsets)
            info["all_buzz_times"]  = _onsets(thr, 0.4)
            return (
                onsets[0],
                onsets[1] if len(onsets) >= 2 else None,
                onsets[2] if len(onsets) >= 3 else None,
                info,
            )

    raise ValueError(
        f"No buzzer events detected in CH1 "
        f"(recording: {duration_s / 60:.1f} min, "
        f"max residual: {info['residual_max']:.2e}, "
        f"MAD: {global_mad:.2e}). "
        "Use the manual override fields to enter the buzzer times directly."
    )


# ── Public API: waveform envelope for visualisation ────────────────────────────

def compute_waveform_envelope(
    wav_path: str,
    t_start: float,
    t_end: float,
    n_bins: int = 400,
    lowcut: float | None = None,
    highcut: float | None = None,
) -> tuple[list[float], float, float, float]:
    """Compute RMS amplitude envelope for waveform display.

    Parameters
    ----------
    wav_path:       Path to a mono WAV file.
    t_start, t_end: Time window of interest (seconds from start of file).
    n_bins:         Number of RMS bins to compute.
    lowcut, highcut: Optional bandpass limits (Hz).

    Returns
    -------
    (bins, actual_t0, actual_t1, total_duration_s)
    ``bins`` is a list of ``n_bins`` floats in [0, 1] normalised to the peak.
    """
    try:
        data, sr = _load_wav(wav_path)
    except Exception:
        return [0.0] * n_bins, float(t_start), float(t_end), 0.0

    total = len(data) / sr
    t0    = max(0.0, float(t_start))
    t1    = min(total, float(t_end))

    if t1 <= t0:
        return [0.0] * n_bins, t0, t1, total

    seg = data[int(t0 * sr): int(t1 * sr)].copy()

    if lowcut and highcut and highcut > lowcut:
        try:
            seg = bandpass_filter(seg, sr, lowcut, highcut)
        except Exception:
            pass

    n        = min(n_bins, max(1, len(seg)))
    bin_size = max(1, len(seg) // n)
    wf: list[float] = [
        float(np.sqrt(np.mean(seg[i * bin_size: (i + 1) * bin_size] ** 2)))
        for i in range(n)
    ]

    while len(wf) < n_bins:
        wf.append(0.0)
    wf = wf[:n_bins]

    mx = max(wf) if wf else 0.0
    if mx > 1e-8:
        wf = [v / mx for v in wf]

    return wf, t0, t1, total
