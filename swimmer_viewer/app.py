import os
import uuid
import mimetypes
import subprocess
import logging
import glob
import shutil
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, send_file

from audio_analysis import (
    AUDIO_SAMPLE_RATE,
    detect_buzzer_onsets,
    detect_buzzer_csv_onset,
    compute_waveform_envelope,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _timestr_to_seconds(s):
    """Convert any Phidget timestamp string to float seconds.

    Handles:
      - Full datetime: '2026/05/11 11:34:59.208317' → epoch seconds
      - HH:MM:SS.f    → total seconds
      - MM:SS.f       → total seconds
      - Plain float   → passthrough
    Only relative differences matter for calibration, so epoch seconds are fine.
    """
    s = str(s).strip()

    # Full datetime formats (Phidget Bridge default)
    for fmt in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass

    # Colon-separated time: HH:MM:SS.f or MM:SS.f
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
    except ValueError:
        pass

    return float(s)  # plain number fallback


def _try_parse_time_col(series):
    """Return (float_series, success) after attempting time-string conversion."""
    try:
        converted = series.map(_timestr_to_seconds)
        if converted.notna().sum() > 10:
            return converted.astype(np.float64), True
    except Exception:
        pass
    return None, False


def parse_phidget_csv(filepath):
    """Return (times, voltages) numpy float64 arrays from a Phidget Bridge CSV.

    Handles both plain-float timestamps and Phidget 'MM:SS.f' wall-clock strings.
    """
    errors = []
    for kwargs in [
        dict(comment="#"),
        dict(header=None, comment="#"),
        dict(sep=None, engine="python", comment="#"),
    ]:
        try:
            df = pd.read_csv(filepath, **kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how="all")

            # Build a working copy with numeric coercion
            work = pd.DataFrame()
            for col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                if numeric_col.notna().sum() > 10:
                    work[col] = numeric_col
                else:
                    # Try time-string conversion for columns that failed numeric parse
                    converted, ok = _try_parse_time_col(df[col])
                    if ok:
                        work[col] = converted

            if work.shape[1] >= 2:
                t = work.iloc[:, 0].dropna().values.astype(np.float64)
                v = work.iloc[:, 1].dropna().values.astype(np.float64)
                if len(t) > 10:
                    return t, v
        except Exception as e:
            errors.append(str(e))
    raise ValueError(f"Cannot parse CSV '{os.path.basename(filepath)}'. Tried multiple formats. Last error: {errors[-1] if errors else 'unknown'}")


# ---------------------------------------------------------------------------
# Video audio extraction
# ---------------------------------------------------------------------------

def _valid_ffmpeg_candidate(candidate):
    """Return a usable ffmpeg executable path from a file or directory candidate."""
    if not candidate:
        return None

    candidate = os.path.expandvars(os.path.expanduser(str(candidate).strip().strip('"')))
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    paths = [candidate]
    if os.path.isdir(candidate):
        paths = [
            os.path.join(candidate, exe_name),
            os.path.join(candidate, "bin", exe_name),
        ]

    for path in paths:
        if os.path.isfile(path):
            return path
    return None


def resolve_ffmpeg_executable():
    """Find ffmpeg even when the Flask process was started before PATH changed."""
    for env_name in ("FFMPEG_PATH", "FFMPEG_BINARY"):
        path = _valid_ffmpeg_candidate(os.environ.get(env_name))
        if path:
            return path

    path = shutil.which("ffmpeg")
    if path:
        return path

    app_dir = os.path.dirname(os.path.abspath(__file__))
    search_patterns = [
        os.path.join(app_dir, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(app_dir, "bin", "ffmpeg.exe"),
    ]

    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        programdata = os.environ.get("ProgramData")
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]

        if local_appdata:
            search_patterns.append(
                os.path.join(
                    local_appdata,
                    "Microsoft",
                    "WinGet",
                    "Packages",
                    "*",
                    "ffmpeg*",
                    "bin",
                    "ffmpeg.exe",
                )
            )
        if programdata:
            search_patterns.append(os.path.join(programdata, "chocolatey", "bin", "ffmpeg.exe"))
        for root in program_files:
            if root:
                search_patterns.append(os.path.join(root, "ffmpeg", "bin", "ffmpeg.exe"))

    for pattern in search_patterns:
        for candidate in glob.glob(pattern):
            path = _valid_ffmpeg_candidate(candidate)
            if path:
                return path

    return None


def extract_audio(video_path, output_wav, sample_rate=AUDIO_SAMPLE_RATE):
    """Use ffmpeg to extract mono WAV from a video file."""
    ffmpeg_path = resolve_ffmpeg_executable()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg_not_found")

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-y", output_wav,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg_not_found")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg_error: {result.stderr[-800:]}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved_videos, saved_csvs = [], []

    for fobj in request.files.getlist("videos"):
        if fobj.filename:
            name = os.path.basename(fobj.filename)
            fobj.save(os.path.join(session_dir, name))
            saved_videos.append(name)

    for fobj in request.files.getlist("csvs"):
        if fobj.filename:
            name = os.path.basename(fobj.filename)
            fobj.save(os.path.join(session_dir, name))
            saved_csvs.append(name)

    logger.info("Session %s: uploaded %d videos, %d CSVs", session_id, len(saved_videos), len(saved_csvs))
    return jsonify({"session_id": session_id, "videos": saved_videos, "csvs": saved_csvs})


@app.route("/process", methods=["POST"])
def process():
    body = request.get_json(force=True)
    session_id = body.get("session_id", "")
    ch1_file = body.get("ch1_file", "")
    ch3_file = body.get("ch3_file", "")
    video_files = body.get("video_files", [])
    # Optional manual overrides: seconds from start of CH1 recording
    buzz1_override = body.get("buzz1_override_s")
    buzz2_override = body.get("buzz2_override_s")
    # Optional bandpass filter for audio buzzer detection
    audio_lowcut  = body.get("audio_lowcut")
    audio_highcut = body.get("audio_highcut")
    use_lowcut  = float(audio_lowcut)  if audio_lowcut  and float(audio_lowcut)  > 10 else None
    use_highcut = float(audio_highcut) if audio_highcut and float(audio_highcut) > 10 else None

    if not session_id or not ch1_file or not ch3_file:
        return jsonify({"error": "Missing required parameters."}), 400

    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    warnings = []

    # --- Parse CSVs ---
    try:
        ch1_times, ch1_voltages = parse_phidget_csv(os.path.join(session_dir, ch1_file))
    except ValueError as e:
        return jsonify({"error": f"CH1 CSV error: {e}"}), 400

    try:
        ch3_times, ch3_voltages = parse_phidget_csv(os.path.join(session_dir, ch3_file))
    except ValueError as e:
        return jsonify({"error": f"CH3 CSV error: {e}"}), 400

    # --- Detect buzzer in CH1 (or use manual override) ---
    all_buzz_times = []
    if buzz1_override is not None:
        t_buzz1_csv = float(ch1_times[0]) + float(buzz1_override)
        t_buzz2_csv = (float(ch1_times[0]) + float(buzz2_override)) if buzz2_override is not None else None
        t_buzz3_csv = None
        all_buzz_times = [t_buzz1_csv] + ([t_buzz2_csv] if t_buzz2_csv is not None else [])
        warnings.append(
            f"Using manual buzzer times: t1 = {float(buzz1_override):.2f} s, "
            + (f"t2 = {float(buzz2_override):.2f} s" if buzz2_override is not None else "t2 = end of signal")
            + " from recording start."
        )
    else:
        try:
            t_buzz1_csv, t_buzz2_csv, t_buzz3_csv, det_info = detect_buzzer_csv_onset(ch1_times, ch1_voltages)
            logger.info("CH1 detection: %s", det_info)
            all_buzz_times = det_info.get("all_buzz_times", [])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # t_buzz2_csv is detected but NOT used to define experiment duration —
    # the full CH3 recording length is used instead.  t_buzz2_csv is only
    # kept for all_buzz_rel metadata and for the default CSV trim end.
    if t_buzz2_csv is None:
        t_buzz2_csv = float(ch1_times[-1])

    logger.info("CSV first buzzer: t1=%.3f s", t_buzz1_csv)

    # --- Normalised CH1 for graph overlay (full recording, re-zeroed at buzz1) ---
    ch1_t_norm = ch1_times - t_buzz1_csv
    ch1_v_norm = ch1_voltages
    if len(ch1_t_norm) > 10000:
        f = len(ch1_t_norm) // 10000 + 1
        ch1_t_norm = ch1_t_norm[::f]
        ch1_v_norm = ch1_v_norm[::f]

    # Exact local maximum of the first CH1 buzzer spike
    gm = float(np.median(ch1_voltages))
    win_mask = np.abs(ch1_times - t_buzz1_csv) < 10.0
    if win_mask.any():
        peak_idx = int(np.argmax(np.abs(ch1_voltages[win_mask] - gm)))
        ch1_first_peak_t = float(ch1_times[win_mask][peak_idx] - t_buzz1_csv)
    else:
        ch1_first_peak_t = 0.0

    # --- Process each video ---
    video_results = []
    for vf in video_files:
        video_path = os.path.join(session_dir, vf)
        audio_path = os.path.join(session_dir, f"__aud_{vf}.wav")

        try:
            extract_audio(video_path, audio_path)
        except RuntimeError as e:
            err = str(e)
            if "ffmpeg_not_found" in err:
                return jsonify({
                    "error": (
                        "ffmpeg is not installed or not in PATH.\n"
                        "Windows: download from https://ffmpeg.org/download.html, "
                        "extract, and add the bin\\ folder to your system PATH."
                    )
                }), 400
            return jsonify({"error": f"Audio extraction failed for '{vf}': {err}"}), 400

        try:
            t_buzz1_vid, t_buzz2_vid = detect_buzzer_onsets(audio_path, lowcut=use_lowcut, highcut=use_highcut)
        except ValueError as e:
            warnings.append(f"Audio read error for '{vf}': {e}")
            t_buzz1_vid, t_buzz2_vid = None, None

        if t_buzz1_vid is None:
            warnings.append(
                f"No buzzer detected in '{vf}'. "
                "Check the recording or signal threshold. Using offset = 0."
            )
            offset = 0.0
            t_buzz1_vid = 0.0
        else:
            offset = t_buzz1_csv - t_buzz1_vid

        logger.info(
            "Video '%s': buzz1_vid=%.3f s, offset=%.3f s",
            vf, t_buzz1_vid, offset
        )
        video_results.append({
            "filename":  vf,
            "offset":    round(float(offset), 4),
            "buzz1_vid": round(float(t_buzz1_vid), 4),
            "buzz2_vid": None,   # user sets this manually on page 2
        })

        # Audio waveform envelope — from before B1 to end of experiment
        csv_post_b1 = float(ch3_times[-1] - t_buzz1_csv)   # seconds from B1 to CSV end
        wf_start = max(0.0, float(t_buzz1_vid) - 5.0)
        wf_end   = float(t_buzz1_vid) + csv_post_b1 + 5.0
        wf_data, wf_t0, wf_t1, _ = compute_waveform_envelope(audio_path, wf_start, wf_end)
        video_results[-1]["waveform"]    = wf_data
        video_results[-1]["waveform_t0"] = round(wf_t0, 4)
        video_results[-1]["waveform_t1"] = round(wf_t1, 4)

    # --- Full CH3 data, re-zeroed so t=0 is at buzz1 ---
    # Keep the entire recording so the user can pan the viewport to find the signal.
    t_trim = ch3_times - t_buzz1_csv
    v_trim = ch3_voltages

    if len(t_trim) > 10000:
        factor = len(t_trim) // 10000 + 1
        t_trim = t_trim[::factor]
        v_trim = v_trim[::factor]

    # experiment_duration = time from B1 to end of CSV recording.
    # This makes the scrub slider span exactly 0 (B1) to end-of-CSV,
    # so the waveform bar fills the full slider width without squeezing.
    ch3_post_b1 = float(ch3_times[-1] - t_buzz1_csv)

    logger.info(
        "CH3: %d points, post-B1 duration=%.3f s",
        len(t_trim), ch3_post_b1
    )

    # Relative buzzer times in master time (t=0 at buzz1) — for display only
    all_buzz_rel = [round(float(t - t_buzz1_csv), 4) for t in all_buzz_times] if all_buzz_times else [0.0]

    return jsonify({
        "session_id": session_id,
        "videos": video_results,
        "ch3": {
            "time": t_trim.tolist(),
            "voltage": v_trim.tolist(),
        },
        "ch1": {
            "time": ch1_t_norm.tolist(),
            "voltage": ch1_v_norm.tolist(),
        },
        "ch1_first_peak_t":  round(ch1_first_peak_t, 4),
        "experiment_duration": round(ch3_post_b1, 4),   # seconds from B1 to end of CSV
        "t_buzz1_csv":  float(t_buzz1_csv),
        "t_buzz2_csv":  float(ch3_times[-1]),             # end of recording (default trim end)
        "all_buzz_rel": all_buzz_rel,
        "ch1_file": ch1_file,
        "ch3_file": ch3_file,
        "warnings": warnings,
    })


@app.route("/cal_waveform/<session_id>/<path:filename>")
def cal_waveform_endpoint(session_id, filename):
    """Return high-resolution audio RMS waveform for calibration page."""
    try:
        t_start = float(request.args.get("t_start", 0))
        t_end   = float(request.args.get("t_end",   60))
        n_bins  = min(2000, max(50, int(request.args.get("n_bins", 800))))
        lowcut  = float(request.args.get("lowcut",  0))
        highcut = float(request.args.get("highcut", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Bad parameters"}), 400

    use_lowcut  = lowcut  if lowcut  > 10 else None
    use_highcut = highcut if highcut > 10 else None

    session_dir = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    audio_path  = os.path.abspath(os.path.join(session_dir, f"__aud_{filename}.wav"))

    if not audio_path.startswith(session_dir + os.sep):
        return jsonify({"error": "Forbidden"}), 403
    if not os.path.isfile(audio_path):
        return jsonify({"error": "Audio not found — run /process first"}), 404

    wf_data, wf_t0, wf_t1, total_dur = compute_waveform_envelope(
        audio_path, t_start, t_end, n_bins,
        lowcut=use_lowcut, highcut=use_highcut,
    )
    return jsonify({"bins": wf_data, "t0": wf_t0, "t1": wf_t1, "total_dur": round(total_dur, 3)})


@app.route("/video/<session_id>/<path:filename>")
def serve_video(session_id, filename):
    """Serve video files with byte-range support for browser seeking."""
    session_dir = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    target = os.path.abspath(os.path.join(session_dir, filename))

    # Security: prevent path traversal
    if not target.startswith(session_dir + os.sep) and target != session_dir:
        return "Forbidden", 403
    if not os.path.isfile(target):
        return "Not Found", 404

    file_size = os.path.getsize(target)
    mime = mimetypes.guess_type(target)[0] or "application/octet-stream"
    range_header = request.headers.get("Range")

    if not range_header:
        resp = send_from_directory(session_dir, filename)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    try:
        parts = range_header.replace("bytes=", "").split("-")
        byte_start = int(parts[0]) if parts[0] else 0
        byte_end = int(parts[1]) if parts[1] else file_size - 1
    except (ValueError, IndexError):
        return "Bad Range Request", 416

    byte_end = min(byte_end, file_size - 1)
    length = byte_end - byte_start + 1

    def stream():
        with open(target, "rb") as f:
            f.seek(byte_start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    resp = Response(stream(), status=206, mimetype=mime, direct_passthrough=True)
    resp.headers["Content-Range"] = f"bytes {byte_start}-{byte_end}/{file_size}"
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = length
    return resp


@app.route("/export", methods=["POST"])
def export():
    """Trim every video and CSV to the calibrated [buzz1, buzz2] window and
    return them all in a single ZIP download."""
    body = request.get_json(force=True)
    session_id = body.get("session_id")
    ch1_file = body.get("ch1_file")
    ch3_file = body.get("ch3_file")
    videos = body.get("videos", [])
    t_buzz1_csv = body.get("t_buzz1_csv")
    t_buzz2_csv = body.get("t_buzz2_csv")

    if not all([session_id, ch1_file, ch3_file, videos]) or t_buzz1_csv is None or t_buzz2_csv is None:
        return jsonify({"error": "Missing required parameters."}), 400

    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found."}), 404

    ffmpeg_path = resolve_ffmpeg_executable()
    if not ffmpeg_path:
        return jsonify({"error": "ffmpeg not found — cannot trim videos."}), 400

    out_dir = os.path.join(session_dir, "trimmed")
    os.makedirs(out_dir, exist_ok=True)
    output_files = []

    # --- Trim videos with ffmpeg ---
    # vid_start / vid_end are the master-time trim window mapped back to video time
    # (computed in the frontend as: masterToVideo(trimStart/End, videoData[i]))
    for vid in videos:
        fname = vid.get("filename")
        b1 = float(vid.get("vid_start", vid.get("buzz1_vid", 0)))
        b2 = float(vid.get("vid_end",   vid.get("buzz2_vid", b1 + 1)))
        if b2 <= b1:
            return jsonify({"error": f"Invalid trim window for '{fname}': end ({b2:.3f}s) must be > start ({b1:.3f}s)."}), 400

        in_path = os.path.join(session_dir, fname)
        if not os.path.isfile(in_path):
            return jsonify({"error": f"Source video missing: {fname}"}), 400

        base, ext = os.path.splitext(fname)
        out_name = f"{base}_trimmed{ext}"
        out_path = os.path.join(out_dir, out_name)

        # Fast path: stream copy (keyframe-aligned, may have a few frames slack)
        cmd_copy = [
            ffmpeg_path, "-y",
            "-ss", str(b1), "-to", str(b2),
            "-i", in_path,
            "-c", "copy",
            out_path,
        ]
        result = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=600)
        copy_ok = (
            result.returncode == 0
            and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 1000
        )

        if not copy_ok:
            # Frame-accurate fallback: re-encode
            cmd_enc = [
                ffmpeg_path, "-y",
                "-i", in_path,
                "-ss", str(b1), "-to", str(b2),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac",
                out_path,
            ]
            result = subprocess.run(cmd_enc, capture_output=True, text=True, timeout=1200)
            if result.returncode != 0:
                return jsonify({
                    "error": f"Failed to trim '{fname}': {result.stderr[-400:]}"
                }), 400

        output_files.append(out_path)
        logger.info("Trimmed %s [%.3f → %.3f s]", fname, b1, b2)

    # --- Trim CSVs (keep original timestamp format) ---
    for csv_file in (ch1_file, ch3_file):
        in_path = os.path.join(session_dir, csv_file)
        try:
            df = pd.read_csv(in_path)
            time_col = df.columns[0]
            times_sec = df[time_col].astype(str).map(_timestr_to_seconds).astype(float)
            mask = (times_sec >= float(t_buzz1_csv)) & (times_sec <= float(t_buzz2_csv))
            trimmed = df[mask].copy()

            base, ext = os.path.splitext(csv_file)
            out_name = f"{base}_trimmed{ext}"
            out_path = os.path.join(out_dir, out_name)
            trimmed.to_csv(out_path, index=False)
            output_files.append(out_path)
            logger.info("Trimmed %s: %d → %d rows", csv_file, len(df), len(trimmed))
        except Exception as e:
            return jsonify({"error": f"Failed to trim CSV '{csv_file}': {e}"}), 400

    # --- Build ZIP in memory ---
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in output_files:
            zipf.write(path, os.path.basename(path))
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"swimmer_calibrated_{session_id[:8]}.zip",
        mimetype="application/zip",
    )


if __name__ == "__main__":
    print("Starting Swimmer Viewer at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, host="127.0.0.1")
