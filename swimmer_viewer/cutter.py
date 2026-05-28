import os
import uuid
import re
import io
import zipfile
import subprocess
import logging
import glob
import shutil
import mimetypes

import numpy as np
from scipy.io import wavfile
from flask import (
    Flask, request, jsonify, render_template,
    send_file, Response, send_from_directory,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None  # no size limit

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuts_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg resolution (same logic as app.py)
# ---------------------------------------------------------------------------

def _valid_ffmpeg_candidate(candidate):
    if not candidate:
        return None
    candidate = os.path.expandvars(os.path.expanduser(str(candidate).strip().strip('"')))
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    paths = [candidate]
    if os.path.isdir(candidate):
        paths = [os.path.join(candidate, exe_name), os.path.join(candidate, "bin", exe_name)]
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def resolve_ffmpeg():
    for env_name in ("FFMPEG_PATH", "FFMPEG_BINARY"):
        p = _valid_ffmpeg_candidate(os.environ.get(env_name))
        if p:
            return p
    p = shutil.which("ffmpeg")
    if p:
        return p
    app_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = [
        os.path.join(app_dir, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(app_dir, "bin", "ffmpeg.exe"),
    ]
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        programdata = os.environ.get("ProgramData")
        if local_appdata:
            patterns.append(os.path.join(
                local_appdata, "Microsoft", "WinGet", "Packages",
                "*", "ffmpeg*", "bin", "ffmpeg.exe",
            ))
        if programdata:
            patterns.append(os.path.join(programdata, "chocolatey", "bin", "ffmpeg.exe"))
    for pattern in patterns:
        for candidate in glob.glob(pattern):
            p = _valid_ffmpeg_candidate(candidate)
            if p:
                return p
    return None


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def extract_audio(video_path, wav_path, sample_rate=16000):
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_found")
    cmd = [ffmpeg, "-i", video_path, "-ac", "1", "-ar", str(sample_rate), "-vn", "-y", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg_error: {result.stderr[-600:]}")


def compute_waveform(wav_path, n_bins=1500):
    """Return (normalized_rms_list, total_duration_s)."""
    try:
        sr, data = wavfile.read(wav_path)
    except Exception:
        return [], 0.0
    data = data.astype(np.float32)
    total = len(data) / sr
    n_bins = min(n_bins, len(data))
    bin_size = max(1, len(data) // n_bins)
    wf = [
        float(np.sqrt(np.mean(data[i * bin_size:(i + 1) * bin_size] ** 2)))
        for i in range(n_bins)
    ]
    mx = max(wf) if wf else 1.0
    if mx > 1e-8:
        wf = [v / mx for v in wf]
    return wf, round(total, 3)


def detect_all_buzzers(wav_path, min_separation=2.0):
    """Return sorted list of all buzzer timestamps (seconds) in the audio."""
    try:
        sr, data = wavfile.read(wav_path)
    except Exception as e:
        raise ValueError(f"Cannot read audio: {e}")

    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data)))
    if peak < 1e-8:
        return []
    data /= peak

    window_size = max(2, int(0.05 * sr))   # 50 ms
    hop_size    = max(1, window_size // 2)
    n_frames    = max(1, (len(data) - window_size) // hop_size + 1)

    try:
        shape   = (n_frames, window_size)
        strides = (data.strides[0] * hop_size, data.strides[0])
        frames  = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    except Exception:
        rms = np.array([
            float(np.sqrt(np.mean(data[i * hop_size: i * hop_size + window_size] ** 2)))
            for i in range(n_frames)
        ])

    frame_times = np.arange(n_frames) * hop_size / sr
    median_rms  = float(np.median(rms))
    if median_rms < 1e-8:
        return []

    threshold  = 3.0 * median_rms
    min_frames = max(1, int(0.1 / (hop_size / sr)))  # 100 ms minimum event duration

    events = []
    in_event = ev_start = ev_len = 0
    for i, above in enumerate(rms > threshold):
        if above and not in_event:
            in_event, ev_start, ev_len = True, i, 1
        elif above and in_event:
            ev_len += 1
        elif not above and in_event:
            in_event = False
            if ev_len >= min_frames:
                center = (ev_start + i) // 2
                events.append(float(frame_times[center]))
    if in_event and ev_len >= min_frames:
        center = (ev_start + min(ev_start + ev_len, n_frames - 1)) // 2
        events.append(float(frame_times[center]))

    distinct = []
    for t in events:
        if not distinct or (t - distinct[-1]) >= min_separation:
            distinct.append(t)
    return distinct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(name, fallback):
    clean = re.sub(r"[^\w\-]", "_", str(name).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return (clean[:60] or fallback)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("cutter.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify({"error": "No video uploaded."}), 400

    session_id  = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    filename   = os.path.basename(f.filename)
    video_path = os.path.join(session_dir, filename)
    f.save(video_path)
    logger.info("Saved %s (session %s)", filename, session_id)

    wav_path = os.path.join(session_dir, "__audio.wav")
    try:
        extract_audio(video_path, wav_path)
    except RuntimeError as e:
        if "ffmpeg_not_found" in str(e):
            return jsonify({"error": "ffmpeg not found. Install it and add to PATH."}), 400
        return jsonify({"error": f"Audio extraction failed: {e}"}), 400

    waveform, duration = compute_waveform(wav_path, n_bins=1500)
    logger.info("Upload done, duration=%.1f s", duration)

    return jsonify({
        "session_id": session_id,
        "filename":   filename,
        "duration":   duration,
        "waveform":   waveform,
    })


@app.route("/redetect", methods=["POST"])
def redetect():
    body       = request.get_json(force=True)
    session_id = body.get("session_id", "")
    min_sep    = float(body.get("min_sep", 2.0))

    wav_path = os.path.join(UPLOAD_FOLDER, session_id, "__audio.wav")
    if not os.path.isfile(wav_path):
        return jsonify({"error": "Session not found."}), 404

    try:
        buzzers = detect_all_buzzers(wav_path, min_separation=min_sep)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"buzzers": [round(t, 3) for t in buzzers]})


@app.route("/cut", methods=["POST"])
def cut():
    body       = request.get_json(force=True)
    session_id = body.get("session_id", "")
    filename   = body.get("filename", "")
    segments   = body.get("segments", [])   # [{name, start, end}]

    if not session_id or not filename or not segments:
        return jsonify({"error": "Missing parameters."}), 400

    session_dir = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    video_path  = os.path.abspath(os.path.join(session_dir, filename))
    if not video_path.startswith(session_dir + os.sep):
        return jsonify({"error": "Forbidden"}), 403
    if not os.path.isfile(video_path):
        return jsonify({"error": "Video file not found."}), 404

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        return jsonify({"error": "ffmpeg not found."}), 400

    ext     = os.path.splitext(filename)[1] or ".mp4"
    out_dir = os.path.join(session_dir, "cuts")
    os.makedirs(out_dir, exist_ok=True)

    cut_paths = []
    for i, seg in enumerate(segments):
        start    = float(seg.get("start", 0))
        end      = float(seg.get("end",   0))
        name     = safe_filename(seg.get("name", ""), f"segment_{i + 1:02d}")
        out_path = os.path.join(out_dir, f"{name}{ext}")

        if end <= start:
            continue

        # Fast stream-copy attempt
        cmd = [ffmpeg, "-y", "-ss", str(start), "-to", str(end),
               "-i", video_path, "-c", "copy", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = (result.returncode == 0
              and os.path.isfile(out_path)
              and os.path.getsize(out_path) > 100)

        if not ok:
            # Re-encode fallback for precise frame-accurate cuts
            cmd = [ffmpeg, "-y", "-i", video_path,
                   "-ss", str(start), "-to", str(end),
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                   "-c:a", "aac", out_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if result.returncode != 0:
                return jsonify({"error": f"Failed to cut '{name}': {result.stderr[-300:]}"}), 400

        cut_paths.append(out_path)
        logger.info("Cut %s [%.2f → %.2f s]", name, start, end)

    if not cut_paths:
        return jsonify({"error": "No segments produced."}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in cut_paths:
            zf.write(p, os.path.basename(p))
    buf.seek(0)

    zip_name = f"cuts_{os.path.splitext(filename)[0]}_{session_id[:6]}.zip"
    return send_file(buf, as_attachment=True, download_name=zip_name, mimetype="application/zip")


@app.route("/video/<session_id>/<path:filename>")
def serve_video(session_id, filename):
    session_dir = os.path.abspath(os.path.join(UPLOAD_FOLDER, session_id))
    target      = os.path.abspath(os.path.join(session_dir, filename))
    if not target.startswith(session_dir + os.sep):
        return "Forbidden", 403
    if not os.path.isfile(target):
        return "Not Found", 404

    file_size   = os.path.getsize(target)
    mime        = mimetypes.guess_type(target)[0] or "video/mp4"
    range_header = request.headers.get("Range")

    if not range_header:
        resp = send_from_directory(session_dir, filename)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    try:
        parts      = range_header.replace("bytes=", "").split("-")
        byte_start = int(parts[0]) if parts[0] else 0
        byte_end   = int(parts[1]) if parts[1] else file_size - 1
    except (ValueError, IndexError):
        return "Bad Range", 416

    byte_end = min(byte_end, file_size - 1)
    length   = byte_end - byte_start + 1

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
    resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
    resp.headers["Accept-Ranges"]  = "bytes"
    resp.headers["Content-Length"] = length
    return resp


if __name__ == "__main__":
    print("Video Buzzer Cutter running at http://127.0.0.1:5001")
    app.run(debug=True, port=5001, host="127.0.0.1")
