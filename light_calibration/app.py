"""Flask app for the light calibration tool.

Workflow: select local folders -> confirm CSV pairs -> set per-video trim start
-> mark/auto-locate the light ROI per camera -> run windowed detection -> review
-> export aligned clips + CSVs as a ZIP. Videos are referenced by local path and
never copied (they can be several GB each)."""
from __future__ import annotations

import json
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template,
                   request, send_file, url_for)

from lightcalib import csv_utils, ffmpeg_utils, pipeline, roi_assist

app = Flask(__name__)
DATA = Path(__file__).parent / "data"
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
_JOB_CACHE: dict[str, dict] = {}
_RUNS: dict[str, dict] = {}            # job_id -> {state, log, results}


# --------------------------------------------------------------------------- #
# Job persistence (small JSON; the heavy media stays where it is)             #
# --------------------------------------------------------------------------- #
def job_path(job_id: str) -> Path:
    return JOBS / job_id / "job.json"


def load_job(job_id: str) -> dict:
    if job_id in _JOB_CACHE:
        return _JOB_CACHE[job_id]
    p = job_path(job_id)
    if not p.exists():
        abort(404)
    job = json.loads(p.read_text(encoding="utf-8"))
    _JOB_CACHE[job_id] = job
    return job


def save_job(job: dict) -> None:
    _JOB_CACHE[job["id"]] = job
    p = job_path(job["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pages                                                                       #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/create", methods=["POST"])
def create():
    video_dir = Path(request.form.get("video_dir", "").strip().strip('"'))
    csv_dir = Path(request.form.get("csv_dir", "").strip().strip('"'))
    errors = []
    if not video_dir.is_dir():
        errors.append(f"Video folder not found: {video_dir}")
    if not csv_dir.is_dir():
        errors.append(f"CSV folder not found: {csv_dir}")
    if errors:
        return render_template("index.html", errors=errors,
                               video_dir=str(video_dir), csv_dir=str(csv_dir))

    job_id = uuid.uuid4().hex[:12]
    # videos (ordered by name; cameras are independent recordings)
    videos = []
    for i, vp in enumerate(sorted(p for p in video_dir.iterdir()
                                  if p.suffix.lower() in VIDEO_EXTS), 1):
        try:
            meta = ffmpeg_utils.probe(vp)
        except Exception as exc:  # keep going; report in UI
            meta = None
        videos.append({"id": f"video_{i:02d}", "name": vp.name, "path": str(vp),
                       "meta": meta.as_dict() if meta else None,
                       "trim_start": 0.0, "roi": None})

    # CSV pairs (ordered by modified time) + pulse-derived durations
    pairs, warnings = csv_utils.pair_csv_files(list(csv_dir.glob("*.csv")))
    csv_pairs = []
    for pr in pairs:
        entry = {"index": pr["index"], "key": "".join(pr["key"]),
                 "buzzer": str(pr["buzzer"]), "force": str(pr["force"]),
                 "buzzer_name": pr["buzzer"].name, "force_name": pr["force"].name,
                 "duration": None, "pulses": 0, "warnings": []}
        try:
            bz = csv_utils.read_sensor_csv(pr["buzzer"])
            pulses = csv_utils.detect_pulses(bz.times, bz.values)
            entry["pulses"] = len(pulses)
            if len(pulses) >= 2:
                entry["duration"] = pulses[1].onset_time - pulses[0].onset_time
                entry["onset_abs"] = [pulses[0].onset_time, pulses[1].onset_time]
            else:
                entry["warnings"].append("Fewer than two pulses detected.")
        except Exception as exc:
            entry["warnings"].append(f"CSV read error: {exc}")
        csv_pairs.append(entry)

    job = {"id": job_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "video_dir": str(video_dir), "csv_dir": str(csv_dir),
           "videos": videos, "csv_pairs": csv_pairs, "warnings": warnings}
    save_job(job)
    return redirect(url_for("pairs", job_id=job_id))


@app.route("/job/<job_id>/pairs")
def pairs(job_id):
    return render_template("pairs.html", job=load_job(job_id))


@app.route("/job/<job_id>/trim")
def trim(job_id):
    return render_template("trim.html", job=load_job(job_id))


@app.route("/job/<job_id>/trim", methods=["POST"])
def trim_save(job_id):
    job = load_job(job_id)
    for v in job["videos"]:
        val = request.form.get(f"trim_{v['id']}")
        if val is not None:
            try:
                v["trim_start"] = max(0.0, float(val))
            except ValueError:
                pass
    save_job(job)
    return redirect(url_for("roi", job_id=job_id))


@app.route("/job/<job_id>/roi")
def roi(job_id):
    return render_template("roi.html", job=load_job(job_id))


@app.route("/job/<job_id>/roi", methods=["POST"])
def roi_save(job_id):
    job = load_job(job_id)
    data = request.get_json(force=True)
    for v in job["videos"]:
        if v["id"] in data:
            v["roi"] = {k: int(round(float(data[v["id"]][k]))) for k in ("x", "y", "w", "h")}
    save_job(job)
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# Lightweight media endpoints (no full-video load)                            #
# --------------------------------------------------------------------------- #
@app.route("/job/<job_id>/frame/<video_id>")
def frame(job_id, video_id):
    job = load_job(job_id)
    v = next((x for x in job["videos"] if x["id"] == video_id), None)
    if not v:
        abort(404)
    t = float(request.args.get("t", v.get("trim_start", 0.0)))
    out = JOBS / job_id / "frames" / f"{video_id}_{int(t*1000):09d}.jpg"
    if not out.exists():
        ffmpeg_utils.extract_frame(Path(v["path"]), out, t, scale_w=900)
    return send_file(out, mimetype="image/jpeg")


@app.route("/job/<job_id>/autoroi/<video_id>", methods=["POST"])
def autoroi(job_id, video_id):
    job = load_job(job_id)
    v = next((x for x in job["videos"] if x["id"] == video_id), None)
    if not v or not v.get("meta"):
        abort(404)
    meta = ffmpeg_utils.VideoMeta(**v["meta"])
    try:
        roi = roi_assist.locate_light(Path(v["path"]), meta,
                                      start_sec=v.get("trim_start", 0.0), duration_sec=180.0)
    except Exception as exc:
        return jsonify(ok=False, message=f"Auto-locate failed: {exc}")
    if roi is None:
        return jsonify(ok=False, message="Could not auto-locate a light; draw it manually.")
    return jsonify(ok=True, roi=roi)


# --------------------------------------------------------------------------- #
# Detection (runs in a background thread with a progress log)                 #
# --------------------------------------------------------------------------- #
def _build_inputs(job: dict):
    cameras = []
    for v in job["videos"]:
        if v.get("roi") and v.get("meta"):
            cameras.append({"id": v["id"], "path": Path(v["path"]),
                            "meta": ffmpeg_utils.VideoMeta(**v["meta"]),
                            "roi": v["roi"], "trim_start": v.get("trim_start", 0.0)})
    csv_segs = [{"index": p["index"], "key_str": p["key"], "duration": p["duration"]}
                for p in job["csv_pairs"] if p.get("duration")]
    return cameras, csv_segs


def _run_detection(job_id: str):
    run = _RUNS[job_id]
    job = load_job(job_id)
    try:
        cameras, csv_segs = _build_inputs(job)
        if not cameras:
            raise RuntimeError("No camera has an ROI set.")
        if not csv_segs:
            raise RuntimeError("No CSV segment has two pulses.")
        results, info = pipeline.analyze_job(cameras, csv_segs,
                                             progress=lambda m: run["log"].append(m))
        job["results"] = [r.as_dict() for r in results]
        job["info"] = info
        save_job(job)
        run["state"] = "done"
    except Exception as exc:
        run["log"].append(f"ERROR: {exc}")
        run["state"] = "error"


@app.route("/job/<job_id>/detect")
def detect(job_id):
    return render_template("detect.html", job=load_job(job_id),
                           running=_RUNS.get(job_id, {}).get("state") == "running")


@app.route("/job/<job_id>/detect/start", methods=["POST"])
def detect_start(job_id):
    if _RUNS.get(job_id, {}).get("state") == "running":
        return jsonify(ok=True, already=True)
    _RUNS[job_id] = {"state": "running", "log": []}
    threading.Thread(target=_run_detection, args=(job_id,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/job/<job_id>/detect/status")
def detect_status(job_id):
    run = _RUNS.get(job_id)
    if not run:
        job = load_job(job_id)
        return jsonify(state="done" if job.get("results") else "idle", log=[])
    return jsonify(state=run["state"], log=run["log"][-200:])


@app.route("/job/<job_id>/review")
def review(job_id):
    job = load_job(job_id)
    return render_template("review.html", job=job, results=job.get("results"))


# --------------------------------------------------------------------------- #
# Export: frame-accurate clips + masked CSVs + metadata, zipped               #
# --------------------------------------------------------------------------- #
@app.route("/job/<job_id>/export", methods=["POST"])
def export(job_id):
    job = load_job(job_id)
    results = job.get("results") or []
    vids = {v["id"]: v for v in job["videos"]}
    pairs = {p["index"]: p for p in job["csv_pairs"]}
    out_dir = JOBS / job_id / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seg in results:
        idx = seg["csv_index"]
        pair = pairs.get(idx)
        if not pair or not pair.get("onset_abs"):
            continue
        seg_dir = out_dir / f"segment_{idx:03d}_{seg['csv_key']}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        dur = seg["csv_duration"]
        a0, a1 = pair["onset_abs"]
        # CSV: mask on the shared timestamp clock
        csv_utils.trim_csv_to_window(Path(pair["force"]), seg_dir / pair["force_name"], a0, a1)
        csv_utils.trim_csv_to_window(Path(pair["buzzer"]), seg_dir / pair["buzzer_name"], a0, a1)
        # videos: cut each confident camera at its detected start, length = CSV duration
        clips = {}
        for cam_id, cr in seg["cameras"].items():
            if cr.get("start_time") is None or cr.get("confidence") == "none":
                continue
            v = vids[cam_id]
            meta = ffmpeg_utils.VideoMeta(**v["meta"])
            out = seg_dir / f"{cam_id}_{Path(v['name']).stem}.mp4"
            ffmpeg_utils.export_clip(Path(v["path"]), out, cr["start_time"], dur, meta)
            clips[cam_id] = {"file": out.name, "start_time": cr["start_time"],
                             "duration": dur, "confidence": cr["confidence"]}
        (seg_dir / "segment.json").write_text(json.dumps(
            {"segment": idx, "key": seg["csv_key"], "csv_duration": dur,
             "confirmed": seg["confirmed"], "clips": clips,
             "force_csv": pair["force_name"], "buzzer_csv": pair["buzzer_name"]},
            indent=2), encoding="utf-8")

    zip_path = JOBS / job_id / f"light_calibration_{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))
    return send_file(zip_path, as_attachment=True, download_name=zip_path.name)


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
