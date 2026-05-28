# Claude Code Prompt — Swimmer Elastic Band Experiment: Synchronized Data Viewer

## Context

I am running a swimming experiment where a swimmer is attached to an elastic band (resistance band). The setup records:

- **Multiple video files** from cameras positioned around the pool.
- **CSV files exported from a Phidget Bridge DAQ**, one CSV per channel:
  - **Channel 1 (CH1):** Buzzer pulse signal — a brief voltage spike that marks the start and end of the experiment.
  - **Channel 3 (CH3):** Force/strain sensor signal — continuous voltage output from a Strain Gauge force cell. This is the primary measurement channel.

A buzzer is activated twice per trial: once at the **start** and once at the **end** of the swimmer's effort. The same buzzer is audible in the video recordings *and* creates a voltage pulse on CH1. The absolute timestamps of the video files and the CSV files are **not aligned** — there is an unknown offset between them that must be calibrated automatically.

---

## Goal

Build a **local web application** (runs in the browser, served from a simple Python backend) that:

1. Accepts uploaded video files and Phidget Bridge CSV files.
2. Automatically detects the buzzer events in both the audio track of the videos and in the CH1 CSV signal.
3. Uses those detections to **calibrate the time offset** between the video timeline and the CSV timeline.
4. Trims the active experiment window to the interval between the first and second buzzer events.
5. Displays a synchronized viewer with:
   - All uploaded videos playing **side by side**, synchronized to a single playback clock.
   - A **force graph** (CH3 voltage vs. time) below or beside the videos.
   - A **red tracking dot** on the graph that moves in real time as the video plays.
   - A **tooltip bubble** attached to the red dot showing the precise current time (seconds from experiment start) and the exact CH3 voltage value at that moment.
6. Allows the user to **scrub** (seek forward and backward) in any video, with all other videos and the graph dot updating instantly.

---

## Technical Specification

### Stack

- **Backend:** Python 3 with `Flask` (or `FastAPI`). Handles file upload, audio analysis, CSV parsing, and calibration computation. Serves a single HTML page.
- **Frontend:** Single HTML file with vanilla JavaScript and `Chart.js` for the graph. No React or build step required.
- **Audio analysis:** `librosa` (or `scipy` + `soundfile`) for extracting audio from video and detecting the buzzer.
- **Video extraction:** `ffmpeg` (via `subprocess`) to extract the audio track from each video file for analysis. Do NOT require the user to pre-extract audio.
- **CSV parsing:** `pandas`.

### Project Structure

```
swimmer_viewer/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Single-page frontend
├── static/
│   └── (Chart.js served from CDN, no local files needed)
├── uploads/                # Created at runtime for uploaded files
└── requirements.txt
```

### Step 1 — File Upload

The upload page should accept:
- One or more **video files** (`.mp4`, `.mov`, `.avi`, `.mkv`).
- One or more **CSV files** from Phidget Bridge. The user selects which CSV is CH1 and which is CH3 (via a dropdown next to each uploaded CSV, defaulting to detected channel from filename if it contains "ch1" or "ch3" or "channel1" / "channel3").

After upload, a "Process" button triggers the backend pipeline.

### Step 2 — CSV Parsing (Phidget Bridge format)

Phidget Bridge CSVs have a header row followed by rows of `timestamp, voltage`. Parse each CSV into a `(time_array, voltage_array)` pair. The timestamp is in **seconds** (float). Store CH1 and CH3 separately.

Example expected format:
```
Timestamp (s), Voltage (V)
0.000, 0.00012
0.010, 0.00011
...
```

Handle slight variations (extra whitespace, alternate column names). If the format differs, attempt auto-detection of the two numeric columns.

### Step 3 — Buzzer Detection in CH1 CSV

On the CH1 signal:
1. Compute a rolling baseline (median over a 2-second window).
2. A **buzzer pulse** is a sample where `abs(voltage - baseline) > threshold`, where threshold is auto-set to `5 × median_absolute_deviation` of the full signal.
3. Cluster consecutive above-threshold samples into single events.
4. Take the **two earliest distinct events** (separated by at least 5 seconds) as `t_buzz1_csv` and `t_buzz2_csv` (in CSV time, seconds).
5. The **active experiment window** in CSV time is `[t_buzz1_csv, t_buzz2_csv]`.

### Step 4 — Buzzer Detection in Video Audio

For each uploaded video:
1. Extract mono audio at 16 kHz using ffmpeg: `ffmpeg -i video.mp4 -ac 1 -ar 16000 -vn audio.wav`.
2. Load the waveform with `librosa` or `scipy.io.wavfile`.
3. Compute a short-time RMS energy with a window of ~50 ms.
4. A buzzer event is a frame where `RMS > 3 × median(RMS)` sustained for at least 100 ms.
5. Cluster into events; take the **two earliest distinct events** (separated by at least 3 seconds) as `t_buzz1_vid` and `t_buzz2_vid` (in video time, seconds from start of video file).
6. If only one buzz is detected in a video, log a warning but continue.

### Step 5 — Time Calibration

For each video, compute the **offset** between video time and CSV time:
```
offset = t_buzz1_csv - t_buzz1_vid
```
This means: `csv_time = video_time + offset`.

Verify: `t_buzz2_csv` should approximately equal `t_buzz2_vid + offset` (tolerance ±0.5 s). If not, log a warning to the console.

Store `offset_per_video[video_filename]` for use in the viewer.

### Step 6 — Data Preparation for Frontend

After calibration, prepare and return a JSON payload to the frontend containing:

```json
{
  "videos": [
    {
      "filename": "cam1.mp4",
      "offset": 12.34,
      "buzz1_vid": 3.21,
      "buzz2_vid": 47.88
    }
  ],
  "ch3": {
    "time": [0.0, 0.01, 0.02, ...],   // seconds, trimmed to experiment window, re-zeroed so t=0 is buzz1
    "voltage": [0.00012, 0.00011, ...]
  },
  "experiment_duration": 44.67
}
```

Trim the CH3 data to `[t_buzz1_csv - 0.5, t_buzz2_csv + 0.5]` and re-zero so that `t=0` corresponds to the first buzzer event. Downsample if necessary to keep the array under 10,000 points (use `numpy` decimation).

### Step 7 — Frontend Viewer (`index.html`)

#### Layout

```
+--------------------------------------------------+
|  [Video 1]   [Video 2]   [Video 3]  ...          |
|                                                  |
|  [===Force Graph (CH3) — full width============] |
|       red dot with tooltip bubble                |
+--------------------------------------------------+
```

#### Videos

- Display all videos in a flex row, each in a `<video>` element.
- All videos share a **single master clock** driven by the first video's `timeupdate` event.
- When video 1 fires `timeupdate`, compute `masterTime = video1.currentTime - buzz1_vid_video1` (experiment time, seconds from first buzz).
- For each other video: compute `targetTime = masterTime + buzz1_vid_videoN`, then if `|videoN.currentTime - targetTime| > 0.15s`, set `videoN.currentTime = targetTime`.
- A single Play/Pause button controls all videos simultaneously.
- A single scrub slider (range input, 0 to `experiment_duration`) lets the user seek all videos and the graph dot together.

#### Force Graph (Chart.js)

- Use `Chart.js` loaded from CDN (`https://cdn.jsdelivr.net/npm/chart.js`).
- Line chart: X axis = time (seconds from experiment start, 0 to `experiment_duration`), Y axis = voltage (V).
- Line color: `#2E75B6`, no point markers on the line itself.
- Add a **single red scatter point** as a second dataset, initially at `(0, ch3_voltage_at_t0)`.
- On each `timeupdate` event, update the red dot's x to `masterTime` and y to the interpolated CH3 voltage at that time (use linear interpolation between the two nearest data points).
- Display a **custom tooltip** that is always visible (not hover-only) next to the red dot showing: `t = X.XX s` and `V = Y.YYYY V`. Implement this as an absolutely-positioned `<div>` overlay updated via JavaScript, not Chart.js's built-in tooltip.
- The graph should NOT re-render the full chart on every frame — only update the red dot dataset and the overlay div. Use `chart.update('none')` (no animation) for performance.

#### Scrubbing

- `<input type="range">` below the graph.
- On `input` event: set `video1.currentTime = sliderValue + buzz1_vid_video1`, then trigger sync for all other videos and update the graph dot directly.
- On `timeupdate`: update slider position to reflect current `masterTime`.

---

## Error Handling

- If no buzzer is detected in a file, show a clear warning in the UI: "No buzzer detected in [filename]. Please check the signal threshold or upload a different file."
- If uploaded CSVs cannot be parsed, show the specific parsing error.
- If ffmpeg is not installed, show an installation instruction.

---

## Dependencies (`requirements.txt`)

```
flask
pandas
numpy
scipy
librosa
```

ffmpeg must be installed separately on the system (provide a note in the README).

---

## Deliverables

1. `app.py` — complete Flask application.
2. `templates/index.html` — complete single-page frontend.
3. `requirements.txt`.
4. A brief `README.md` explaining how to install and run (`pip install -r requirements.txt`, then `python app.py`), and how to install ffmpeg on Windows/macOS/Linux.

---

## Important Implementation Notes

- **Do not use React, Vue, or any build tools.** Pure HTML + vanilla JS only.
- **Do not use `localStorage` or `sessionStorage`.**
- The app must work fully offline after `pip install` and `npm`/CDN resources for Chart.js are loaded once.
- The red dot update loop must run at the video's native frame rate — use `requestAnimationFrame` tied to the video's `timeupdate` for smooth motion.
- All time arithmetic must use **float seconds** throughout — do not convert to integer frames at any intermediate step.
- The calibration offset can be negative (if the video starts before the CSV recording).
- Keep all uploaded files in `uploads/` sub-folders named by a UUID per session to avoid filename collisions.
