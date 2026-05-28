# Swimmer Elastic Band — Synchronized Viewer

Local web app for synchronizing pool-camera videos with Phidget Bridge force/buzzer CSV data.

## Quick Start (easiest)

**Windows:** double-click `start.bat` in the root folder — it handles everything and opens the app.  
**Mac/Linux:** run `bash start.sh` from the root folder.

Then open **http://127.0.0.1:5000** in your browser.

---

## Manual Setup

### 1. Python 3.9+

Install from https://python.org if you don't have it. Check with `python --version`.

### 2. Python packages
```
cd swimmer_viewer
pip install -r requirements.txt
```

### 3. ffmpeg (required for audio extraction)

| OS | Instructions |
|----|-------------|
| **Windows** | Download from https://ffmpeg.org/download.html → extract zip → add the `bin\` folder to your system `PATH` environment variable. Verify with `ffmpeg -version` in a new terminal. |
| **macOS** | `brew install ffmpeg` |
| **Linux** | `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora) |

On Windows, the app also checks the common WinGet and Chocolatey install locations. If `ffmpeg` is installed somewhere custom, set `FFMPEG_PATH` to either `ffmpeg.exe` or the folder containing it before running `python app.py`.

### 4. Run
```
cd swimmer_viewer
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

---

## Usage

1. **Select video files** — one or more `.mp4 / .mov / .avi / .mkv` camera recordings.
2. **Select CSV files** — both Phidget Bridge CSVs (CH1 buzzer signal and CH3 force signal).  
   Use the dropdown next to each file to assign **CH1 — Buzzer** or **CH3 — Force**.
3. Click **Process Files**.  
   The backend will extract audio, detect the two buzzer events, calibrate the time offset, and open the synchronized viewer.
4. Use **▶ Play / ⏸ Pause** and the scrub slider to navigate.  
   All videos stay locked to the same experiment clock; the red dot on the force graph tracks in real time.

## Expected CSV format (Phidget Bridge)

```
Timestamp (s), Voltage (V)
0.000, 0.00012
0.010, 0.00011
...
```

Column names are flexible — the parser auto-detects the first two numeric columns.

## Project structure

```
swimmer_viewer/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Single-page frontend (vanilla JS + Chart.js CDN)
├── uploads/                # Created at runtime, one UUID sub-folder per session
└── requirements.txt
```
