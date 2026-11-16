# Swim Experiment Video Tools

Three small web apps for the swimmer elastic-band experiment. Each one runs **on your own computer** and opens in the browser. Nothing is uploaded to the internet.

| # | Tool | Use it when… | Start (Windows) | Address |
|---|------|--------------|-----------------|---------|
| 1 | **Swimmer Viewer** | You want to watch the camera videos **together with the force graph**, synchronized, and export a trimmed video + CSV | `start.bat` | http://127.0.0.1:5000 |
| 2 | **Video Buzzer Cutter** | You have one long video and just want to **cut it into pieces**, using the sound waveform as a guide | `start_cutter.bat` | http://127.0.0.1:5001 |
| 3 | **Light Calibration Tool** | You have long multi-camera GoPro videos where a **lamp flashes** at the start and end of each swim, and you want every swim cut out with its matching CSVs | `start_light_calibration.bat` | http://127.0.0.1:5001 |

> Tools 2 and 3 use the same port (5001), so run them one at a time.

---

## 1. Swimmer Viewer — sync video with force data (buzzer sound)

**Input:** one or more camera videos + the two Phidget CSVs (CH1 buzzer, CH3 force).

**What it does:**
1. Finds the buzzer beep in each video's audio and in the buzzer CSV, and lines them up on one clock.
2. Shows all videos side by side, with the force graph underneath. A red dot follows the video in real time.
3. Lets you fine-tune the alignment frame by frame.
4. Exports a ZIP with the trimmed videos and the matching CSV section.

Details: [swimmer_viewer/README.md](swimmer_viewer/README.md)

## 2. Video Buzzer Cutter — quick cutting by sound

**Input:** a single video.

**What it does:** shows the video next to its sound waveform, so the buzzer beeps are easy to spot. You add segments by hand, name them and set each start and end. Then click **Cut** and download a ZIP with one clip per segment. No CSV is needed.

## 3. Light Calibration Tool — cut swims by a light flash

**Input:** a folder of long camera videos (CamA/CamB/CamD…) + a folder of buzzer/force CSVs.
Here the "buzzer" channel holds the **lamp voltage**, not sound.

**What it does:**
1. Pairs each buzzer CSV with its force CSV and finds the two light pulses, which give the swim's start and end.
2. You skip the dead time at the start of each video and draw a small box around the lamp in each camera. It can also find the lamp automatically.
3. It scans the videos for the lamp flashes and matches each swim to its CSV by the swim's duration.
4. You review the results. Swims where the cameras disagree are flagged.
5. Exports the video clips + aligned force/buzzer CSVs for every swim.

Details: [light_calibration/README.md](light_calibration/README.md)

---

## Setup (once per computer)

1. **Python 3.11+**: https://python.org. Tick *"Add python.exe to PATH"* during install.
2. **ffmpeg**: https://ffmpeg.org/download.html. Unzip it and add its `bin` folder to `PATH`. Check with `ffmpeg -version`.
3. **Get the code:** click the green **Code → Download ZIP** button on this page and unzip it, or run:
   ```
   git clone https://github.com/Azaryad/calibration-tool.git
   ```
4. Double-click the tool's `.bat` file (see table above). The first run installs everything it needs, then tells you which address to open in the browser.

Mac/Linux: `bash start.sh` starts the Swimmer Viewer. For the other tools run `pip install -r requirements.txt` and then `python cutter.py` (inside `swimmer_viewer/`) or `python app.py` (inside `light_calibration/`).
