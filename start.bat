@echo off
setlocal

cd /d "%~dp0swimmer_viewer"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install from https://python.org
        pause
        exit /b 1
    )
)

echo Installing dependencies...
.venv\Scripts\pip install -q -r requirements.txt

echo.
echo Starting Swimmer Viewer...
echo Open http://127.0.0.1:5000 in your browser
echo Press Ctrl+C to stop.
echo.
.venv\Scripts\python app.py

pause
