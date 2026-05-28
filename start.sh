#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/swimmer_viewer"

if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

echo ""
echo "Starting Swimmer Viewer..."
echo "Open http://127.0.0.1:5000 in your browser"
echo "Press Ctrl+C to stop."
echo ""
.venv/bin/python app.py
