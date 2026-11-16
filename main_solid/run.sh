#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/installed ]; then
    PY=$(command -v python3.11 || command -v python3)
    echo "First run: creating .venv with $("$PY" --version)"
    if ! "$PY" -c "import tkinter" 2>/dev/null; then
        echo "tkinter is missing for $PY. Install it first, e.g.: sudo dnf install python3-tkinter"
        exit 1
    fi
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/installed
fi

.venv/bin/python main.py
