#!/usr/bin/env bash
set -e

find_compatible_python() {
    # Prefer Python 3.12, 3.11, or 3.10 (standard PyTorch wheel support)
    for ver in 3.12 3.11 3.10; do
        if command -v py >/dev/null 2>&1 && py -"$ver" -c "import sys; assert sys.maxsize > 2**32" 2>/dev/null; then
            py -"$ver" -c "import sys; print(sys.executable)"
            return 0
        fi
        if command -v "python$ver" >/dev/null 2>&1 && "python$ver" -c "import sys; assert sys.maxsize > 2**32" 2>/dev/null; then
            command -v "python$ver"
            return 0
        fi
    done

    # Fallback to default python3 / python
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            command -v "$cmd"
            return 0
        fi
    done
    return 1
}

# 1. Ensure a working virtual environment is active
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    # If a previous .venv exists but lacks torch (e.g., broken install), recreate it cleanly
    if [ -d ".venv" ]; then
        VENV_PY=".venv/Scripts/python.exe"
        [ ! -f "$VENV_PY" ] && VENV_PY=".venv/bin/python"
        if [ ! -f "$VENV_PY" ] || ! "$VENV_PY" -c "import torch" 2>/dev/null; then
            echo "[SH] Removing incomplete .venv and recreating with compatible Python..."
            rm -rf .venv
        fi
    fi

    if [ ! -d ".venv" ]; then
        BASE_PYTHON=$(find_compatible_python)
        if [ -z "$BASE_PYTHON" ]; then
            echo "[SH] Error: No Python installation found."
            exit 1
        fi
        echo "[SH] Creating .venv using: $BASE_PYTHON ($("$BASE_PYTHON" --version 2>&1))"
        "$BASE_PYTHON" -m venv .venv
    fi

    if [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "[SH] Error: Could not find virtual environment activation script."
        exit 1
    fi
fi

PYTHON_CMD=$(command -v python || command -v python3)
echo "[SH] Using Python: $("$PYTHON_CMD" --version 2>&1) ($PYTHON_CMD)"

# 2. Install required packages if missing
if ! "$PYTHON_CMD" -c "import torch, numpy, cv2, matplotlib" 2>/dev/null; then
    echo "[SH] Installing required packages from requirements.txt..."
    "$PYTHON_CMD" -m pip install --upgrade pip
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[SH] NVIDIA GPU detected. Installing PyTorch with CUDA support..."
        "$PYTHON_CMD" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        "$PYTHON_CMD" -m pip install -r requirements.txt
    else
        "$PYTHON_CMD" -m pip install -r requirements.txt
    fi
fi

# 3. Run the requested experiment
MODE="${1:-full}"

if [ "$MODE" = "test_reconstruction" ]; then
    echo "[SH] Running reconstruction test..."
    "$PYTHON_CMD" code/main.py test_reconstruction
elif [ "$MODE" = "full" ]; then
    echo "[SH] Running full DFF pipeline..."
    "$PYTHON_CMD" code/main.py
else
    echo "[SH] Unknown argument: $MODE"
    echo "Usage: bash run.sh [test_reconstruction]"
    exit 1
fi