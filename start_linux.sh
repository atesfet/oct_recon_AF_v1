#!/usr/bin/env bash
# OCT reconstruction web app launcher (Linux / macOS).
# First run: creates the conda env "oct_reconstruction" (or a local .venv), adds GPU
# support when an NVIDIA driver is present, then starts the app and opens the browser.
set -e
cd "$(dirname "$0")"
ENV_NAME="${OCT_ENV_NAME:-oct_reconstruction}"

find_conda() {
  for c in "${CONDA_EXE:-}" "$(command -v mamba 2>/dev/null)" "$(command -v conda 2>/dev/null)" \
           "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/anaconda3/bin/conda" "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
           "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

if CONDA="$(find_conda)"; then
  echo "[launcher] using conda: $CONDA"
  if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[launcher] creating environment '$ENV_NAME' (first run only, a few minutes) ..."
    "$CONDA" env create -n "$ENV_NAME" -f environment.yml
  fi
  "$CONDA" run -n "$ENV_NAME" python tools/setup_gpu.py --conda "$CONDA" --env "$ENV_NAME" || true
  exec "$CONDA" run -n "$ENV_NAME" --no-capture-output python launch.py "$@"
fi

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "Neither conda nor Python 3 found. Install Miniforge: https://conda-forge.org/download/" >&2
  exit 1
fi
# venv lives in the user's home (the repo may sit on exFAT/NTFS drives without symlinks)
VENV="${OCT_VENV:-$HOME/.octrecon_venv}"
echo "[launcher] conda not found -> using a Python virtual environment in $VENV"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements.txt
"$VENV/bin/python" tools/setup_gpu.py --pip || true
exec "$VENV/bin/python" launch.py "$@"
