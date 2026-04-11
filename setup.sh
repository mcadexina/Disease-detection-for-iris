#!/bin/bash
set -e

echo "Upgrading pip..."
python -m pip install --upgrade pip

# If this repo uses Git LFS (e.g. *.h5 models), pull the actual binaries.
# This is best-effort: it will no-op if git-lfs isn't installed.
if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
  echo "Pulling Git LFS assets..."
  git lfs install --local || true
  git lfs pull || true
fi

echo "Installing dependencies..."
python -m pip install -r requirements.txt
