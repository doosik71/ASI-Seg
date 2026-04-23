#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/ASI/.venv/bin/python"
PREPARE_SCRIPT="${SCRIPT_DIR}/ASI/tools/prepare_train_data.py"
RAW_DATA_ROOT="${SCRIPT_DIR}/data"
OUTPUT_ROOT="${SCRIPT_DIR}/data/train_data"
SAM_CHECKPOINT="${SCRIPT_DIR}/data/sam_model/sam_vit_h_4b8939.pth"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
  echo "Prepare script not found: ${PREPARE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}" >&2
  echo "Run ./scripts/03-download-sam.sh first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
  --dataset all \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  "$@"
