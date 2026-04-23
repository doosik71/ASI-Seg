#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASI_DIR="${PROJECT_ROOT}/ASI"
PYTHON_BIN="${ASI_DIR}/.venv/bin/python"
TRAIN_DATA_DIR="${PROJECT_ROOT}/data/train_data"
SAM_SOURCE="${PROJECT_ROOT}/data/sam_model/sam_vit_h_4b8939.pth"
SAM_COMPAT_DIR="${PROJECT_ROOT}/ckp/sam"
SAM_COMPAT_PATH="${SAM_COMPAT_DIR}/sam_vit_h_4b8939.pth"
CLIP_DIR="${ASI_DIR}/CLIP"

main() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    echo "Run ./scripts/01-init-env.sh first." >&2
    exit 1
  fi

  if [[ ! -d "${TRAIN_DATA_DIR}" ]]; then
    echo "Train data directory not found: ${TRAIN_DATA_DIR}" >&2
    echo "Run ./scripts/04-prepare-data.sh first." >&2
    exit 1
  fi

  if [[ ! -d "${CLIP_DIR}" ]]; then
    echo "CLIP repository not found: ${CLIP_DIR}" >&2
    echo "Run ./scripts/05-setup-clip.sh first." >&2
    exit 1
  fi

  if [[ ! -f "${SAM_SOURCE}" ]]; then
    echo "SAM checkpoint not found: ${SAM_SOURCE}" >&2
    echo "Run ./scripts/03-download-sam.sh first." >&2
    exit 1
  fi

  mkdir -p "${SAM_COMPAT_DIR}"
  if [[ ! -e "${SAM_COMPAT_PATH}" ]]; then
    ln -s "${SAM_SOURCE}" "${SAM_COMPAT_PATH}"
  fi

  echo "Starting training from ${ASI_DIR}"
  (
    cd "${ASI_DIR}"
    "${PYTHON_BIN}" train.py "$@"
  )
}

main "$@"
