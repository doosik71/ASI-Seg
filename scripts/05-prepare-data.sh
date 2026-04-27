#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/ASI/.venv/bin/python"
PREPARE_SCRIPT="${PROJECT_ROOT}/ASI/tools/prepare_train_data.py"
RAW_DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_ROOT="${PROJECT_ROOT}/data/train_data"
SAM_CHECKPOINT="${PROJECT_ROOT}/data/sam_model/sam_vit_h_4b8939.pth"

print_next_step() {
  echo "train data is ready."
  echo "Next step: ./scripts/06-train-model.sh"
}

has_existing_train_data() {
  [[ -d "${OUTPUT_ROOT}" ]] && find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 | read -r _
}

main() {
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

  # if has_existing_train_data; then
  #   echo "Existing train data found at ${OUTPUT_ROOT}. Skipping data preparation."
  #   print_next_step
  #   exit 0
  # fi

  mkdir -p "${OUTPUT_ROOT}"

  echo "Preparing train data into ${OUTPUT_ROOT}"
  echo "This can take a while. Progress bars will be shown below."

  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --dataset all \
    --raw-data-root "${RAW_DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    "$@"

  print_next_step
}

main "$@"
