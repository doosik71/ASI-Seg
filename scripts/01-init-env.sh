#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASI_DIR="${PROJECT_ROOT}/ASI"
VENV_DIR="${ASI_DIR}/.venv"

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

ensure_uv_on_path() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "uv not found. Installing uv..."

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  else
    echo "Neither curl nor wget is available, so uv cannot be installed automatically." >&2
    exit 1
  fi

  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation finished, but uv is still not on PATH." >&2
    echo "Try reopening your shell, then rerun this script." >&2
    exit 1
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

main() {
  if [[ ! -d "${ASI_DIR}" ]]; then
    echo "ASI directory not found: ${ASI_DIR}" >&2
    exit 1
  fi

  ensure_uv_on_path

  echo "Using uv: $(command -v uv)"
  uv --version

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    uv venv "${VENV_DIR}"
  else
    echo "Virtual environment already exists: ${VENV_DIR}"
  fi

  echo "Syncing dependencies in ${ASI_DIR}..."
  (
    cd "${ASI_DIR}"
    uv sync
  )

  activate_venv

  echo
  echo "Environment is ready."
  echo "Next step: ./scripts/02-download-endovis.sh"

  if ! is_sourced; then
    echo
    echo "Note: because this script was executed, the virtual environment activation"
    echo "does not persist in your current shell."
    echo "To activate it now, run:"
    echo "source ASI/.venv/bin/activate"
  fi
}

main "$@"
