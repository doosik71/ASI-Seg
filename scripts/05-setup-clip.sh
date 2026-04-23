#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLIP_DIR="${PROJECT_ROOT}/ASI/CLIP"
CLIP_REPO_URL="https://github.com/openai/CLIP.git"

print_next_step() {
  echo "CLIP repository is ready."
  echo "Next step: ./scripts/06-train-model.sh"
}

main() {
  if [[ -d "${CLIP_DIR}" ]]; then
    echo "CLIP repository already exists at ${CLIP_DIR}. Skipping clone."
    print_next_step
    exit 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required but not installed." >&2
    exit 1
  fi

  echo "Cloning CLIP into ${CLIP_DIR}"
  git clone "${CLIP_REPO_URL}" "${CLIP_DIR}"

  print_next_step
}

main "$@"
