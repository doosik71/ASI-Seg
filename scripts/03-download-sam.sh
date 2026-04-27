#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/data/sam_model"
TARGET_FILE="${TARGET_DIR}/sam_vit_h_4b8939.pth"
PRIMARY_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
FALLBACK_URL="https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth?download=true"
EXPECTED_SHA256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"

mkdir -p "${TARGET_DIR}"

download_with_curl() {
  local url="$1"
  curl -L --fail --progress-bar "${url}" -o "${TARGET_FILE}"
}

download_with_wget() {
  local url="$1"
  wget --show-progress "${url}" -O "${TARGET_FILE}"
}

download_with_available_tool() {
  local url="$1"

  if command -v curl >/dev/null 2>&1; then
    download_with_curl "${url}"
  elif command -v wget >/dev/null 2>&1; then
    download_with_wget "${url}"
  else
    echo "Neither curl nor wget is installed." >&2
    exit 1
  fi
}

verify_checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    local actual_sha256
    actual_sha256="$(sha256sum "${TARGET_FILE}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
      echo "Checksum mismatch for ${TARGET_FILE}" >&2
      echo "Expected: ${EXPECTED_SHA256}" >&2
      echo "Actual:   ${actual_sha256}" >&2
      exit 1
    fi
    echo "Checksum verified: ${actual_sha256}"
  else
    echo "sha256sum not found; skipping checksum verification."
  fi
}

if [[ -f "${TARGET_FILE}" ]]; then
  echo "File already exists: ${TARGET_FILE}"
  verify_checksum
  echo "SAM model is ready."
  echo "Next step: ./scripts/04-prepare-data.sh"
  exit 0
fi

echo "Downloading SAM model from primary source..."
if ! download_with_available_tool "${PRIMARY_URL}"; then
  echo "Primary download failed. Retrying with fallback source..."
  rm -f "${TARGET_FILE}"
  download_with_available_tool "${FALLBACK_URL}"
fi

verify_checksum
echo "Downloaded to ${TARGET_FILE}"
echo "SAM model is ready."
echo "Next step: ./scripts/04-setup-clip.sh"
