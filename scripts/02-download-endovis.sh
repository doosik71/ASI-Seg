#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
TEMP_DIR="${DATA_DIR}/temp"

ENDOVIS2017_URL="https://zenodo.org/records/10527017/files/endovis2017.zip?download=1"
ENDOVIS2018_URL="https://zenodo.org/records/10527017/files/endovis2018.zip?download=1"

ENDOVIS2017_ARCHIVE="${TEMP_DIR}/endovis2017.zip"
ENDOVIS2018_ARCHIVE="${TEMP_DIR}/endovis2018.zip"

ENDOVIS2017_DIR="${DATA_DIR}/endovis2017"
ENDOVIS2018_DIR="${DATA_DIR}/endovis2018"

download_file() {
  local url="$1"
  local archive_path="$2"

  if [[ -f "${archive_path}" ]]; then
    echo "Using existing archive: ${archive_path}"
    return 0
  fi

  echo "Downloading ${url}"

  if command -v wget >/dev/null 2>&1; then
    wget -O "${archive_path}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail "${url}" -o "${archive_path}"
  else
    echo "Neither wget nor curl is available." >&2
    exit 1
  fi
}

extract_if_needed() {
  local name="$1"
  local archive_path="$2"
  local output_dir="$3"

  if [[ -d "${output_dir}" ]]; then
    echo "${name} already exists at ${output_dir}. Skipping download and extraction."
    return 0
  fi

  echo "Extracting ${archive_path} to ${DATA_DIR}"
  unzip -oq "${archive_path}" -d "${DATA_DIR}"

  if [[ -d "${output_dir}" ]]; then
    echo "${name} is ready at ${output_dir}"
  else
    echo "Extraction finished, but expected directory was not found: ${output_dir}" >&2
    exit 1
  fi
}

main() {
  mkdir -p "${DATA_DIR}" "${TEMP_DIR}"

  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required but not installed." >&2
    exit 1
  fi

  if [[ ! -d "${ENDOVIS2017_DIR}" ]]; then
    download_file "${ENDOVIS2017_URL}" "${ENDOVIS2017_ARCHIVE}"
    extract_if_needed "endovis2017" "${ENDOVIS2017_ARCHIVE}" "${ENDOVIS2017_DIR}"
  else
    echo "endovis2017 already exists at ${ENDOVIS2017_DIR}. Skipping."
  fi

  if [[ ! -d "${ENDOVIS2018_DIR}" ]]; then
    download_file "${ENDOVIS2018_URL}" "${ENDOVIS2018_ARCHIVE}"
    extract_if_needed "endovis2018" "${ENDOVIS2018_ARCHIVE}" "${ENDOVIS2018_DIR}"
  else
    echo "endovis2018 already exists at ${ENDOVIS2018_DIR}. Skipping."
  fi

  echo "Endovis dataset is ready."
  echo "Next step: ./scripts/03-download-sam.sh"
}

main "$@"
