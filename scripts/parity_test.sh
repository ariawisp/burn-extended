#!/usr/bin/env bash
set -euo pipefail

TMP_DIR=${1:-"/tmp/gptoss_fixture"}
RUST_BIN="${2:-/tmp/model_rust.bin}"
PY_BIN="${3:-/tmp/model_python.bin}"

echo "[parity] Generating tiny fixture at ${TMP_DIR}"
cargo run -q -p burn-extended --bin gptoss_fixture -- -o "${TMP_DIR}"

echo "[parity] Exporting with Rust to ${RUST_BIN}"
cargo run -q -p burn-extended --bin gptoss_export -- -s "${TMP_DIR}" -d "${RUST_BIN}"

echo "[parity] Exporting with Python to ${PY_BIN}"
python ../gpt-oss/gpt_oss/metal/scripts/create-local-model.py -s "${TMP_DIR}" -d "${PY_BIN}"

echo "[parity] Comparing artifacts"
if cmp -s "${RUST_BIN}" "${PY_BIN}"; then
  echo "[parity] OK: files are byte-identical"
  exit 0
else
  echo "[parity] MISMATCH: files differ"
  sha256sum "${RUST_BIN}" "${PY_BIN}" || shasum -a 256 "${RUST_BIN}" "${PY_BIN}"
  exit 1
fi

