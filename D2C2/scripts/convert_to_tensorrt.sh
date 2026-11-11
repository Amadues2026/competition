#!/usr/bin/env bash
# Build an FP16-enabled engine from an explicit ONNX model. No old engine is removed.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/best.onnx /path/to/new.engine" >&2
    exit 2
fi
onnx_path="$1"
engine_path="$2"
trtexec_bin="${TRTEXEC:-trtexec}"
[[ -f "$onnx_path" ]] || { echo "ONNX model not found: $onnx_path" >&2; exit 1; }
[[ ! -e "$engine_path" ]] || { echo "Output already exists: $engine_path" >&2; exit 1; }
command -v "$trtexec_bin" >/dev/null || { echo "trtexec is not available" >&2; exit 1; }
mkdir -p "$(dirname "$engine_path")"
"$trtexec_bin" --onnx="$onnx_path" --saveEngine="$engine_path" --fp16
[[ -s "$engine_path" ]] || { echo "No engine was generated" >&2; exit 1; }
echo "Engine generated: $engine_path"
