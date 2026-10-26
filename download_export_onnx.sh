#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --locked ./download_export_onnx.py "$@"
