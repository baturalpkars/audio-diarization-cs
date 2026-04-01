#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <wav_dir> <out_dir> <vad.onnx> <spk.onnx>"
  exit 1
fi

WAV_DIR="$1"
OUT_DIR="$2"
VAD="$3"
SPK="$4"

mkdir -p "$OUT_DIR"

for f in "$WAV_DIR"/*.wav; do
  [ -e "$f" ] || continue
  out="$OUT_DIR/$(basename "${f%.wav}_segments.json")"
  echo "Processing: $f -> $out"
  dotnet run -- --vad "$VAD" --spk "$SPK" --audio "$f" --out "$out" \
    --vad-onset 0.8 --vad-offset 0.4 --vad-min-silence 0.3 --vad-min-speech 0.0 \
    --vad-pad-onset 0.0 --vad-pad-offset 0.0 --vad-smoothing none --vad-overlap 0.5 \
    --cluster-search 1 --merge-gap 0.25 --min-seg 1.0 --subseg-win 2.0 --subseg-shift 1.0
done
