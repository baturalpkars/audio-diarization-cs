#!/usr/bin/env bash
set -euo pipefail

WAV_DIR="$1"
OUT_DIR="$2"
N="${3:-5}"

VAD_MODEL="$4"
SPK_MODEL="$5"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
echo "file,segments,speakers" > "$SUMMARY"

count=0
for f in "$WAV_DIR"/*_bp.wav; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f" .wav)"
  out_json="$OUT_DIR/${base}_segments.json"

  echo "Processing: $f -> $out_json"

  dotnet run -- \
    --vad "$VAD_MODEL" \
    --spk "$SPK_MODEL" \
    --audio "$f" \
    --out "$out_json" \
    --vad-onset 0.95 --vad-offset 0.8 --vad-min-silence 0.6 --vad-min-speech 0.2 \
    --vad-pad-onset 0.0 --vad-pad-offset 0.0 --vad-smoothing none --vad-overlap 0.5 \
    --cluster-search 0 --merge-gap 0.6 --min-seg 1.0 \
    --subseg-win 6.0 --subseg-shift 6.0

  segs=$(python3 - <<PY
import json
with open("$out_json") as f:
    d=json.load(f)
print(len(d))
PY
)
  spks=$(python3 - <<PY
import json
with open("$out_json") as f:
    d=json.load(f)
print(len(set(x["speaker"] for x in d)))
PY
)

  echo "$base,$segs,$spks" >> "$SUMMARY"

  count=$((count+1))
  if [[ "$count" -ge "$N" ]]; then
    break
  fi
done

echo "Done. Summary: $SUMMARY"
