# Audio Diarization (C#) — NeMo-Compatible Offline Pipeline

This repository contains a pure C# pipeline for offline speaker diarization and face–speaker alignment. It mirrors the NeMo diarization flow and the Python alignment logic used in the original project, while being deployable on-device without network access.

## Overview

**Pipeline stages**
1. **Diarization (ONNX)**  
   VAD → speaker embeddings → NMESC clustering → diarization segments
2. **FaceReader extraction**  
   Convert detailed FaceReader `.xlsx` into per-participant CSVs
3. **Face–speaker alignment**  
   Align diarized speakers to participants using mouth activity features

## Requirements

- .NET 7 SDK
- ffmpeg (for converting mp4 → wav)
- ONNX Runtime (via NuGet)
- FaceReader detailed export `.xlsx`

## Project Structure

```
AudioDiarizationCS/
  Alignment/                # FaceReader extractor + alignment logic
  Clustering/               # NMESC + spectral clustering
  Diarization/              # VAD + embedding inference
  IO/                       # JSON writer
  Preprocessing/            # Audio + mel extraction
  Program.cs                # CLI entrypoint
  data/                     # Local inputs (ignored by git)
  runs/                     # Local outputs (ignored by git)
```

## Models

Place ONNX models (and their `.data` files if exported) under a local folder, e.g.:

```
models/
  vad_multilingual_marblenet.onnx
  vad_multilingual_marblenet.onnx.data
  titanet_large.onnx
  titanet_large.onnx.data
```

> Note: ONNX Runtime requires the `.onnx.data` files if your export produced them.

## Audio Preparation

Use ffmpeg to convert video to wav:

```bash
/Users/baturalpkarslioglu/anaconda3/bin/ffmpeg \
  -i /path/to/video.mp4 \
  -ar 16000 -ac 1 -vn /path/to/audio.wav
```

## Diarization (Batch)

Example for batch processing 5 test files in `data/`:

```bash
dotnet run -- \
  --vad models/vad_multilingual_marblenet.onnx \
  --spk models/titanet_large.onnx \
  --batch-dir /Users/baturalpkarslioglu/Desktop/BK/Internship/Audio-Diarization-CS/AudioDiarizationCS/data \
  --vad-onset 0.8 \
  --vad-offset 0.4 \
  --vad-min-silence 0.3 \
  --vad-min-speech 0.0 \
  --vad-pad-onset 0.0 \
  --vad-pad-offset 0.0 \
  --vad-smoothing none \
  --vad-overlap 0.5 \
  --cluster-search 1 \
  --merge-gap 0.25 \
  --min-seg 1.0 \
  --subseg-win 2.0 \
  --subseg-shift 1.0
```

**Outputs**
```
csharp_test_01_segments.json
csharp_test_02_segments.json
...
```

## FaceReader Extraction

Convert detailed FaceReader `.xlsx` into per-participant CSVs:

```bash
dotnet run -- \
  --facereader-xlsx /path/to/facereader_detailed_01.xlsx \
  --facereader-out /path/to/test_01_participants
```

Optional:
```
--facereader-sheet <sheetname>
--facereader-quality-threshold 0.7
--facereader-keep-all
--facereader-min-block-rows 200
```

Output:
```
participant_01.csv
participant_02.csv
participants_summary.csv
```

## Face–Speaker Alignment

Run alignment on diarization output + FaceReader participants:

```bash
dotnet run -- \
  --segments /path/to/csharp_test_01_segments.json \
  --participants-dir /path/to/test_01_participants \
  --out /path/to/runs/align_test_01 \
  --align-merge-gap 0.25
```

**Alignment outputs**
```
participant_debug.csv
segment_vvad_debug.csv
score_matrix.csv
speaker_to_participant.json
aligned_segments.json
```

### Alignment Defaults (match Python)
- `min_seg = 0.6`
- `margin_ratio = 1.08`
- `best_min = 0.06`
- `stick_ratio = 0.92`
- `use_global_map = true`
- `smooth_ms = 240`

Override via CLI if needed:
```
--align-min-seg 0.6
--align-margin-ratio 1.08
--align-best-min 0.06
--align-stick-ratio 0.92
--align-use-global-map
--align-smooth-ms 240
--align-merge-gap 0.25
```

## Evaluation (Python)

You can evaluate C# alignment outputs with the existing Python evaluator:

```bash
PYTHONPATH="/Users/baturalpkarslioglu/Desktop/BK/Internship/Audio Diarization" \
python "/Users/baturalpkarslioglu/Desktop/BK/Internship/Audio Diarization/scripts/run_eval_alignment.py" \
  --cases "/Users/baturalpkarslioglu/Desktop/BK/Internship/Audio-Diarization-CS/AudioDiarizationCS/csharp_eval_cases.json" \
  --skip-extract --skip-align
```

## Notes

- ONNX diarization is offline and device‑local.
- The C# alignment logic mirrors the Python implementation.
- For production, keep `data/` and `runs/` out of Git (see `.gitignore`).

