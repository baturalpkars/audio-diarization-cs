using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AudioDiarizationCS.Diarization;

public sealed class VadOnnx : IDisposable
{
    private readonly InferenceSession _session;

    public VadOnnx(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public float[] Run(float[][] melFrames)
    {
        // melFrames: [T][80] -> input [1,80,T]
        var t = melFrames.Length;
        if (t == 0) return Array.Empty<float>();
        var bins = melFrames[0].Length;

        var input = new DenseTensor<float>(new[] { 1, bins, t });
        for (var i = 0; i < t; i++)
        {
            for (var b = 0; b < bins; b++)
            {
                input[0, b, i] = melFrames[i][b];
            }
        }

        var len = new DenseTensor<long>(new[] { 1 });
        len[0] = t;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", input),
            NamedOnnxValue.CreateFromTensor("length", len),
        };

        using var results = _session.Run(inputs);
        var logits = results.First().AsTensor<float>();
        var dimsStr = string.Join(",", logits.Dimensions.ToArray());
        // Console.WriteLine($"VAD logits dims: {dimsStr}");

        // logits shape: [T,2] or [1,T,2] depending on export.
        // Flatten to [T,2].
        var dims = logits.Dimensions.ToArray();
        int time, classes;
        if (dims.Length == 2)
        {
            time = dims[0];
            classes = dims[1];
        }
        else if (dims.Length == 3)
        {
            time = dims[1];
            classes = dims[2];
        }
        else
        {
            throw new InvalidOperationException($"Unexpected VAD logits dims: {string.Join(',', dims)}");
        }

        if (classes != 2)
        {
            throw new InvalidOperationException($"Expected 2 classes, got {classes}");
        }

        var speechProb = new float[time];
        for (var i = 0; i < time; i++)
        {
            float log0, log1;
            if (dims.Length == 2)
            {
                log0 = logits[i, 0];
                log1 = logits[i, 1];
            }
            else
            {
                log0 = logits[0, i, 0];
                log1 = logits[0, i, 1];
            }

            // softmax for class 1
            var m = Math.Max(log0, log1);
            var e0 = Math.Exp(log0 - m);
            var e1 = Math.Exp(log1 - m);
            speechProb[i] = (float)(e1 / (e0 + e1));
        }

        return speechProb;
    }

    public float[] RunSliding(
        float[][] melFrames,
        double frameHopSec = 0.010,
        double windowSec = 0.63,
        double shiftSec = 0.01,
        string smoothingMethod = "none",
        double overlap = 0.5)
    {
        var totalFrames = melFrames.Length;
        if (totalFrames == 0) return Array.Empty<float>();

        var winFrames = Math.Max(1, (int)Math.Round(windowSec / frameHopSec));
        var shiftFrames = Math.Max(1, (int)Math.Round(shiftSec / frameHopSec));

        var sum = new float[totalFrames];
        var count = new int[totalFrames];
        var windowMeans = new List<float>();

        for (var i = 0; i + winFrames <= totalFrames; i += shiftFrames)
        {
            var window = new float[winFrames][];
            Array.Copy(melFrames, i, window, 0, winFrames);
            var p = Run(window);
            var end = i + winFrames;
            if (p.Length == 0) continue;

            // Most exports return per-frame probabilities for the window.
            // If we only got a single value, broadcast it across the window.
            if (p.Length == 1)
            {
                var val = p[0];
                for (var t = i; t < end; t++)
                {
                    sum[t] += val;
                    count[t] += 1;
                }
                windowMeans.Add(val);
            }
            else
            {
                var limit = Math.Min(p.Length, winFrames);
                for (var t = 0; t < limit; t++)
                {
                    var idx = i + t;
                    sum[idx] += p[t];
                    count[idx] += 1;
                }
                // mean over this window for optional smoothing
                var mean = 0f;
                for (var t = 0; t < limit; t++) mean += p[t];
                windowMeans.Add(mean / limit);
            }
        }

        var probs = new float[totalFrames];
        for (var i = 0; i < totalFrames; i++)
        {
            probs[i] = count[i] > 0 ? (sum[i] / count[i]) : 0f;
        }

        if (string.Equals(smoothingMethod, "none", StringComparison.OrdinalIgnoreCase))
        {
            return probs;
        }

        return ApplyOverlapSmoothing(
            windowMeans.ToArray(),
            totalFrames,
            winFrames,
            shiftFrames,
            overlap,
            smoothingMethod);
    }

    private static float[] ApplyOverlapSmoothing(
        float[] frame,
        int targetFrames,
        int winFrames,
        int shiftFrames,
        double overlap,
        string smoothingMethod)
    {
        if (frame.Length == 0) return new float[targetFrames];

        var seg = winFrames + 1;
        var jumpOnTarget = (int)Math.Round(seg * (1 - overlap));
        var jumpOnFrame = Math.Max(1, (int)Math.Round(jumpOnTarget / (double)shiftFrames));

        if (jumpOnFrame < 1) jumpOnFrame = 1;

        var preds = new float[targetFrames];
        var counts = new int[targetFrames];

        if (smoothingMethod.Equals("mean", StringComparison.OrdinalIgnoreCase))
        {
            for (var i = 0; i < frame.Length; i++)
            {
                if (i % jumpOnFrame != 0) continue;
                var start = i * shiftFrames;
                var end = Math.Min(start + seg, targetFrames);
                for (var j = start; j < end; j++)
                {
                    preds[j] += frame[i];
                    counts[j] += 1;
                }
            }

            float last = 0f;
            for (var i = 0; i < targetFrames; i++)
            {
                if (counts[i] > 0)
                {
                    preds[i] /= counts[i];
                    last = preds[i];
                }
                else
                {
                    preds[i] = last;
                }
            }
            return preds;
        }

        if (smoothingMethod.Equals("median", StringComparison.OrdinalIgnoreCase))
        {
            var buckets = new List<float>[targetFrames];
            for (var i = 0; i < targetFrames; i++) buckets[i] = new List<float>();

            for (var i = 0; i < frame.Length; i++)
            {
                if (i % jumpOnFrame != 0) continue;
                var start = i * shiftFrames;
                var end = Math.Min(start + seg, targetFrames);
                for (var j = start; j < end; j++)
                {
                    buckets[j].Add(frame[i]);
                }
            }

            float last = 0f;
            for (var i = 0; i < targetFrames; i++)
            {
                if (buckets[i].Count == 0)
                {
                    preds[i] = last;
                    continue;
                }
                buckets[i].Sort();
                var mid = buckets[i].Count / 2;
                preds[i] = buckets[i][mid];
                last = preds[i];
            }
            return preds;
        }

        throw new ArgumentException("smoothingMethod should be 'none', 'mean', or 'median'");
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
