using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AudioDiarizationCS.Diarization;

public sealed class SpeakerEmbeddingOnnx : IDisposable
{
    private readonly string _modelPath;

    public SpeakerEmbeddingOnnx(string modelPath)
    {
        _modelPath = modelPath;
    }

    public float[] Run(float[][] melFrames)
    {
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

        using var session = new InferenceSession(_modelPath);
        using var results = session.Run(inputs);
        // outputs: logits, embs
        var embs = results.Last().AsTensor<float>();
        var dims = embs.Dimensions.ToArray();
        if (dims.Length != 2)
        {
            throw new InvalidOperationException($"Unexpected embedding dims: {string.Join(',', dims)}");
        }
        var d = dims[1];
        var vec = new float[d];
        for (var i = 0; i < d; i++)
        {
            vec[i] = embs[0, i];
        }
        return vec;
    }

    public void Dispose()
    {
        // No-op. Sessions are created per Run to avoid dynamic-shape reuse issues.
    }
}
