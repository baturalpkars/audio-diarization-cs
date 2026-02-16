using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Utils;
using NWaves.Windows;

namespace AudioDiarizationCS.Preprocessing;

public static class MelExtractor
{
    private static List<float[]> ExtractInternal(float[] samples, int sampleRate, bool perFeatureNormalize, float dither)
    {
        var fftSize = 512;
        var melBins = 80;

        if (dither > 0)
        {
            var rnd = new Random(0);
            for (var i = 0; i < samples.Length; i++)
            {
                samples[i] += (float)((rnd.NextDouble() * 2.0 - 1.0) * dither);
            }
        }

        var options = new FilterbankOptions
        {
            SamplingRate = sampleRate,
            FrameDuration = 0.025,
            HopDuration = 0.010,
            FilterBankSize = melBins,
            FftSize = fftSize,
            Window = WindowType.Hann,
            SpectrumType = SpectrumType.Power,
            NonLinearity = NonLinearityType.Log10,
            LogFloor = 1e-9f,
            PreEmphasis = 0.0
        };

        var extractor = new FilterbankExtractor(options);
        var feats = extractor.ComputeFrom(samples);

        if (perFeatureNormalize && feats.Count > 0)
        {
            var T = feats.Count;
            var F = feats[0].Length;
            var mean = new float[F];
            var std = new float[F];

            for (var t = 0; t < T; t++)
            {
                for (var f = 0; f < F; f++) mean[f] += feats[t][f];
            }
            for (var f = 0; f < F; f++) mean[f] /= T;

            for (var t = 0; t < T; t++)
            {
                for (var f = 0; f < F; f++)
                {
                    var d = feats[t][f] - mean[f];
                    std[f] += d * d;
                }
            }
            for (var f = 0; f < F; f++)
            {
                std[f] = (float)Math.Sqrt(std[f] / T + 1e-12);
            }

            for (var t = 0; t < T; t++)
            {
                for (var f = 0; f < F; f++)
                {
                    feats[t][f] = (feats[t][f] - mean[f]) / std[f];
                }
            }
        }

        return feats;
    }

    public static List<float[]> ExtractVad(float[] samples, int sampleRate)
        => ExtractInternal(samples, sampleRate, perFeatureNormalize: false, dither: 1e-05f);

    public static List<float[]> ExtractSpk(float[] samples, int sampleRate)
        => ExtractInternal(samples, sampleRate, perFeatureNormalize: true, dither: 1e-05f);
}
