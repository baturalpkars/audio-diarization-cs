using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Collections.Generic;
using System.IO;

namespace AudioDiarizationCS.Preprocessing;

public sealed class AudioData
{
    public float[] Samples { get; }
    public int SampleRate { get; }

    public AudioData(float[] samples, int sampleRate)
    {
        Samples = samples;
        SampleRate = sampleRate;
    }
}

public static class AudioLoader
{
    public static AudioData LoadMono(string path, int targetSampleRate = 16000)
    {
        // NAudio's AudioFileReader uses MediaFoundation on macOS for mp3,
        // which depends on mfplat.dll (Windows). For now, require WAV input.
        var ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext != ".wav")
        {
            throw new NotSupportedException(
                $"Unsupported format '{ext}'. Please convert to WAV first (e.g., ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav).");
        }

        using var reader = new WaveFileReader(path);
        ISampleProvider provider = reader.ToSampleProvider();

        // Ensure sample rate
        if (reader.WaveFormat.SampleRate != targetSampleRate)
        {
            provider = new WdlResamplingSampleProvider(provider, targetSampleRate);
        }

        // Ensure mono
        if (provider.WaveFormat.Channels > 1)
        {
            provider = new StereoToMonoSampleProvider(provider);
        }

        var buffer = new float[targetSampleRate * 30];
        var samples = new List<float>();
        int read;
        while ((read = provider.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (var i = 0; i < read; i++)
            {
                samples.Add(buffer[i]);
            }
        }

        return new AudioData(samples.ToArray(), targetSampleRate);
    }
}
