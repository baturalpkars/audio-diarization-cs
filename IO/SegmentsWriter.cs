using System.Text.Json;

namespace AudioDiarizationCS.IO;

public static class SegmentsWriter
{
    public static void Write(string path, IReadOnlyList<(double start, double end, string speaker)> segments)
    {
        var payload = segments.Select(s => new
        {
            start = s.start,
            end = s.end,
            speaker = s.speaker
        }).ToList();

        var opts = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, JsonSerializer.Serialize(payload, opts));
    }
}
