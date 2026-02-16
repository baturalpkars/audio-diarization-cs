using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using ClosedXML.Excel;

namespace AudioDiarizationCS.Alignment;

public sealed class FaceReaderExtractOptions
{
    public string XlsxPath { get; set; } = "";
    public string? Sheet { get; set; } = null;
    public string OutDir { get; set; } = "";
    public double QualityThreshold { get; set; } = 0.7;
    public bool KeepAll { get; set; } = false;
    public int MinBlockRows { get; set; } = 200;
}

public static class FaceReaderExtractor
{
    private static readonly Regex TimeRe = new(@"^(?<h>\d+):(?<m>\d+):(?<s>\d+)\.(?<ms>\d+)$", RegexOptions.Compiled);

    public static void Run(FaceReaderExtractOptions opt)
    {
        if (!File.Exists(opt.XlsxPath)) throw new FileNotFoundException($"Missing xlsx: {opt.XlsxPath}");
        Directory.CreateDirectory(opt.OutDir);

        using var wb = new XLWorkbook(opt.XlsxPath);
        var ws = !string.IsNullOrWhiteSpace(opt.Sheet) ? wb.Worksheet(opt.Sheet) : wb.Worksheets.First();

        var headerRow = ws.FirstRowUsed();
        var headers = headerRow.Cells().Select(c => c.GetString()).ToList();
        var timeCol = PickCol(headers, new[] { "Video Time", "Time" });
        var qualityCol = PickCol(headers, new[] { "Quality" });
        var mouthCol = PickCol(headers, new[] { "Mouth" });
        var au10Col = PickCol(headers, new[] { "Action Unit 10 - Upper Lip Raiser", "Action Unit 10", "AU10", "Upper Lip Raiser" });
        var au12Col = PickCol(headers, new[] { "Action Unit 12 - Lip Corner Puller", "Action Unit 12", "AU12", "Lip Corner Puller" });
        var au25Col = PickCol(headers, new[] { "Action Unit 25 - Lips Part", "Action Unit 25", "AU25", "Lips Part" });
        var au26Col = PickCol(headers, new[] { "Action Unit 26 - Jaw Drop", "Action Unit 26", "AU26", "Jaw Drop" });

        var rows = new List<RowData>();
        foreach (var row in ws.RowsUsed().Skip(1))
        {
            var timeRaw = row.Cell(timeCol).GetString();
            if (string.IsNullOrWhiteSpace(timeRaw)) continue;
            var timeSec = TimeToSeconds(timeRaw);
            var mouthStr = row.Cell(mouthCol).GetString();
            var mouthOpen = MouthOpenToInt(mouthStr);
            var au10 = ParseFloat(row.Cell(au10Col).GetString());
            var au12 = ParseFloat(row.Cell(au12Col).GetString());
            var au25 = ParseFloat(row.Cell(au25Col).GetString());
            var au26 = ParseFloat(row.Cell(au26Col).GetString());
            var quality = ParseFloat(row.Cell(qualityCol).GetString());

            if (!opt.KeepAll && quality < opt.QualityThreshold) continue;

            var mouthScore = 0.45f * au25 + 0.35f * au26 + 0.15f * au10 + 0.05f * au12;
            rows.Add(new RowData
            {
                TimeSec = timeSec,
                Quality = quality,
                MouthOpen = mouthOpen,
                Au10 = au10,
                Au12 = au12,
                Au25 = au25,
                Au26 = au26,
                MouthScoreRaw = mouthScore
            });
        }

        var blocks = SplitByTimeResets(rows);
        blocks = blocks.Where(b => b.Count >= opt.MinBlockRows).ToList();

        var summary = new List<Dictionary<string, string>>();
        for (var i = 0; i < blocks.Count; i++)
        {
            var pid = $"participant_{i + 1:00d}";
            var outPath = Path.Combine(opt.OutDir, $"{pid}.csv");
            WriteParticipantCsv(outPath, blocks[i]);
            summary.Add(new Dictionary<string, string>
            {
                ["participant_block"] = pid,
                ["n_rows"] = blocks[i].Count.ToString(CultureInfo.InvariantCulture),
                ["t_min"] = blocks[i].Min(r => r.TimeSec).ToString(CultureInfo.InvariantCulture),
                ["t_max"] = blocks[i].Max(r => r.TimeSec).ToString(CultureInfo.InvariantCulture),
                ["mean_quality"] = blocks[i].Average(r => r.Quality).ToString(CultureInfo.InvariantCulture)
            });
        }

        WriteCsv(Path.Combine(opt.OutDir, "participants_summary.csv"), summary);

        Console.WriteLine($"Sheet: {ws.Name}");
        Console.WriteLine($"Detected participant blocks: {blocks.Count}");
        Console.WriteLine($"Wrote: {Path.Combine(opt.OutDir, "participants_summary.csv")}");
        Console.WriteLine($"Wrote participant CSVs to: {opt.OutDir}");
    }

    private static int PickCol(List<string> headers, IEnumerable<string> candidates)
    {
        var map = headers.Select((h, i) => (h, i: i + 1)).ToDictionary(x => x.h.Trim(), x => x.i);
        foreach (var cand in candidates)
        {
            if (map.TryGetValue(cand, out var idx)) return idx;
        }
        // case-insensitive exact
        var lower = headers.Select((h, i) => (h, i: i + 1, low: h.ToLowerInvariant())).ToList();
        foreach (var cand in candidates)
        {
            var candLow = cand.ToLowerInvariant();
            var match = lower.FirstOrDefault(x => x.low == candLow);
            if (match.i != 0) return match.i;
        }
        throw new KeyNotFoundException($"Could not find any of these columns: {string.Join(", ", candidates)}");
    }

    private static float TimeToSeconds(string t)
    {
        var m = TimeRe.Match(t.Trim());
        if (!m.Success) throw new FormatException($"Unexpected time format: {t}");
        var h = int.Parse(m.Groups["h"].Value);
        var mi = int.Parse(m.Groups["m"].Value);
        var s = int.Parse(m.Groups["s"].Value);
        var ms = int.Parse(m.Groups["ms"].Value);
        return h * 3600 + mi * 60 + s + ms / 1000f;
    }

    private static int MouthOpenToInt(string x)
    {
        return x.Trim().Equals("open", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
    }

    private static float ParseFloat(string s)
    {
        if (float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) return v;
        if (float.TryParse(s, NumberStyles.Float, CultureInfo.CurrentCulture, out v)) return v;
        return 0f;
    }

    private static List<List<RowData>> SplitByTimeResets(List<RowData> rows)
    {
        var blocks = new List<List<RowData>>();
        if (rows.Count == 0) return blocks;
        var current = new List<RowData> { rows[0] };
        for (var i = 1; i < rows.Count; i++)
        {
            if (rows[i].TimeSec < rows[i - 1].TimeSec)
            {
                blocks.Add(current);
                current = new List<RowData>();
            }
            current.Add(rows[i]);
        }
        blocks.Add(current);
        return blocks;
    }

    private static void WriteParticipantCsv(string path, List<RowData> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("time_sec,quality,mouth_open,au10,au12,au25,au26,mouth_score_raw");
        foreach (var r in rows)
        {
            sb.AppendLine(string.Join(",",
                r.TimeSec.ToString(CultureInfo.InvariantCulture),
                r.Quality.ToString(CultureInfo.InvariantCulture),
                r.MouthOpen.ToString(CultureInfo.InvariantCulture),
                r.Au10.ToString(CultureInfo.InvariantCulture),
                r.Au12.ToString(CultureInfo.InvariantCulture),
                r.Au25.ToString(CultureInfo.InvariantCulture),
                r.Au26.ToString(CultureInfo.InvariantCulture),
                r.MouthScoreRaw.ToString(CultureInfo.InvariantCulture)));
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static void WriteCsv(string path, List<Dictionary<string, string>> rows)
    {
        if (rows.Count == 0)
        {
            File.WriteAllText(path, "");
            return;
        }
        var headers = rows.SelectMany(r => r.Keys).Distinct().ToList();
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(",", headers));
        foreach (var row in rows)
        {
            var vals = headers.Select(h => row.TryGetValue(h, out var v) ? v : "").Select(EscapeCsv);
            sb.AppendLine(string.Join(",", vals));
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static string EscapeCsv(string v)
    {
        if (v.Contains(',') || v.Contains('\"') || v.Contains('\n'))
        {
            var s = v.Replace("\"", "\"\"");
            return $"\"{s}\"";
        }
        return v;
    }

    private sealed class RowData
    {
        public float TimeSec { get; set; }
        public float Quality { get; set; }
        public int MouthOpen { get; set; }
        public float Au10 { get; set; }
        public float Au12 { get; set; }
        public float Au25 { get; set; }
        public float Au26 { get; set; }
        public float MouthScoreRaw { get; set; }
    }
}
