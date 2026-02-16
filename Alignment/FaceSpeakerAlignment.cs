using System.Globalization;
using System.Text;
using System.Text.Json;

namespace AudioDiarizationCS.Alignment;

public sealed class AlignmentOptions
{
    public string SegmentsPath { get; set; } = "";
    public string ParticipantsDir { get; set; } = "";
    public string OutDir { get; set; } = "";
    public double MinSegSec { get; set; } = 0.6;
    public double AvOffsetSec { get; set; } = 0.0;
    public double SmoothMs { get; set; } = 240.0;
    public double MarginRatio { get; set; } = 1.08;
    public double BestMin { get; set; } = 0.06;
    public double StickRatio { get; set; } = 0.92;
    public bool UseGlobalMap { get; set; } = true;
    public double AmbiguousGlobalMaxSec { get; set; } = -1.0;
    public double AmbiguousGlobalMaxRatio { get; set; } = 0.05;
    public double AmbiguousLocalMargin { get; set; } = 1.01;
    public double SoftmaxTemp { get; set; } = 0.08;
    public double MergeGapSec { get; set; } = 0.0;
    public int MaxParticipants { get; set; } = 0;
    public double MinSpdMean { get; set; } = 0.0;
}

public static class FaceSpeakerAlignment
{
    private sealed class Segment
    {
        public double start { get; set; }
        public double end { get; set; }
        public string speaker { get; set; } = "";
    }

    public static void Run(AlignmentOptions opt)
    {
        if (!File.Exists(opt.SegmentsPath)) throw new FileNotFoundException($"Missing segments: {opt.SegmentsPath}");
        if (!Directory.Exists(opt.ParticipantsDir)) throw new DirectoryNotFoundException(opt.ParticipantsDir);
        Directory.CreateDirectory(opt.OutDir);

        var segments = JsonSerializer.Deserialize<List<Segment>>(File.ReadAllText(opt.SegmentsPath)) ?? new List<Segment>();
        var participants = LoadParticipants(opt.ParticipantsDir);

        // Adaptive ambiguous threshold
        if (opt.AmbiguousGlobalMaxSec <= 0)
        {
            var tMin = segments.Count > 0 ? segments.Min(s => s.start) : 0.0;
            var tMax = segments.Count > 0 ? segments.Max(s => s.end) : 0.0;
            var totalDur = Math.Max(0.0, tMax - tMin);
            var adaptive = totalDur * opt.AmbiguousGlobalMaxRatio;
            opt.AmbiguousGlobalMaxSec = Math.Max(5.0, Math.Min(30.0, adaptive));
        }

        var speakers = segments.Select(s => s.speaker).Distinct().OrderBy(s => s).ToList();
        var partIds = participants.Keys.OrderBy(s => s).ToList();
        var speakerToI = speakers.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i);

        var scoreLookup = new Dictionary<string, (float[] t, float[] spd)>();
        var openLookup = new Dictionary<string, (float[] t, float[] open)>();
        var participantDebugRows = new List<Dictionary<string, string>>();
        var spdMeans = new Dictionary<string, double>();

        foreach (var pid in partIds)
        {
            var df = participants[pid];
            var t = df.Time;
            var mouth = df.MouthScoreRaw;

            var dt = MedianDiff(t);
            if (dt <= 0) dt = 0.04f;
            var window = Math.Max(1, (int)Math.Round((opt.SmoothMs / 1000.0) / dt));
            var mouthSmooth = SmoothSeries(mouth, window);

            var p10 = Percentile(mouthSmooth, 10);
            var p90 = Percentile(mouthSmooth, 90);
            var denom = (p90 - p10) > 1e-6 ? (p90 - p10) : 1.0;
            var mouthNorm = mouthSmooth.Select(v => (float)Math.Clamp((v - p10) / denom, 0.0, 1.0)).ToArray();

            var spd = MouthSpeed(mouthNorm, dt);
            var p95 = Percentile(spd, 95);
            var spdNorm = spd.Select(v => (float)Math.Clamp(v / (p95 + 1e-6f), 0.0, 1.0)).ToArray();

            scoreLookup[pid] = (t, spdNorm);
            openLookup[pid] = (df.Time, df.MouthOpen);
            spdMeans[pid] = Mean(spdNorm);

            participantDebugRows.Add(new Dictionary<string, string>
            {
                ["participant"] = pid,
                ["dt"] = dt.ToString(CultureInfo.InvariantCulture),
                ["window"] = window.ToString(CultureInfo.InvariantCulture),
                ["p10"] = p10.ToString(CultureInfo.InvariantCulture),
                ["p90"] = p90.ToString(CultureInfo.InvariantCulture),
                ["mean_raw"] = Mean(mouth).ToString(CultureInfo.InvariantCulture),
                ["mean_smooth"] = Mean(mouthSmooth).ToString(CultureInfo.InvariantCulture),
                ["mean_norm"] = Mean(mouthNorm).ToString(CultureInfo.InvariantCulture),
                ["spd_p95"] = p95.ToString(CultureInfo.InvariantCulture),
                ["spd_mean"] = Mean(spdNorm).ToString(CultureInfo.InvariantCulture),
            });
        }

        // Optional participant activity filter
        if (opt.MinSpdMean > 0 || opt.MaxParticipants > 0)
        {
            var filtered = spdMeans
                .Where(kv => kv.Value >= opt.MinSpdMean)
                .OrderByDescending(kv => kv.Value)
                .Select(kv => kv.Key)
                .ToList();

            if (opt.MaxParticipants > 0 && filtered.Count > opt.MaxParticipants)
            {
                filtered = filtered.Take(opt.MaxParticipants).ToList();
            }

            if (filtered.Count > 0)
            {
                partIds = filtered;
            }
        }

        foreach (var row in participantDebugRows)
        {
            var pid = row["participant"];
            row["kept"] = partIds.Contains(pid).ToString();
        }

        WriteCsv(Path.Combine(opt.OutDir, "participant_debug.csv"), participantDebugRows);

        var globalScore = new float[speakers.Count, partIds.Count];
        var segmentDebugRows = new List<Dictionary<string, string>>();
        var segKeyToDebug = new Dictionary<string, Dictionary<string, string>>();

        foreach (var seg in segments)
        {
            var start = seg.start + opt.AvOffsetSec;
            var end = seg.end + opt.AvOffsetSec;
            var dur = end - start;
            if (dur < opt.MinSegSec) continue;

            var si = speakerToI[seg.speaker];
            var perP = new Dictionary<string, float>();

            foreach (var pid in partIds)
            {
                var (tSpd, spd) = scoreLookup[pid];
                var (muSpd, sdSpd) = MeanStdInInterval(tSpd, spd, start, end);
                var S = muSpd + 0.5f * sdSpd;

                var (tO, oSig) = openLookup[pid];
                var (O, _) = MeanStdInInterval(tO, oSig, start, end);
                var F = OscillationRate(tO, oSig, start, end);

                var combined = (float)(S * (0.85 + 0.15 * O) + 0.02 * Math.Min(F, 2.0));
                perP[pid] = combined;
            }

            var sorted = perP.OrderByDescending(kv => kv.Value).ToList();
            var best = sorted[0];
            var top2 = sorted.Count > 1 ? sorted[1] : (KeyValuePair<string, float>?)null;
            var top3 = sorted.Count > 2 ? sorted[2] : (KeyValuePair<string, float>?)null;

            var bestOverTop2 = top2.HasValue ? best.Value / (top2.Value.Value + 1e-9f) : (float?)null;
            var ambiguous = best.Value < opt.BestMin || (top2.HasValue && best.Value < opt.MarginRatio * top2.Value.Value);

            var vals = partIds.Select(pid => perP[pid]).ToArray();
            var probs = Softmax(vals, opt.SoftmaxTemp);
            for (var j = 0; j < partIds.Count; j++)
            {
                globalScore[si, j] += (float)(dur * Math.Log(probs[j] + 1e-12));
            }

            var row = new Dictionary<string, string>
            {
                ["start"] = start.ToString(CultureInfo.InvariantCulture),
                ["end"] = end.ToString(CultureInfo.InvariantCulture),
                ["dur"] = dur.ToString(CultureInfo.InvariantCulture),
                ["speaker"] = seg.speaker,
                ["best_participant"] = best.Key,
                ["best_combined"] = best.Value.ToString(CultureInfo.InvariantCulture),
                ["top2_participant"] = top2?.Key ?? "",
                ["top2_combined"] = top2?.Value.ToString(CultureInfo.InvariantCulture) ?? "",
                ["top3_participant"] = top3?.Key ?? "",
                ["top3_combined"] = top3?.Value.ToString(CultureInfo.InvariantCulture) ?? "",
                ["ambiguous"] = ambiguous.ToString(),
                ["best_over_top2"] = bestOverTop2?.ToString() ?? "",
            };
            segmentDebugRows.Add(row);
            segKeyToDebug[$"{start:F3}|{end:F3}|{seg.speaker}"] = row;
        }

        WriteCsv(Path.Combine(opt.OutDir, "segment_vvad_debug.csv"), segmentDebugRows);
        WriteScoreMatrix(Path.Combine(opt.OutDir, "score_matrix.csv"), globalScore, speakers, partIds);

        // global mapping
        var globalMap = new Dictionary<string, string>();
        for (var si = 0; si < speakers.Count; si++)
        {
            var bestJ = 0;
            var bestVal = globalScore[si, 0];
            for (var j = 1; j < partIds.Count; j++)
            {
                if (globalScore[si, j] > bestVal)
                {
                    bestVal = globalScore[si, j];
                    bestJ = j;
                }
            }
            globalMap[speakers[si]] = partIds[bestJ];
        }
        File.WriteAllText(Path.Combine(opt.OutDir, "speaker_to_participant.json"), JsonSerializer.Serialize(globalMap, new JsonSerializerOptions { WriteIndented = true }));

        // final alignment
        string? prevPid = null;
        var aligned = new List<Dictionary<string, object?>>();
        foreach (var seg in segments)
        {
            var start = seg.start + opt.AvOffsetSec;
            var end = seg.end + opt.AvOffsetSec;
            var dur = end - start;
            if (dur < opt.MinSegSec) continue;

            if (!segKeyToDebug.TryGetValue($"{start:F3}|{end:F3}|{seg.speaker}", out var row))
            {
                aligned.Add(new Dictionary<string, object?>
                {
                    ["start"] = start,
                    ["end"] = end,
                    ["speaker"] = seg.speaker,
                    ["participant"] = prevPid,
                    ["ambiguous"] = true,
                    ["reason"] = "missing_debug_row"
                });
                continue;
            }

            var bestPid = row["best_participant"];
            var bestVal = ParseFloat(row["best_combined"]);
            var top2Pid = row["top2_participant"];
            var top2Val = ParseFloat(row["top2_combined"]);
            var ambiguous = bool.TryParse(row["ambiguous"], out var amb) && amb;
            var bestOverTop2 = ParseFloat(row["best_over_top2"]);

            string chosen;
            if (opt.UseGlobalMap)
            {
                var superAmb = ambiguous && bestOverTop2 > 0 && bestOverTop2 <= opt.AmbiguousLocalMargin;
                if (superAmb || (ambiguous && dur > opt.AmbiguousGlobalMaxSec))
                {
                    chosen = bestPid;
                }
                else
                {
                    chosen = globalMap.TryGetValue(seg.speaker, out var gp) ? gp : bestPid;
                    var overrideRatio = 1.6;
                    if (top2Val > 0 && bestVal >= overrideRatio * top2Val)
                    {
                        chosen = bestPid;
                    }
                }
            }
            else
            {
                chosen = bestPid;
            }

            if (ambiguous && prevPid is not null)
            {
                var prevVal = 0f;
                if (prevPid == bestPid) prevVal = bestVal;
                else if (prevPid == top2Pid && top2Val > 0) prevVal = top2Val;
                if (prevVal >= opt.StickRatio * bestVal)
                {
                    chosen = prevPid;
                }
            }

            prevPid = chosen;
            aligned.Add(new Dictionary<string, object?>
            {
                ["start"] = start,
                ["end"] = end,
                ["speaker"] = seg.speaker,
                ["participant"] = chosen,
                ["ambiguous"] = ambiguous,
                ["best_pid"] = bestPid,
                ["best_val"] = bestVal,
                ["top2_pid"] = top2Pid,
                ["top2_val"] = top2Val > 0 ? top2Val : null,
                ["global_pid"] = globalMap.TryGetValue(seg.speaker, out var gp2) ? gp2 : null,
            });
        }

        var merged = MergeAligned(aligned, opt.MergeGapSec);
        File.WriteAllText(Path.Combine(opt.OutDir, "aligned_segments.json"), JsonSerializer.Serialize(merged, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine("Saved:");
        Console.WriteLine(" - participant_debug.csv");
        Console.WriteLine(" - segment_vvad_debug.csv");
        Console.WriteLine(" - score_matrix.csv  (GLOBAL speaker x participant)");
        Console.WriteLine(" - speaker_to_participant.json  (GLOBAL mapping)");
        Console.WriteLine(" - aligned_segments.json");
    }

    private sealed class ParticipantData
    {
        public float[] Time { get; init; } = Array.Empty<float>();
        public float[] MouthScoreRaw { get; init; } = Array.Empty<float>();
        public float[] MouthOpen { get; init; } = Array.Empty<float>();
    }

    private static List<Dictionary<string, object?>> MergeAligned(
        List<Dictionary<string, object?>> aligned,
        double gapSec)
    {
        if (aligned.Count == 0) return aligned;
        var ordered = aligned
            .OrderBy(a => Convert.ToDouble(a["start"] ?? 0.0, CultureInfo.InvariantCulture))
            .ToList();
        var merged = new List<Dictionary<string, object?>>();

        Dictionary<string, object?> cur = ordered[0];
        for (var i = 1; i < ordered.Count; i++)
        {
            var nxt = ordered[i];
            var curPid = cur.TryGetValue("participant", out var cp) ? cp?.ToString() : null;
            var nxtPid = nxt.TryGetValue("participant", out var np) ? np?.ToString() : null;
            var curEnd = Convert.ToDouble(cur["end"] ?? 0.0, CultureInfo.InvariantCulture);
            var nxtStart = Convert.ToDouble(nxt["start"] ?? 0.0, CultureInfo.InvariantCulture);
            var nxtEnd = Convert.ToDouble(nxt["end"] ?? 0.0, CultureInfo.InvariantCulture);

            if (!string.IsNullOrWhiteSpace(curPid) &&
                curPid == nxtPid &&
                nxtStart <= curEnd + gapSec + 1e-6)
            {
                cur["end"] = Math.Max(curEnd, nxtEnd);
                // Keep ambiguous if any segment was ambiguous
                var curAmb = cur.TryGetValue("ambiguous", out var ca) && ca is bool cb && cb;
                var nxtAmb = nxt.TryGetValue("ambiguous", out var na) && na is bool nb && nb;
                cur["ambiguous"] = curAmb || nxtAmb;
                continue;
            }
            merged.Add(cur);
            cur = nxt;
        }
        merged.Add(cur);
        return merged;
    }


    private static Dictionary<string, ParticipantData> LoadParticipants(string dir)
    {
        var files = Directory.GetFiles(dir, "participant_*.csv").OrderBy(f => f).ToList();
        if (files.Count == 0) throw new FileNotFoundException($"No participant_*.csv found in {dir}");
        var map = new Dictionary<string, ParticipantData>();
        foreach (var f in files)
        {
            var rows = ReadCsv(f);
            if (!rows.ContainsKey("time_sec") || !rows.ContainsKey("mouth_score_raw") || !rows.ContainsKey("mouth_open"))
            {
                throw new InvalidDataException($"{Path.GetFileName(f)} missing required columns");
            }
            map[Path.GetFileNameWithoutExtension(f)] = new ParticipantData
            {
                Time = rows["time_sec"],
                MouthScoreRaw = rows["mouth_score_raw"],
                MouthOpen = rows["mouth_open"]
            };
        }
        return map;
    }

    private static Dictionary<string, float[]> ReadCsv(string path)
    {
        var lines = File.ReadAllLines(path);
        if (lines.Length < 2) return new Dictionary<string, float[]>();
        var header = lines[0].Split(',').Select(h => h.Trim()).ToArray();
        var cols = header.ToDictionary(h => h, h => new List<float>());
        for (var i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            for (var j = 0; j < header.Length && j < parts.Length; j++)
            {
                if (float.TryParse(parts[j], NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ||
                    float.TryParse(parts[j], NumberStyles.Float, CultureInfo.CurrentCulture, out v))
                {
                    cols[header[j]].Add(v);
                }
                else
                {
                    cols[header[j]].Add(0f);
                }
            }
        }
        return cols.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());
    }

    private static float[] SmoothSeries(float[] x, int window)
    {
        if (window <= 1) return x.ToArray();
        var w = 1.0f / window;
        var y = new float[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            float sum = 0;
            var count = 0;
            for (var j = i - window / 2; j <= i + window / 2; j++)
            {
                if (j < 0 || j >= x.Length) continue;
                sum += x[j];
                count++;
            }
            y[i] = count > 0 ? sum / count : x[i];
        }
        return y;
    }

    private static float[] MouthSpeed(float[] mouth, float dt)
    {
        var d = new float[mouth.Length];
        for (var i = 0; i < mouth.Length; i++)
        {
            var prev = i == 0 ? mouth[0] : mouth[i - 1];
            d[i] = Math.Abs(mouth[i] - prev) / Math.Max(dt, 1e-6f);
        }
        return d;
    }

    private static (float mean, float std) MeanStdInInterval(float[] t, float[] x, double start, double end)
    {
        if (end <= start) return (0, 0);
        var i0 = LowerBound(t, (float)start);
        var i1 = UpperBound(t, (float)end);
        if (i1 - i0 < 3) return (0, 0);
        var sum = 0f;
        for (var i = i0; i < i1; i++) sum += x[i];
        var mean = sum / (i1 - i0);
        var varSum = 0f;
        for (var i = i0; i < i1; i++) varSum += (x[i] - mean) * (x[i] - mean);
        var std = (float)Math.Sqrt(varSum / (i1 - i0));
        return (mean, std);
    }

    private static float OscillationRate(float[] t, float[] o, double start, double end)
    {
        if (end <= start) return 0f;
        var i0 = LowerBound(t, (float)start);
        var i1 = UpperBound(t, (float)end);
        if (i1 - i0 <= 2) return 0f;
        var flips = 0f;
        for (var i = i0 + 1; i < i1; i++)
        {
            flips += Math.Abs(o[i] - o[i - 1]);
        }
        var dur = end - start;
        return (float)(flips / Math.Max(dur, 1e-6));
    }

    private static float[] Softmax(float[] vals, double temp)
    {
        var v = vals.Select(x => (double)x).ToArray();
        var max = v.Max();
        var denom = 0.0;
        var outv = new float[v.Length];
        for (var i = 0; i < v.Length; i++)
        {
            var e = Math.Exp((v[i] - max) / Math.Max(temp, 1e-6));
            denom += e;
            outv[i] = (float)e;
        }
        denom = denom + 1e-12;
        for (var i = 0; i < outv.Length; i++) outv[i] = (float)(outv[i] / denom);
        return outv;
    }

    private static float MedianDiff(float[] t)
    {
        if (t.Length < 2) return 0.04f;
        var diffs = new float[t.Length - 1];
        for (var i = 1; i < t.Length; i++) diffs[i - 1] = t[i] - t[i - 1];
        Array.Sort(diffs);
        return diffs[diffs.Length / 2];
    }

    private static float Percentile(float[] x, double p)
    {
        if (x.Length == 0) return 0f;
        var copy = x.ToArray();
        Array.Sort(copy);
        var idx = (int)Math.Round((p / 100.0) * (copy.Length - 1));
        idx = Math.Clamp(idx, 0, copy.Length - 1);
        return copy[idx];
    }

    private static float Mean(float[] x)
    {
        if (x.Length == 0) return 0f;
        float sum = 0;
        for (var i = 0; i < x.Length; i++) sum += x[i];
        return sum / x.Length;
    }

    private static int LowerBound(float[] t, float v)
    {
        var l = 0;
        var r = t.Length;
        while (l < r)
        {
            var m = (l + r) / 2;
            if (t[m] < v) l = m + 1; else r = m;
        }
        return l;
    }

    private static int UpperBound(float[] t, float v)
    {
        var l = 0;
        var r = t.Length;
        while (l < r)
        {
            var m = (l + r) / 2;
            if (t[m] <= v) l = m + 1; else r = m;
        }
        return l;
    }

    private static float ParseFloat(string s)
    {
        if (float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) return v;
        if (float.TryParse(s, NumberStyles.Float, CultureInfo.CurrentCulture, out v)) return v;
        return 0f;
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

    private static void WriteScoreMatrix(string path, float[,] mat, List<string> speakers, List<string> parts)
    {
        var sb = new StringBuilder();
        sb.Append("speaker");
        foreach (var p in parts) sb.Append($",{p}");
        sb.AppendLine();
        for (var i = 0; i < speakers.Count; i++)
        {
            sb.Append(speakers[i]);
            for (var j = 0; j < parts.Count; j++)
            {
                sb.Append(",");
                sb.Append(mat[i, j].ToString(CultureInfo.InvariantCulture));
            }
            sb.AppendLine();
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
}
