using Microsoft.ML.OnnxRuntime;
using AudioDiarizationCS.Preprocessing;
using AudioDiarizationCS.Diarization;
using System.Linq;
using System.Globalization;
using AudioDiarizationCS.IO;
using AudioDiarizationCS.Clustering;
using AudioDiarizationCS.Alignment;
using MathNet.Numerics.LinearAlgebra;

static string? GetArg(string[] args, string name)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name)
        {
            return args[i + 1];
        }
    }
    return null;
}

var vadPath = GetArg(args, "--vad");
var spkPath = GetArg(args, "--spk");
var audioPath = GetArg(args, "--audio");
var outPath = GetArg(args, "--out") ?? "segments.json";
var runDir = GetArg(args, "--run-dir"); // deprecated: avoid writing to Python repo
var batchDir = GetArg(args, "--batch-dir");
var localOutDir = GetArg(args, "--local-out");
var vadOnsetArg = GetArg(args, "--vad-onset");
var vadOffsetArg = GetArg(args, "--vad-offset");
var vadMinSilenceArg = GetArg(args, "--vad-min-silence");
var vadMinSpeechArg = GetArg(args, "--vad-min-speech");
var vadPadOnsetArg = GetArg(args, "--vad-pad-onset");
var vadPadOffsetArg = GetArg(args, "--vad-pad-offset");
var vadSmoothingArg = GetArg(args, "--vad-smoothing");
var vadOverlapArg = GetArg(args, "--vad-overlap");
var embWinArg = GetArg(args, "--emb-win");
var embShiftArg = GetArg(args, "--emb-shift");
var embWinListArg = GetArg(args, "--emb-win-list");
var embShiftListArg = GetArg(args, "--emb-shift-list");
var embWeightListArg = GetArg(args, "--emb-weights");
var subsegWinArg = GetArg(args, "--subseg-win");
var subsegShiftArg = GetArg(args, "--subseg-shift");
var mergeGapArg = GetArg(args, "--merge-gap");
var minSegArg = GetArg(args, "--min-seg");
var maxSpeakersArg = GetArg(args, "--max-speakers");
var alignSegmentsArg = GetArg(args, "--segments");
var alignParticipantsArg = GetArg(args, "--participants-dir");
var alignOutArg = GetArg(args, "--out");
var alignMinSegArg = GetArg(args, "--align-min-seg");
var alignAvOffsetArg = GetArg(args, "--align-av-offset");
var alignSmoothMsArg = GetArg(args, "--align-smooth-ms");
var alignMarginRatioArg = GetArg(args, "--align-margin-ratio");
var alignBestMinArg = GetArg(args, "--align-best-min");
var alignStickRatioArg = GetArg(args, "--align-stick-ratio");
var alignUseGlobalMap = args.Contains("--align-use-global-map");
var alignAmbGlobalMaxSecArg = GetArg(args, "--align-amb-global-max-sec");
var alignAmbGlobalMaxRatioArg = GetArg(args, "--align-amb-global-max-ratio");
var alignAmbLocalMarginArg = GetArg(args, "--align-amb-local-margin");
var alignSoftmaxTempArg = GetArg(args, "--align-softmax-temp");
var alignMergeGapArg = GetArg(args, "--align-merge-gap");
var alignMaxParticipantsArg = GetArg(args, "--align-max-participants");
var alignMinSpdMeanArg = GetArg(args, "--align-min-spd-mean");
var frXlsxArg = GetArg(args, "--facereader-xlsx");
var frSheetArg = GetArg(args, "--facereader-sheet");
var frOutArg = GetArg(args, "--facereader-out");
var frQualityArg = GetArg(args, "--facereader-quality-threshold");
var frKeepAll = args.Contains("--facereader-keep-all");
var frMinBlockRowsArg = GetArg(args, "--facereader-min-block-rows");
var clusterThreshArg = GetArg(args, "--cluster-threshold");
var clusterSearch = GetArg(args, "--cluster-search");
var clusterThreshold = 0.3f;
if (!string.IsNullOrWhiteSpace(clusterThreshArg))
{
    if (!float.TryParse(clusterThreshArg, NumberStyles.Float, CultureInfo.InvariantCulture, out clusterThreshold))
    {
        // Fallback to current culture (comma decimal)
        if (!float.TryParse(clusterThreshArg, NumberStyles.Float, CultureInfo.CurrentCulture, out clusterThreshold))
    {
        clusterThreshold = 0.3f;
    }
}
}

double ParseDouble(string? v, double fallback)
{
    if (string.IsNullOrWhiteSpace(v)) return fallback;
    if (double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) return d;
    if (double.TryParse(v, NumberStyles.Float, CultureInfo.CurrentCulture, out d)) return d;
    return fallback;
}

List<double> ParseList(string? v, List<double> fallback)
{
    if (string.IsNullOrWhiteSpace(v)) return new List<double>(fallback);
    var sep = v.Contains(';') ? ';' : ',';
    var parts = v.Split(new[] { sep, ' ' }, StringSplitOptions.RemoveEmptyEntries);
    var list = new List<double>();
    foreach (var p in parts)
    {
        if (double.TryParse(p, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) list.Add(d);
        else if (double.TryParse(p, NumberStyles.Float, CultureInfo.CurrentCulture, out d)) list.Add(d);
    }
    return list.Count > 0 ? list : new List<double>(fallback);
}

var vadOnset = (float)ParseDouble(vadOnsetArg, 0.9);
var vadOffset = (float)ParseDouble(vadOffsetArg, 0.5);
var vadMinSilence = ParseDouble(vadMinSilenceArg, 0.6);
var vadMinSpeech = ParseDouble(vadMinSpeechArg, 0.0);
var vadPadOnset = ParseDouble(vadPadOnsetArg, 0.0);
var vadPadOffset = ParseDouble(vadPadOffsetArg, 0.0);
var vadOverlap = ParseDouble(vadOverlapArg, 0.5);
var vadSmoothing = string.IsNullOrWhiteSpace(vadSmoothingArg) ? "none" : vadSmoothingArg.Trim().ToLowerInvariant();
var defaultWins = new List<double> { 3.0, 2.5, 2.0, 1.5, 1.0, 0.5 };
var defaultShifts = new List<double> { 1.5, 1.25, 1.0, 0.75, 0.5, 0.25 };
var embWinList = ParseList(embWinListArg ?? embWinArg, defaultWins);
var embShiftList = ParseList(embShiftListArg ?? embShiftArg, defaultShifts);
var embWeightList = ParseList(embWeightListArg, Enumerable.Repeat(1.0, embWinList.Count).ToList());
var subsegWinSec = ParseDouble(subsegWinArg, 1.5);
var subsegShiftSec = ParseDouble(subsegShiftArg, 0.75);
var mergeGapSec = ParseDouble(mergeGapArg, 0.0);
var minSegSec = ParseDouble(minSegArg, 0.0);
var maxSpeakers = (int)Math.Round(ParseDouble(maxSpeakersArg, 8));

if (!string.IsNullOrWhiteSpace(frXlsxArg) && !string.IsNullOrWhiteSpace(frOutArg))
{
    var opt = new FaceReaderExtractOptions
    {
        XlsxPath = frXlsxArg!,
        Sheet = frSheetArg,
        OutDir = frOutArg!,
        QualityThreshold = ParseDouble(frQualityArg, 0.7),
        KeepAll = frKeepAll,
        MinBlockRows = (int)Math.Round(ParseDouble(frMinBlockRowsArg, 200)),
    };
    FaceReaderExtractor.Run(opt);
    return;
}

// Alignment mode
if (!string.IsNullOrWhiteSpace(alignSegmentsArg) &&
    !string.IsNullOrWhiteSpace(alignParticipantsArg) &&
    !string.IsNullOrWhiteSpace(alignOutArg))
{
    var opt = new AlignmentOptions
    {
        SegmentsPath = alignSegmentsArg!,
        ParticipantsDir = alignParticipantsArg!,
        OutDir = alignOutArg!,
        MinSegSec = ParseDouble(alignMinSegArg, 0.4),
        AvOffsetSec = ParseDouble(alignAvOffsetArg, 0.0),
        SmoothMs = ParseDouble(alignSmoothMsArg, 240.0),
        MarginRatio = ParseDouble(alignMarginRatioArg, 1.12),
        BestMin = ParseDouble(alignBestMinArg, 0.06),
        StickRatio = ParseDouble(alignStickRatioArg, 0.92),
        UseGlobalMap = alignUseGlobalMap,
        AmbiguousGlobalMaxSec = ParseDouble(alignAmbGlobalMaxSecArg, -1.0),
        AmbiguousGlobalMaxRatio = ParseDouble(alignAmbGlobalMaxRatioArg, 0.05),
        AmbiguousLocalMargin = ParseDouble(alignAmbLocalMarginArg, 1.01),
        SoftmaxTemp = ParseDouble(alignSoftmaxTempArg, 0.08),
        MergeGapSec = ParseDouble(alignMergeGapArg, 0.0),
        MaxParticipants = (int)Math.Round(ParseDouble(alignMaxParticipantsArg, 0.0)),
        MinSpdMean = ParseDouble(alignMinSpdMeanArg, 0.0),
    };
    FaceSpeakerAlignment.Run(opt);
    return;
}

if (embShiftList.Count != embWinList.Count)
{
    embShiftList = embShiftList.Count == 1
        ? Enumerable.Repeat(embShiftList[0], embWinList.Count).ToList()
        : defaultShifts;
}
if (embWeightList.Count != embWinList.Count)
{
    embWeightList = Enumerable.Repeat(1.0, embWinList.Count).ToList();
}

if (string.IsNullOrWhiteSpace(vadPath) || string.IsNullOrWhiteSpace(spkPath))
{
    Console.WriteLine("Usage: dotnet run -- --vad <vad.onnx> --spk <speaker.onnx>");
    Console.WriteLine("Optional: --audio <file> to print mel shape");
    Console.WriteLine("Example:");
    Console.WriteLine("  dotnet run -- --vad /path/vad_multilingual_marblenet.onnx --spk /path/titanet_large.onnx --audio /path/file.wav");
    return;
}

Console.WriteLine("Loading ONNX models...");
using var vadSession = new InferenceSession(vadPath);
using var spkSession = new InferenceSession(spkPath);

Console.WriteLine("VAD inputs:");
foreach (var i in vadSession.InputMetadata)
{
    Console.WriteLine($"  {i.Key}: {string.Join(",", i.Value.Dimensions)} {i.Value.ElementType}");
}

Console.WriteLine("Speaker inputs:");
foreach (var i in spkSession.InputMetadata)
{
    Console.WriteLine($"  {i.Key}: {string.Join(",", i.Value.Dimensions)} {i.Value.ElementType}");
}

Console.WriteLine("ONNX models loaded successfully.");

if (!string.IsNullOrWhiteSpace(batchDir))
{
    var items = new List<(string wav, string run)>();
    items.Add((Path.Combine(batchDir, "testvideo_1_6min.wav"), "csharp_test_01"));
    items.Add((Path.Combine(batchDir, "testvideo_2_6min.wav"), "csharp_test_02"));
    items.Add((Path.Combine(batchDir, "test_3_6_min.wav"), "csharp_test_03"));
    items.Add((Path.Combine(batchDir, "test_4_6_min.wav"), "csharp_test_04"));
    items.Add((Path.Combine(batchDir, "test_5_6_min.wav"), "csharp_test_05"));

    foreach (var (wav, outRun) in items)
    {
        if (!File.Exists(wav))
        {
            Console.WriteLine($"Missing file: {wav}");
            continue;
        }
        Console.WriteLine($"\n=== Processing {Path.GetFileName(wav)} ===");
        var target = !string.IsNullOrWhiteSpace(localOutDir) ? localOutDir : ".";
        ProcessOne(vadPath!, spkPath!, wav, target, clusterThreshold, vadOnset, vadOffset, vadMinSpeech, vadMinSilence, vadPadOnset, vadPadOffset, vadSmoothing, vadOverlap, subsegWinSec, subsegShiftSec, mergeGapSec, minSegSec, maxSpeakers, embWinList, embShiftList, embWeightList, localOutDir, outRun, clusterSearch);
    }
    return;
}

if (!string.IsNullOrWhiteSpace(audioPath))
{
    ProcessOne(vadPath!, spkPath!, audioPath, outPath, clusterThreshold, vadOnset, vadOffset, vadMinSpeech, vadMinSilence, vadPadOnset, vadPadOffset, vadSmoothing, vadOverlap, subsegWinSec, subsegShiftSec, mergeGapSec, minSegSec, maxSpeakers, embWinList, embShiftList, embWeightList, localOutDir, null, clusterSearch);
}

static void ProcessOne(
    string vadPath,
    string spkPath,
    string audioPath,
    string outPath,
    float clusterThreshold,
    float vadOnset,
    float vadOffset,
    double vadMinSpeech,
    double vadMinSilence,
    double vadPadOnset,
    double vadPadOffset,
    string vadSmoothing,
    double vadOverlap,
    double subsegWinSec,
    double subsegShiftSec,
    double mergeGapSec,
    double minSegSec,
    int maxSpeakers,
    List<double> embWinSecList,
    List<double> embShiftSecList,
    List<double> embWeightList,
    string? localOutDir,
    string? nameTag,
    string? clusterSearch)
{
    var audio = AudioLoader.LoadMono(audioPath, 16000);
    var melVad = MelExtractor.ExtractVad(audio.Samples, audio.SampleRate);
    var melSpk = MelExtractor.ExtractSpk(audio.Samples, audio.SampleRate);
    Console.WriteLine($"Mel(VAD) shape: {melVad.Count} frames x {melVad[0].Length} bins");
    Console.WriteLine($"Mel(SPK) shape: {melSpk.Count} frames x {melSpk[0].Length} bins");

    using var vad = new VadOnnx(vadPath);
    var speechProb = vad.RunSliding(
        melVad.ToArray(),
        windowSec: 0.63,
        shiftSec: 0.01,
        smoothingMethod: vadSmoothing,
        overlap: vadOverlap);
    Console.WriteLine($"SpeechProb: min={speechProb.Min():F4} max={speechProb.Max():F4} mean={speechProb.Average():F4}");
    var vadSegments = VadSegmenter.BuildSegments(
        speechProb,
        onset: vadOnset,
        offset: vadOffset,
        padOnsetSec: vadPadOnset,
        padOffsetSec: vadPadOffset,
        minSpeechSec: vadMinSpeech,
        minSilenceSec: vadMinSilence);

    Console.WriteLine($"VAD segments: {vadSegments.Count}");
    foreach (var s in vadSegments.Take(5))
    {
        Console.WriteLine($"  {s.Start:F2} - {s.End:F2}");
    }
    var lastEnd = vadSegments.Count > 0 ? vadSegments.Max(s => s.End) : 0.0;
    Console.WriteLine($"Audio duration: {audio.Samples.Length / (double)audio.SampleRate:F2}s | Last VAD end: {lastEnd:F2}s");

    var segments = Subsegment(vadSegments, subsegWinSec, subsegShiftSec);
    Console.WriteLine($"Subsegments: {segments.Count} (win={subsegWinSec:F2}s shift={subsegShiftSec:F2}s)");

    // Speaker embeddings + clustering
    using var spk = new SpeakerEmbeddingOnnx(spkPath);
    var embList = new List<float[]>();
    const double hopSec = 0.010;

    foreach (var seg in segments)
    {
        var startFrame = (int)Math.Floor(seg.Start / hopSec);
        var endFrame = (int)Math.Ceiling(seg.End / hopSec);
        startFrame = Math.Max(0, startFrame);
        endFrame = Math.Min(melSpk.Count, endFrame);
        if (endFrame <= startFrame) continue;

        var segFrames = melSpk.GetRange(startFrame, endFrame - startFrame).ToArray();
        var scaleEmbeds = new List<(float[] vec, double weight)>();
        for (var s = 0; s < embWinSecList.Count; s++)
        {
            var winFrames = (int)Math.Round(embWinSecList[s] / hopSec);
            var shiftFrames = (int)Math.Round(embShiftSecList[s] / hopSec);
            winFrames = Math.Max(1, winFrames);
            shiftFrames = Math.Max(1, shiftFrames);

            var segEmbeds = new List<float[]>();
            if (segFrames.Length < winFrames)
            {
                var padded = new float[winFrames][];
                Array.Copy(segFrames, padded, segFrames.Length);
                var last = segFrames[^1];
                for (var i = segFrames.Length; i < winFrames; i++) padded[i] = last;
                segEmbeds.Add(spk.Run(padded));
            }
            else
            {
                for (var i = 0; i + winFrames <= segFrames.Length; i += shiftFrames)
                {
                    var window = new float[winFrames][];
                    Array.Copy(segFrames, i, window, 0, winFrames);
                    segEmbeds.Add(spk.Run(window));
                }
            }

            if (segEmbeds.Count == 0) continue;
            var avg = new float[segEmbeds[0].Length];
            foreach (var e in segEmbeds)
            {
                for (var k = 0; k < avg.Length; k++) avg[k] += e[k];
            }
            for (var k = 0; k < avg.Length; k++) avg[k] /= segEmbeds.Count;
            scaleEmbeds.Add((avg, embWeightList[s]));
        }

        if (scaleEmbeds.Count == 0) continue;
        var dim = scaleEmbeds[0].vec.Length;
        var combined = new float[dim];
        double wsum = 0.0;
        foreach (var (vec, w) in scaleEmbeds)
        {
            for (var k = 0; k < dim; k++) combined[k] += (float)(vec[k] * w);
            wsum += w;
        }
        if (wsum > 0)
        {
            for (var k = 0; k < dim; k++) combined[k] = (float)(combined[k] / wsum);
        }
        embList.Add(combined);
    }

    var embeddings = embList.ToArray();
    int[] labels;
    if (!string.IsNullOrWhiteSpace(clusterSearch) && clusterSearch == "1")
    {
        var aff = BuildAffinity(embeddings);
        var nmesc = new Nmesc(aff, maxNumSpeakers: maxSpeakers, maxRpThreshold: 0.25, sparseSearchVolume: 30);
        var (pValue, nSpk) = nmesc.Estimate();
        var graph = BuildAffinityGraph(aff, pValue);
        labels = SpectralClustering.Cluster(graph, nSpk, nTrials: 10);
        Console.WriteLine($"Clusters: {labels.Distinct().Count()} (p={pValue}, nSpk={nSpk})");
    }
    else
    {
        labels = AgglomerativeClusterer.Cluster(embeddings, clusterThreshold);
        Console.WriteLine($"Clusters: {labels.Distinct().Count()} (threshold={clusterThreshold.ToString("F2", CultureInfo.InvariantCulture)})");
    }

    var diar = new List<(double start, double end, string speaker)>();
    for (var i = 0; i < segments.Count && i < labels.Length; i++)
    {
        diar.Add((segments[i].Start, segments[i].End, $"speaker_{labels[i]}"));
    }
    diar = MergeAdjacent(diar, mergeGapSec);
    if (minSegSec > 0)
    {
        diar = diar.Where(s => (s.end - s.start) >= minSegSec).ToList();
    }
    var finalOut = outPath;
    if (Directory.Exists(outPath) || outPath.EndsWith(Path.DirectorySeparatorChar))
    {
        Directory.CreateDirectory(outPath);
        var name = nameTag ?? Path.GetFileNameWithoutExtension(audioPath);
        finalOut = Path.Combine(outPath, $"{name}_segments.json");
    }
    SegmentsWriter.Write(finalOut, diar);
    Console.WriteLine($"Wrote segments: {finalOut}");

    if (!string.IsNullOrWhiteSpace(localOutDir))
    {
        Directory.CreateDirectory(localOutDir);
        var name = nameTag ?? Path.GetFileNameWithoutExtension(audioPath);
        var localPath = Path.Combine(localOutDir, $"{name}_segments.json");
        SegmentsWriter.Write(localPath, diar);
        Console.WriteLine($"Wrote local segments: {localPath}");
    }
}

static Matrix<double> BuildAffinity(float[][] embeddings)
{
    var n = embeddings.Length;
    if (n == 0) return Matrix<double>.Build.Dense(0, 0);
    if (n == 1) return Matrix<double>.Build.Dense(1, 1, 1.0);

    // L2 normalize embeddings
    var normed = new float[n][];
    for (var i = 0; i < n; i++)
    {
        var v = embeddings[i];
        double norm = 0;
        for (var k = 0; k < v.Length; k++) norm += v[k] * v[k];
        norm = Math.Sqrt(norm) + 1e-10;
        var nv = new float[v.Length];
        for (var k = 0; k < v.Length; k++) nv[k] = (float)(v[k] / norm);
        normed[i] = nv;
    }

    // Cosine similarity matrix
    var mat = Matrix<double>.Build.Dense(n, n);
    for (var i = 0; i < n; i++)
    {
        for (var j = 0; j < n; j++)
        {
            mat[i, j] = Dot(normed[i], normed[j]);
        }
    }
    for (var i = 0; i < n; i++) mat[i, i] = 1.0;

    // Min-max normalize to [0,1] like NeMo
    var min = mat.Enumerate().Min();
    var max = mat.Enumerate().Max();
    if (max - min < 1e-12) return Matrix<double>.Build.Dense(n, n, 1.0);
    var scaled = mat.Map(v => (v - min) / (max - min));
    return scaled;
}

static Matrix<double> BuildAffinityGraph(Matrix<double> mat, int p)
{
    return Nmesc.GetAffinityGraph(mat, p);
}

static double Dot(float[] a, float[] b)
{
    double dot = 0;
    for (var i = 0; i < a.Length; i++) dot += a[i] * b[i];
    return dot;
}

static List<Segment> Subsegment(List<Segment> vadSegments, double winSec, double shiftSec)
{
    var result = new List<Segment>();
    if (vadSegments.Count == 0) return result;

    foreach (var seg in vadSegments)
    {
        var dur = seg.End - seg.Start;
        if (dur <= 0) continue;

        if (dur <= winSec)
        {
            result.Add(seg);
            continue;
        }

        var t = seg.Start;
        while (t + winSec <= seg.End + 1e-6)
        {
            var end = t + winSec;
            if (end > seg.End) end = seg.End;
            result.Add(new Segment(t, end));
            t += shiftSec;
        }

        // Ensure we cover the tail if we didn't land exactly
        if (result.Count > 0)
        {
            var last = result[^1];
            if (last.End < seg.End - 1e-6)
            {
                var start = Math.Max(seg.Start, seg.End - winSec);
                result.Add(new Segment(start, seg.End));
            }
        }
    }
    return result;
}

static List<(double start, double end, string speaker)> MergeAdjacent(
    List<(double start, double end, string speaker)> segments,
    double gapSec)
{
    if (segments.Count == 0) return segments;
    var ordered = segments.OrderBy(s => s.start).ToList();
    var merged = new List<(double start, double end, string speaker)>();
    var cur = ordered[0];
    for (var i = 1; i < ordered.Count; i++)
    {
        var nxt = ordered[i];
        if (nxt.speaker == cur.speaker && nxt.start <= cur.end + gapSec + 1e-6)
        {
            cur = (cur.start, Math.Max(cur.end, nxt.end), cur.speaker);
        }
        else
        {
            merged.Add(cur);
            cur = nxt;
        }
    }
    merged.Add(cur);
    return merged;
}
