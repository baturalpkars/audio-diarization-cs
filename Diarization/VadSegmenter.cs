namespace AudioDiarizationCS.Diarization;

public sealed record Segment(double Start, double End);

public static class VadSegmenter
{
    public static List<Segment> BuildSegments(
        float[] speechProb,
        double frameHopSec = 0.010,
        float onset = 0.9f,
        float offset = 0.5f,
        double padOnsetSec = 0.0,
        double padOffsetSec = 0.0,
        double minSpeechSec = 0.0,
        double minSilenceSec = 0.6)
    {
        var segments = new List<Segment>();
        if (speechProb.Length == 0) return segments;

        var inSpeech = false;
        var startIdx = 0;
        var lastSpeechIdx = 0;

        for (var i = 0; i < speechProb.Length; i++)
        {
            var p = speechProb[i];
            if (!inSpeech)
            {
                if (p >= onset)
                {
                    inSpeech = true;
                    startIdx = i;
                    lastSpeechIdx = i;
                }
            }
            else
            {
                if (p >= offset)
                {
                    lastSpeechIdx = i;
                }
                else
                {
                    var silenceDur = (i - lastSpeechIdx) * frameHopSec;
                    if (silenceDur >= minSilenceSec)
                    {
                        var endIdx = lastSpeechIdx;
                        AddIfLongEnough(segments, startIdx, endIdx, frameHopSec, padOnsetSec, padOffsetSec, minSpeechSec);
                        inSpeech = false;
                    }
                }
            }
        }

        if (inSpeech)
        {
            AddIfLongEnough(segments, startIdx, lastSpeechIdx, frameHopSec, padOnsetSec, padOffsetSec, minSpeechSec);
        }

        return segments;
    }

    private static void AddIfLongEnough(
        List<Segment> segments,
        int startIdx,
        int endIdx,
        double hop,
        double padOnsetSec,
        double padOffsetSec,
        double minSpeechSec)
    {
        var start = startIdx * hop - padOnsetSec;
        var end = (endIdx + 1) * hop + padOffsetSec;
        if (start < 0) start = 0;
        if (end - start >= minSpeechSec)
        {
            segments.Add(new Segment(start, end));
        }
    }
}
