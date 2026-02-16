namespace AudioDiarizationCS.Diarization;

public static class AgglomerativeClusterer
{
    public static int[] Cluster(float[][] embeddings, float mergeThreshold = 0.3f)
    {
        var n = embeddings.Length;
        if (n == 0) return Array.Empty<int>();
        if (n == 1) return new[] { 0 };

        // Start: each embedding is its own cluster
        var clusters = new List<List<int>>();
        for (var i = 0; i < n; i++) clusters.Add(new List<int> { i });

        while (true)
        {
            float bestDist = float.MaxValue;
            int bestA = -1, bestB = -1;

            for (var i = 0; i < clusters.Count; i++)
            {
                for (var j = i + 1; j < clusters.Count; j++)
                {
                    var dist = CosineDistance(
                        Centroid(embeddings, clusters[i]),
                        Centroid(embeddings, clusters[j])
                    );
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestA = i;
                        bestB = j;
                    }
                }
            }

            if (bestA == -1 || bestB == -1 || bestDist > mergeThreshold)
            {
                break;
            }

            // Merge B into A
            clusters[bestA].AddRange(clusters[bestB]);
            clusters.RemoveAt(bestB);
        }

        var labels = new int[n];
        for (var c = 0; c < clusters.Count; c++)
        {
            foreach (var idx in clusters[c])
            {
                labels[idx] = c;
            }
        }
        return labels;
    }

    public static (int[] labels, float threshold, double score) SearchBestThreshold(
        float[][] embeddings,
        float minThreshold = 0.10f,
        float maxThreshold = 0.50f,
        float step = 0.02f,
        double clusterCountPenalty = 0.05)
    {
        if (embeddings.Length <= 1)
        {
            return (Cluster(embeddings, maxThreshold), maxThreshold, 0.0);
        }

        int[]? bestLabels = null;
        float bestT = minThreshold;
        double bestScore = double.NegativeInfinity;

        for (var t = minThreshold; t <= maxThreshold + 1e-6; t += step)
        {
            var labels = Cluster(embeddings, t);
            var k = labels.Distinct().Count();
            var sil = Silhouette(embeddings, labels);
            var score = sil - clusterCountPenalty * k;

            if (score > bestScore)
            {
                bestScore = score;
                bestT = t;
                bestLabels = labels;
            }
        }

        return (bestLabels ?? Cluster(embeddings, minThreshold), bestT, bestScore);
    }

    private static float[] Centroid(float[][] embeddings, List<int> idxs)
    {
        var d = embeddings[0].Length;
        var sum = new float[d];
        foreach (var i in idxs)
        {
            var v = embeddings[i];
            for (var k = 0; k < d; k++) sum[k] += v[k];
        }
        var inv = 1.0f / idxs.Count;
        for (var k = 0; k < d; k++) sum[k] *= inv;
        return sum;
    }

    private static float CosineDistance(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        var denom = Math.Sqrt(na) * Math.Sqrt(nb);
        if (denom <= 0) return 1.0f;
        var cos = dot / denom;
        return (float)(1.0 - cos);
    }

    private static double Silhouette(float[][] embeddings, int[] labels)
    {
        var n = embeddings.Length;
        var k = labels.Distinct().Count();
        if (k <= 1) return -1.0;

        // Precompute pairwise distances
        var d = new double[n, n];
        for (var i = 0; i < n; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                var dist = CosineDistance(embeddings[i], embeddings[j]);
                d[i, j] = dist;
                d[j, i] = dist;
            }
        }

        double total = 0;
        for (var i = 0; i < n; i++)
        {
            var a = AvgIntraDist(i, labels, d);
            var b = AvgNearestClusterDist(i, labels, d);
            var s = (b - a) / Math.Max(a, b);
            total += s;
        }
        return total / n;
    }

    private static double AvgIntraDist(int idx, int[] labels, double[,] d)
    {
        var label = labels[idx];
        double sum = 0;
        int cnt = 0;
        for (var i = 0; i < labels.Length; i++)
        {
            if (i == idx) continue;
            if (labels[i] == label)
            {
                sum += d[idx, i];
                cnt++;
            }
        }
        return cnt == 0 ? 0 : sum / cnt;
    }

    private static double AvgNearestClusterDist(int idx, int[] labels, double[,] d)
    {
        var label = labels[idx];
        var clusters = labels.Distinct().ToArray();
        double best = double.PositiveInfinity;

        foreach (var c in clusters)
        {
            if (c == label) continue;
            double sum = 0;
            int cnt = 0;
            for (var i = 0; i < labels.Length; i++)
            {
                if (labels[i] == c)
                {
                    sum += d[idx, i];
                    cnt++;
                }
            }
            if (cnt > 0)
            {
                var avg = sum / cnt;
                if (avg < best) best = avg;
            }
        }

        return double.IsInfinity(best) ? 0 : best;
    }
}
