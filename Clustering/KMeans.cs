using MathNet.Numerics.LinearAlgebra;

namespace AudioDiarizationCS.Clustering;

public static class KMeans
{
    public static int[] Cluster(Matrix<double> X, int k, int maxIter = 100, int seed = 0)
    {
        var rand = new Random(seed);
        var n = X.RowCount;
        var d = X.ColumnCount;

        var centroids = Matrix<double>.Build.Dense(k, d);
        var used = new HashSet<int>();
        for (var i = 0; i < k; i++)
        {
            int idx;
            do { idx = rand.Next(n); } while (!used.Add(idx));
            centroids.SetRow(i, X.Row(idx));
        }

        var labels = new int[n];
        for (var iter = 0; iter < maxIter; iter++)
        {
            var changed = false;

            for (var i = 0; i < n; i++)
            {
                var best = 0;
                var bestDist = double.PositiveInfinity;
                for (var c = 0; c < k; c++)
                {
                    var dist = (X.Row(i) - centroids.Row(c)).L2Norm();
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        best = c;
                    }
                }
                if (labels[i] != best)
                {
                    labels[i] = best;
                    changed = true;
                }
            }

            if (!changed) break;
            centroids.Clear();
            var counts = new int[k];
            for (var i = 0; i < n; i++)
            {
                centroids.SetRow(labels[i], centroids.Row(labels[i]) + X.Row(i));
                counts[labels[i]]++;
            }
            for (var c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    centroids.SetRow(c, centroids.Row(c) / counts[c]);
                }
            }
        }

        return labels;
    }
}
