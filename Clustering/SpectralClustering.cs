using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace AudioDiarizationCS.Clustering;

public static class SpectralClustering
{
    public static int[] Cluster(Matrix<double> affinity, int nClusters, int nTrials = 10, int seed = 0)
    {
        // Laplacian
        var lap = Laplacian(affinity);

        // Eigen decomposition
        var evd = lap.Evd();
        var eigVals = evd.EigenValues.Select((v, idx) => (val: v.Real, idx)).OrderBy(x => x.val).ToArray();
        var eigVecs = evd.EigenVectors;

        // take nClusters eigenvectors with smallest eigenvalues
        var embed = Matrix<double>.Build.Dense(eigVecs.RowCount, nClusters);
        for (var k = 0; k < nClusters; k++)
        {
            var colIdx = eigVals[k].idx;
            embed.SetColumn(k, eigVecs.Column(colIdx));
        }

        // k-means on spectral embeddings
        int[] bestLabels = KMeans.Cluster(embed, nClusters, seed: seed);
        return bestLabels;
    }

    private static Matrix<double> Laplacian(Matrix<double> affinity)
    {
        var n = affinity.RowCount;
        var d = Vector<double>.Build.Dense(n);
        for (var i = 0; i < n; i++) affinity[i, i] = 0.0;
        for (var i = 0; i < n; i++)
        {
            d[i] = affinity.Row(i).Sum();
        }
        var D = Matrix<double>.Build.DenseDiagonal(n, n, i => d[i]);
        return D - affinity;
    }
}
