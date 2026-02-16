using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;

namespace AudioDiarizationCS.Clustering;

public sealed class Nmesc
{
    private readonly Matrix<double> _affinity;
    private readonly int _maxNumSpeakers;
    private readonly double _maxRpThreshold;
    private readonly int _sparseSearchVolume;
    private readonly int _nmeMatSize;
    private readonly bool _useSubsampling;
    private readonly double _fixedThres;
    private readonly bool _majVoteSpkCount;
    private readonly double _eps = 1e-10;

    public Nmesc(
        Matrix<double> affinity,
        int maxNumSpeakers = 10,
        double maxRpThreshold = 0.25,
        int sparseSearchVolume = 30,
        int nmeMatSize = 512,
        bool useSubsamplingForNme = true,
        double fixedThres = -1.0,
        bool majVoteSpkCount = false)
    {
        _affinity = affinity;
        _maxNumSpeakers = maxNumSpeakers;
        _maxRpThreshold = maxRpThreshold;
        _sparseSearchVolume = sparseSearchVolume;
        _nmeMatSize = nmeMatSize;
        _useSubsampling = useSubsamplingForNme;
        _fixedThres = fixedThres;
        _majVoteSpkCount = majVoteSpkCount;
    }

    public (int pValue, int nSpeakers) Estimate()
    {
        var mat = _affinity.Clone();
        var n = mat.RowCount;
        if (n == 0) return (2, 1);

        var subsampleRatio = 1;
        if (_useSubsampling && n > _nmeMatSize)
        {
            subsampleRatio = Math.Max(1, (int)Math.Floor(n / (double)_nmeMatSize));
        }

        if (_useSubsampling && subsampleRatio > 1)
        {
            var idx = Enumerable.Range(0, n).Where(i => i % subsampleRatio == 0).ToArray();
            var sub = Matrix<double>.Build.Dense(idx.Length, idx.Length);
            for (var i = 0; i < idx.Length; i++)
            {
                for (var j = 0; j < idx.Length; j++)
                {
                    sub[i, j] = mat[idx[i], idx[j]];
                }
            }
            mat = sub;
        }

        n = mat.RowCount;
        var maxP = Math.Max(2, (int)Math.Floor(n * (_fixedThres > 0 ? _fixedThres : _maxRpThreshold)));
        var pList = GetPValueList(n, maxP, _fixedThres > 0);

        var eigRatioList = new List<double>();
        var estSpkList = new List<int>();

        foreach (var p in pList)
        {
            var (gP, estSpk) = GetEigRatio(mat, p);
            eigRatioList.Add(gP);
            estSpkList.Add(estSpk);
        }

        var minIdx = ArgMin(eigRatioList);
        var rpP = pList[minIdx];

        var graph = GetAffinityGraph(mat, rpP);
        if (!IsGraphFullyConnected(graph))
        {
            var (g2, p2) = GetMinimumConnection(mat, pList);
            graph = g2;
            rpP = p2;
        }

        var pHat = Math.Max(2, rpP * subsampleRatio);
        var estNum = _majVoteSpkCount ? Mode(estSpkList) : estSpkList[minIdx];
        return (pHat, estNum);
    }

    private List<int> GetPValueList(int n, int maxP, bool fixedThres)
    {
        if (fixedThres)
        {
            return new List<int> { Math.Max(2, maxP) };
        }
        if (maxP <= _sparseSearchVolume)
        {
            return Enumerable.Range(2, Math.Max(1, maxP - 1)).ToList();
        }
        // sparse search
        var steps = Math.Min(maxP, _sparseSearchVolume);
        var list = new List<int>();
        for (var i = 0; i < steps; i++)
        {
            var p = 1 + (int)Math.Round(i * (maxP - 1) / (double)Math.Max(1, steps - 1));
            if (p < 2) p = 2;
            if (!list.Contains(p)) list.Add(p);
        }
        return list;
    }

    private (double gP, int estSpk) GetEigRatio(Matrix<double> mat, int p)
    {
        var graph = GetAffinityGraph(mat, p);
        var (estSpk, lambdas, gaps) = EstimateNumSpeakers(graph, _maxNumSpeakers);
        var maxGap = gaps.Take(Math.Min(_maxNumSpeakers, gaps.Length)).DefaultIfEmpty(0.0).Max();
        var maxLambda = lambdas.DefaultIfEmpty(0.0).Max();
        var maxEigGap = maxGap / (maxLambda + _eps);
        var gP = (p / (double)mat.RowCount) / (maxEigGap + _eps);
        return (gP, estSpk);
    }

    private static (int estSpk, double[] lambdas, double[] gaps) EstimateNumSpeakers(Matrix<double> affinity, int maxNum)
    {
        var lap = SpectralClusteringHelper.Laplacian(affinity);
        var evd = lap.Evd();
        var eig = evd.EigenValues.Select(c => c.Real).OrderBy(v => v).ToArray();
        var gaps = Eigengaps(eig);
        var limit = Math.Min(maxNum, gaps.Length);
        var k = limit > 0 ? ArgMax(gaps.Take(limit).ToArray()) + 1 : 1;
        return (k, eig, gaps);
    }

    public static Matrix<double> GetAffinityGraph(Matrix<double> mat, int p)
    {
        var n = mat.RowCount;
        var graph = Matrix<double>.Build.Dense(n, n);
        for (var i = 0; i < n; i++)
        {
            var row = mat.Row(i);
            var idx = row.Select((v, j) => (v, j)).OrderByDescending(x => x.v).Take(p).Select(x => x.j);
            foreach (var j in idx)
            {
                graph[i, j] = 1.0;
            }
        }
        var sym = (graph + graph.Transpose()) * 0.5;
        return sym;
    }

    private static double[] Eigengaps(double[] eig)
    {
        var gaps = new double[eig.Length - 1];
        for (var i = 0; i < gaps.Length; i++)
        {
            gaps[i] = eig[i + 1] - eig[i];
        }
        return gaps;
    }

    private static int ArgMax(double[] arr)
    {
        var best = 0;
        var bestVal = double.NegativeInfinity;
        for (var i = 0; i < arr.Length; i++)
        {
            if (arr[i] > bestVal)
            {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    }

    private static int ArgMin(List<double> arr)
    {
        var best = 0;
        var bestVal = double.PositiveInfinity;
        for (var i = 0; i < arr.Count; i++)
        {
            if (arr[i] < bestVal)
            {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    }

    private static bool IsGraphFullyConnected(Matrix<double> graph)
    {
        var n = graph.RowCount;
        if (n == 0) return true;
        var visited = new bool[n];
        var stack = new Stack<int>();
        stack.Push(0);
        visited[0] = true;
        while (stack.Count > 0)
        {
            var v = stack.Pop();
            for (var j = 0; j < n; j++)
            {
                if (graph[v, j] > 0 && !visited[j])
                {
                    visited[j] = true;
                    stack.Push(j);
                }
            }
        }
        return visited.All(x => x);
    }

    private static (Matrix<double> graph, int p) GetMinimumConnection(Matrix<double> mat, List<int> pList)
    {
        var pValue = 2;
        var graph = GetAffinityGraph(mat, pValue);
        foreach (var p in pList)
        {
            var connected = IsGraphFullyConnected(graph);
            graph = GetAffinityGraph(mat, p);
            if (connected)
            {
                pValue = p;
                break;
            }
        }
        return (graph, pValue);
    }

    private static int Mode(List<int> values)
    {
        return values
            .GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
            .Select(g => g.Key)
            .FirstOrDefault();
    }
}

internal static class SpectralClusteringHelper
{
    public static Matrix<double> Laplacian(Matrix<double> affinity)
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
