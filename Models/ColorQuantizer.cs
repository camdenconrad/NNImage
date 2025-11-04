using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using NNImage.Services;

namespace NNImage.Models;

public class ColorQuantizer
{
    private readonly int _colorCount;
    private List<ColorRgb> _palette = new();
    private bool _isInitialized;
    private GpuAccelerator? _gpu;
    private readonly ConcurrentDictionary<ColorRgb, ColorRgb> _quantizeCache = new();

    public ColorQuantizer(int colorCount, GpuAccelerator? gpu = null)
    {
        _colorCount = colorCount;
        _gpu = gpu;
    }

    public void BuildPalette(List<ColorRgb> colors)
    {
        if (colors.Count <= _colorCount)
        {
            _palette = colors.Distinct().ToList();
            _isInitialized = true;
            return;
        }

        // K-means clustering for color quantization
        _palette = KMeansClustering(colors, _colorCount);
        _isInitialized = true;
    }

    public ColorRgb Quantize(ColorRgb color)
    {
        if (!_isInitialized || _palette.Count == 0)
            return color;

        // Check cache first for massive speedup
        if (_quantizeCache.TryGetValue(color, out var cached))
            return cached;

        // Find nearest color in palette
        var nearest = _palette[0];
        var minDistance = ColorDistance(color, nearest);

        for (int i = 1; i < _palette.Count; i++)
        {
            var distance = ColorDistance(color, _palette[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                nearest = _palette[i];
            }
        }

        // Cache the result
        _quantizeCache[color] = nearest;
        return nearest;
    }

    /// <summary>
    /// MASSIVE SPEEDUP: Quantize entire batch of colors on GPU at once!
    /// Process millions of colors/sec instead of one-by-one
    /// </summary>
    public ColorRgb[] QuantizeBatch(ColorRgb[] colors)
    {
        if (!_isInitialized || _palette.Count == 0)
            return colors;

        // Try GPU batch quantization first for MASSIVE speedup
        var gpuResult = _gpu?.QuantizeColorsBatchGpu(colors, _palette.ToArray());
        if (gpuResult != null)
        {
            // Cache all results for future single lookups
            System.Threading.Tasks.Parallel.For(0, colors.Length, i =>
            {
                _quantizeCache.TryAdd(colors[i], gpuResult[i]);
            });
            return gpuResult;
        }

        // CPU fallback - still faster than one-by-one
        var quantized = new ColorRgb[colors.Length];
        System.Threading.Tasks.Parallel.For(0, colors.Length, i =>
        {
            quantized[i] = Quantize(colors[i]);
        });
        return quantized;
    }

    public List<ColorRgb> GetPalette()
    {
        return _palette;
    }

    private List<ColorRgb> KMeansClustering(List<ColorRgb> colors, int k)
    {
        Console.WriteLine($"[ColorQuantizer] K-means clustering {colors.Count} colors into {k} clusters");
        var random = new Random(42);

        // Initialize centroids randomly
        var centroids = colors.OrderBy(_ => random.Next()).Take(k).ToList();
        var previousCentroids = new List<ColorRgb>();

        const int maxIterations = 20;
        int iteration = 0;

        while (iteration < maxIterations && !CentroidsEqual(centroids, previousCentroids))
        {
            previousCentroids = centroids.ToList();

            // Use GPU if available for assignment
            int[] assignments;
            if (_gpu?.IsAvailable == true)
            {
                Console.WriteLine($"[ColorQuantizer] Using GPU for K-means iteration {iteration + 1}");
                assignments = _gpu.AssignColorsToNearestCentroid(colors.ToArray(), centroids.ToArray());
            }
            else
            {
                // CPU fallback
                assignments = new int[colors.Count];
                System.Threading.Tasks.Parallel.For(0, colors.Count, i =>
                {
                    var color = colors[i];
                    var nearestIndex = 0;
                    var minDistance = ColorDistance(color, centroids[0]);

                    for (int c = 1; c < k; c++)
                    {
                        var distance = ColorDistance(color, centroids[c]);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            nearestIndex = c;
                        }
                    }

                    assignments[i] = nearestIndex;
                });
            }

            // Build clusters from assignments
            var clusters = new List<ColorRgb>[k];
            for (int i = 0; i < k; i++)
                clusters[i] = new List<ColorRgb>();

            for (int i = 0; i < colors.Count; i++)
            {
                clusters[assignments[i]].Add(colors[i]);
            }

            // Update centroids (parallel)
            System.Threading.Tasks.Parallel.For(0, k, i =>
            {
                if (clusters[i].Count > 0)
                {
                    var avgR = (byte)clusters[i].Average(c => c.R);
                    var avgG = (byte)clusters[i].Average(c => c.G);
                    var avgB = (byte)clusters[i].Average(c => c.B);
                    centroids[i] = new ColorRgb(avgR, avgG, avgB);
                }
            });

            iteration++;
        }

        Console.WriteLine($"[ColorQuantizer] K-means completed after {iteration} iterations");
        return centroids;
    }

    private double ColorDistance(ColorRgb c1, ColorRgb c2)
    {
        var dr = c1.R - c2.R;
        var dg = c1.G - c2.G;
        var db = c1.B - c2.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }

    private bool CentroidsEqual(List<ColorRgb> c1, List<ColorRgb> c2)
    {
        if (c1.Count != c2.Count)
            return false;

        for (int i = 0; i < c1.Count; i++)
        {
            if (!c1[i].Equals(c2[i]))
                return false;
        }

        return true;
    }
}
