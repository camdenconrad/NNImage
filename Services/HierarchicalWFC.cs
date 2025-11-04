using System;
using System.Collections.Generic;
using System.Linq;
using Avalonia.Media.Imaging;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// Hierarchical Wave Function Collapse
/// Generates images progressively from coarse to fine resolution
/// </summary>
public class HierarchicalWFC
{
    private readonly MultiScaleContextGraph _multiScaleGraph;
    private readonly int _targetWidth;
    private readonly int _targetHeight;
    private readonly Random _random = new();

    public delegate void ProgressCallback(uint[] pixels, int width, int height, int level, string message);

    public HierarchicalWFC(MultiScaleContextGraph graph, int targetWidth, int targetHeight)
    {
        _multiScaleGraph = graph;
        _targetWidth = targetWidth;
        _targetHeight = targetHeight;
    }

    public uint[] GenerateHierarchical(ProgressCallback? progressCallback = null)
    {
        Console.WriteLine($"[HierarchicalWFC] Starting SEQUENTIAL hierarchical generation: {_targetWidth}x{_targetHeight}");

        // Level 1: Generate at 1/4 resolution (coarse)
        var level1Width = Math.Max(_targetWidth / 4, 64);
        var level1Height = Math.Max(_targetHeight / 4, 64);

        Console.WriteLine($"[HierarchicalWFC] Level 1: {level1Width}x{level1Height} (coarse)");
        // Provide an initial empty frame to the UI to size the surface safely
        progressCallback?.Invoke(new uint[level1Width * level1Height], level1Width, level1Height, 1, "Generating coarse structure...");

        var level1 = GenerateLevelSequential(level1Width, level1Height, null, 1, progressCallback, 1); // 3x3 patterns only
        progressCallback?.Invoke(level1, level1Width, level1Height, 1, "Coarse structure complete");

        // Level 2: Upscale to 1/2 resolution (medium)
        var level2Width = Math.Max(_targetWidth / 2, 128);
        var level2Height = Math.Max(_targetHeight / 2, 128);

        Console.WriteLine($"[HierarchicalWFC] Level 2: {level2Width}x{level2Height} (medium)");
        progressCallback?.Invoke(new uint[level2Width * level2Height], level2Width, level2Height, 2, "Refining structure...");

        var level2 = GenerateLevelSequential(level2Width, level2Height, level1, 1, progressCallback, 2); // 3x3 patterns only
        progressCallback?.Invoke(level2, level2Width, level2Height, 2, "Medium detail complete");

        // Level 3: Final resolution (fine) - sequential streaming
        Console.WriteLine($"[HierarchicalWFC] Level 3: {_targetWidth}x{_targetHeight} (fine) - sequential");
        progressCallback?.Invoke(new uint[_targetWidth * _targetHeight], _targetWidth, _targetHeight, 3, "Adding fine details...");

        var level3 = GenerateLevelSequential(_targetWidth, _targetHeight, level2, 1, progressCallback, 3); // 3x3 patterns
        progressCallback?.Invoke(level3, _targetWidth, _targetHeight, 3, "Complete!");

        Console.WriteLine($"[HierarchicalWFC] Hierarchical generation complete");
        return level3;
    }

    private uint[] GenerateLevelParallel(int width, int height, uint[]? previousLevel, int scaleRadius)
    {
        var pixels = new uint[width * height];
        var collapsed = new ColorRgb?[height, width];

        // Seed constraints from previous level in parallel
        if (previousLevel != null)
        {
            var prevWidth = (int)Math.Sqrt(previousLevel.Length * width / height);
            var prevHeight = previousLevel.Length / prevWidth;

            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    var prevX = x * prevWidth / width;
                    var prevY = y * prevHeight / height;
                    var prevPixel = previousLevel[prevY * prevWidth + prevX];

                    if (_random.NextDouble() < 0.3)
                    {
                        collapsed[y, x] = PixelToColor(prevPixel);
                    }
                }
            });
        }

        // Parallel WFC collapse
        var wfc = new ConstrainedWFC(_multiScaleGraph, width, height, collapsed, scaleRadius);
        return wfc.GenerateParallel();
    }

    private uint[] GenerateLevelSequential(int width, int height, uint[]? previousLevel, int scaleRadius, ProgressCallback? progress, int level)
    {
        var pixels = new uint[width * height];
        var collapsed = new ColorRgb?[height, width];

        // Seed constraints from previous level (down/upscale)
        if (previousLevel != null)
        {
            // Infer previous level dimensions similarly to parallel version
            var prevWidth = (int)Math.Sqrt(previousLevel.Length * width / (double)height);
            if (prevWidth <= 0) prevWidth = Math.Max(1, width / 2);
            var prevHeight = previousLevel.Length / prevWidth;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var prevX = x * prevWidth / width;
                    var prevY = y * prevHeight / height;
                    var prevPixel = previousLevel[prevY * prevWidth + prevX];

                    // Use a slightly higher constraint ratio for smoother guidance
                    if (_random.NextDouble() < 0.5)
                    {
                        collapsed[y, x] = PixelToColor(prevPixel);
                    }
                }
            }
        }

        var wfc = new ConstrainedWFC(_multiScaleGraph, width, height, collapsed, scaleRadius);

        // Throttle UI updates to ~30–60ms per frame
        var sw = new System.Diagnostics.Stopwatch();
        sw.Start();
        int lastRowReported = -1;

        // Generate sequentially, streaming progress row-by-row
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (collapsed[y, x] == null)
                {
                    collapsed[y, x] = wfc.ChooseColorForPublic(x, y); // use exposed chooser
                }
                var color = collapsed[y, x]!.Value;
                pixels[y * width + x] = wfc.ColorToPixelPublic(color);
            }

            // Progress update per row with light time-based throttle
            if (progress != null)
            {
                if (sw.ElapsedMilliseconds >= 33 || y == height - 1)
                {
                    lastRowReported = y;
                    progress(pixels, width, height, level, $"Row {y + 1}/{height}");
                    sw.Restart();
                }
            }
        }

        return pixels;
    }

    private uint[] GenerateLevelTiledParallel(int width, int height, uint[]? previousLevel, int scaleRadius)
    {
        // Subdivide into 4x4 tiles for parallel processing
        var tileSize = Math.Min(width / 4, height / 4);
        var tilesX = (width + tileSize - 1) / tileSize;
        var tilesY = (height + tileSize - 1) / tileSize;

        Console.WriteLine($"[HierarchicalWFC] Processing {tilesX}x{tilesY} tiles in parallel");

        var finalPixels = new uint[width * height];
        var tiles = new List<(int x, int y, int w, int h)>();

        for (int ty = 0; ty < tilesY; ty++)
        {
            for (int tx = 0; tx < tilesX; tx++)
            {
                var tileX = tx * tileSize;
                var tileY = ty * tileSize;
                var tileW = Math.Min(tileSize, width - tileX);
                var tileH = Math.Min(tileSize, height - tileY);
                tiles.Add((tileX, tileY, tileW, tileH));
            }
        }

        // Process tiles in parallel
        System.Threading.Tasks.Parallel.ForEach(tiles, tile =>
        {
            var (tileX, tileY, tileW, tileH) = tile;
            var tilePixels = GenerateLevel(tileW, tileH, previousLevel, scaleRadius);

            // Copy tile back to final image
            lock (finalPixels)
            {
                for (int y = 0; y < tileH; y++)
                {
                    for (int x = 0; x < tileW; x++)
                    {
                        var srcIdx = y * tileW + x;
                        var dstIdx = (tileY + y) * width + (tileX + x);
                        finalPixels[dstIdx] = tilePixels[srcIdx];
                    }
                }
            }
        });

        return finalPixels;
    }

    private uint[] GenerateLevel(int width, int height, uint[]? previousLevel, int scaleRadius)
    {
        var pixels = new uint[width * height];
        var collapsed = new ColorRgb?[height, width];

        // If we have a previous level, use it as constraints
        if (previousLevel != null)
        {
            // Upscale previous level to provide hints
            var prevWidth = (int)Math.Sqrt(previousLevel.Length * width / height);
            var prevHeight = previousLevel.Length / prevWidth;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Map to previous level coordinates
                    var prevX = x * prevWidth / width;
                    var prevY = y * prevHeight / height;
                    var prevPixel = previousLevel[prevY * prevWidth + prevX];

                    // Use previous level as a constraint (seed some pixels)
                    if (_random.NextDouble() < 0.3) // 30% constraint density
                    {
                        collapsed[y, x] = PixelToColor(prevPixel);
                    }
                }
            }
        }

        // Run WFC on this level with constraints
        var wfc = new ConstrainedWFC(_multiScaleGraph, width, height, collapsed, scaleRadius);
        return wfc.Generate();
    }

    private ColorRgb PixelToColor(uint pixel)
    {
        var a = (byte)((pixel >> 24) & 0xFF);
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);

        if (a < 255)
        {
            var alpha = a / 255.0;
            r = (byte)(r * alpha + 255 * (1 - alpha));
            g = (byte)(g * alpha + 255 * (1 - alpha));
            b = (byte)(b * alpha + 255 * (1 - alpha));
        }

        return new ColorRgb(r, g, b);
    }

    private class ConstrainedWFC
    {
        private readonly MultiScaleContextGraph _graph;
        private readonly int _width;
        private readonly int _height;
        private readonly ColorRgb?[,] _collapsed;
        private readonly int _scaleRadius;
        private readonly Random _random = new();

        public ConstrainedWFC(MultiScaleContextGraph graph, int width, int height, ColorRgb?[,] constraints, int scaleRadius)
        {
            _graph = graph;
            _width = width;
            _height = height;
            _collapsed = constraints;
            _scaleRadius = scaleRadius;
        }

        public uint[] Generate()
        {
            // Simple constrained collapse
            for (int y = 0; y < _height; y++)
            {
                for (int x = 0; x < _width; x++)
                {
                    if (_collapsed[y, x] == null)
                    {
                        _collapsed[y, x] = ChooseColor(x, y);
                    }
                }
            }

            return CreatePixelArray();
        }

        public uint[] GenerateParallel()
        {
            // Parallel constrained collapse by rows
            System.Threading.Tasks.Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    if (_collapsed[y, x] == null)
                    {
                        _collapsed[y, x] = ChooseColor(x, y);
                    }
                }
            });

            return CreatePixelArray();
        }

        private uint[] CreatePixelArray()
        {
            var pixels = new uint[_width * _height];

            // Parallel pixel conversion
            System.Threading.Tasks.Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    var color = _collapsed[y, x] ?? new ColorRgb(0, 0, 0);
                    pixels[y * _width + x] = ColorToPixel(color);
                }
            });

            return pixels;
        }

        public ColorRgb ChooseColorForPublic(int x, int y)
        {
            return ChooseColor(x, y);
        }

        public uint ColorToPixelPublic(ColorRgb color)
        {
            return ColorToPixel(color);
        }

        private ColorRgb ChooseColor(int x, int y)
        {
            // Build neighborhood pattern from already collapsed cells
            var neighbors = new Dictionary<Direction, ColorRgb?>();
            for (int i = 0; i < 8; i++)
            {
                var dir = (Direction)i;
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    neighbors[dir] = _collapsed[ny, nx];
                }
                else
                {
                    neighbors[dir] = null;
                }
            }

            // Get average color of neighbors as center estimate
            var validNeighbors = neighbors.Values.Where(c => c.HasValue).Select(c => c!.Value).ToList();
            var centerColor = validNeighbors.Any() 
                ? new ColorRgb(
                    (byte)validNeighbors.Average(c => c.R),
                    (byte)validNeighbors.Average(c => c.G),
                    (byte)validNeighbors.Average(c => c.B))
                : _graph.GetAllColors()[_random.Next(_graph.GetAllColors().Count)];

            var pattern = new NeighborhoodPattern(centerColor, neighbors);

            // Get weighted predictions for an arbitrary direction (or average across all)
            var allPredictions = new Dictionary<ColorRgb, double>();

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var predictions = _graph.GetWeightedNeighborsMultiScale(pattern, dir);
                foreach (var (color, weight) in predictions)
                {
                    if (!allPredictions.ContainsKey(color))
                        allPredictions[color] = 0;
                    allPredictions[color] += weight / 8.0; // Average across directions
                }
            }

            // Weighted random selection
            if (allPredictions.Any())
            {
                var totalWeight = allPredictions.Values.Sum();
                var rand = _random.NextDouble() * totalWeight;
                var cumulative = 0.0;

                foreach (var (color, weight) in allPredictions)
                {
                    cumulative += weight;
                    if (rand <= cumulative)
                        return color;
                }

                return allPredictions.First().Key;
            }

            return centerColor;
        }

        private uint ColorToPixel(ColorRgb color)
        {
            return (uint)((255 << 24) | (color.R << 16) | (color.G << 8) | color.B);
        }
    }
}
