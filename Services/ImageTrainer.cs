using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using NNImage.Models;

namespace NNImage.Services;

public class ImageTrainer
{
    private readonly WeightedContextGraph _contextGraph = new();
    private readonly ColorQuantizer _quantizer;
    private readonly int _quantizationLevel;
    private bool _paletteBuilt;
    private readonly GpuAccelerator _gpu;
    private readonly int _neighborhoodRadius = 1; // Use 3x3 neighborhood (radius 1)

    public ImageTrainer(int quantizationLevel, GpuAccelerator gpu)
    {
        _quantizationLevel = quantizationLevel;
        _gpu = gpu;
        _quantizer = new ColorQuantizer(quantizationLevel, gpu);
        Console.WriteLine($"[ImageTrainer] Created with quantization level: {quantizationLevel}");
        Console.WriteLine($"[ImageTrainer] GPU acceleration: {(gpu?.IsAvailable == true ? "ENABLED" : "DISABLED")}");
        Console.WriteLine($"[ImageTrainer] Using {(_neighborhoodRadius * 2 + 1)}x{(_neighborhoodRadius * 2 + 1)} neighborhood patterns");
    }

    public void ProcessImageData(uint[] pixels, int width, int height)
    {
        Console.WriteLine($"[ImageTrainer] ProcessImageData called - Size: {width}x{height}, Pixels: {pixels.Length}");

        try
        {
            // Extract all colors first if palette not built
            if (!_paletteBuilt)
            {
                Console.WriteLine($"[ImageTrainer] Building color palette with parallel processing...");
                var allColors = new System.Collections.Concurrent.ConcurrentBag<ColorRgb>();

                // Parallel color extraction
                System.Threading.Tasks.Parallel.For(0, pixels.Length, i =>
                {
                    var color = PixelToColor(pixels[i]);
                    allColors.Add(color);
                });

                Console.WriteLine($"[ImageTrainer] Extracted {allColors.Count} color samples, quantizing...");
                _quantizer.BuildPalette(allColors.ToList());
                _paletteBuilt = true;
                Console.WriteLine($"[ImageTrainer] Palette built successfully");
            }

            Console.WriteLine($"[ImageTrainer] Extracting neighborhood patterns with context awareness...");

            var localPatterns = new System.Collections.Concurrent.ConcurrentBag<(NeighborhoodPattern pattern, Direction dir, ColorRgb target)>();

            // Parallel pattern extraction by rows
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    var centerPixel = pixels[y * width + x];
                    var centerColor = _quantizer.Quantize(PixelToColor(centerPixel));

                    // Build neighborhood pattern
                    var neighbors = new Dictionary<Direction, ColorRgb?>();
                    foreach (var direction in DirectionExtensions.AllDirections)
                    {
                        var (dx, dy) = direction.GetOffset();
                        var nx = x + dx;
                        var ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            var neighborPixel = pixels[ny * width + nx];
                            neighbors[direction] = _quantizer.Quantize(PixelToColor(neighborPixel));
                        }
                        else
                        {
                            neighbors[direction] = null; // Out of bounds
                        }
                    }

                    var pattern = new NeighborhoodPattern(centerColor, neighbors);

                    // For each direction, record what color actually appeared
                    foreach (var direction in DirectionExtensions.AllDirections)
                    {
                        var targetColor = neighbors[direction];
                        if (targetColor.HasValue)
                        {
                            localPatterns.Add((pattern, direction, targetColor.Value));

                            // Also add simple adjacency for fallback
                            _contextGraph.AddSimpleAdjacency(centerColor, direction, targetColor.Value);
                        }
                    }
                }
            });

            // Batch add patterns to graph with weighting
            Console.WriteLine($"[ImageTrainer] Adding {localPatterns.Count} context patterns to graph...");

            var patternCounts = new ConcurrentDictionary<(NeighborhoodPattern, Direction, ColorRgb), int>();

            // Parallel counting
            System.Threading.Tasks.Parallel.ForEach(
                localPatterns.GroupBy(p => p).Select(g => (pattern: g.Key, count: g.Count())),
                item =>
                {
                    patternCounts[item.pattern] = item.count;
                });

            // Parallel insertion with batching
            System.Threading.Tasks.Parallel.ForEach(patternCounts, kvp =>
            {
                var ((pattern, dir, target), count) = kvp;
                _contextGraph.AddPattern(pattern, dir, target, count);
            });

            Console.WriteLine($"[ImageTrainer] Extracted {patternCounts.Count} unique context patterns");
            Console.WriteLine($"[ImageTrainer] Total pattern instances: {localPatterns.Count}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ImageTrainer] ERROR in ProcessImageData: {ex.Message}");
            Console.WriteLine($"[ImageTrainer] Stack trace: {ex.StackTrace}");
            throw;
        }
    }

    public WeightedContextGraph GetContextGraph()
    {
        _contextGraph.SetGpuAccelerator(_gpu);
        _contextGraph.Normalize();
        Console.WriteLine($"[ImageTrainer] Context graph contains {_contextGraph.GetPatternCount()} patterns and {_contextGraph.GetColorCount()} unique colors");
        return _contextGraph;
    }

    // Backward compatibility
    public AdjacencyGraph GetAdjacencyGraph()
    {
        // Create a simple adjacency graph from the context graph for compatibility
        var simpleGraph = new AdjacencyGraph();
        var colors = _contextGraph.GetAllColors();

        foreach (var color in colors)
        {
            var emptyNeighbors = new Dictionary<Direction, ColorRgb?>();
            var pattern = new NeighborhoodPattern(color, emptyNeighbors);

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var neighbors = _contextGraph.GetPossibleNeighbors(pattern, dir);
                foreach (var neighbor in neighbors)
                {
                    simpleGraph.AddAdjacency(color, dir, neighbor);
                }
            }
        }

        simpleGraph.Normalize();
        return simpleGraph;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private ColorRgb PixelToColor(uint pixel)
    {
        var a = (byte)((pixel >> 24) & 0xFF);
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);

        // Handle transparency by blending with white
        if (a < 255)
        {
            var alpha = a / 255.0;
            r = (byte)(r * alpha + 255 * (1 - alpha));
            g = (byte)(g * alpha + 255 * (1 - alpha));
            b = (byte)(b * alpha + 255 * (1 - alpha));
        }

        return new ColorRgb(r, g, b);
    }
}
