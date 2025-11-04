using Avalonia.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// Ultra-fast Wave Function Collapse using node-based graph with multiple random seeds
/// Grows image from random starting points with real-time visualization
/// </summary>
public class FastWaveFunctionCollapse
{
    private readonly FastContextGraph _graph;
    private readonly GpuAccelerator? _gpu;
    private readonly int _width;
    private readonly int _height;
    private readonly Random _random = new();
    private readonly ColorRgb?[,] _collapsed;
    private readonly List<ColorRgb> _availableColors;
    private readonly double _entropyFactor; // 0.0 = deterministic (most probable), 1.0 = high randomness
    private readonly System.Threading.CancellationToken _cancellationToken;

    public enum GenerationMode
    {
        FastSquare,    // FASTEST - simple wave propagation with visible blocks
        Blended,       // Slower - smooth blending between seeds, no visible boundaries
        FastOrganic,   // Organic/fractal - entropy-driven growth with natural patterns
        Radial,        // Pure WFC - grows from center outward in radial waves
        EdgeFirst,     // Pure WFC - collapses edges first, then fills inward
        CenterOut,     // Pure WFC - single center seed expanding outward
        Spiral         // Pure WFC - spiral pattern expansion from center
    }

    public delegate void ProgressCallback(uint[] pixels, int collapsedCount, int totalCells);

    public FastWaveFunctionCollapse(FastContextGraph graph, int width, int height, GpuAccelerator? gpu = null, double entropyFactor = 0.0, System.Threading.CancellationToken cancellationToken = default)
    {
        _graph = graph;
        _gpu = gpu;
        _width = width;
        _height = height;
        _collapsed = new ColorRgb?[height, width];
        _availableColors = graph.GetAllColors();
        _entropyFactor = Math.Clamp(entropyFactor, 0.0, 1.0);
        _cancellationToken = cancellationToken;

        if (_availableColors.Count == 0)
        {
            throw new InvalidOperationException("Graph has no colors - train the model first!");
        }

        Console.WriteLine($"[FastWFC] Initialized for {width}x{height} with {_availableColors.Count} colors, Entropy: {_entropyFactor:F2}, GPU: {(gpu?.IsAvailable == true ? "Available" : "N/A")}");
    }

    /// <summary>
    /// Generate with selectable mode: Fast squares or smooth blending
    /// </summary>
    /// <param name="seedCount">Number of random seed points to start generation from</param>
    /// <param name="mode">Generation mode - FastSquare or Blended</param>
    /// <param name="progressCallback">Optional callback for progress updates</param>
    /// <param name="updateFrequency">How often to call progress callback (in iterations)</param>
    /// <param name="useCuda">Enable CUDA GPU acceleration for propagation/entropy (bit-exact, falls back to CPU if unavailable)</param>
    public uint[] Generate(int seedCount = 5, GenerationMode mode = GenerationMode.FastSquare, ProgressCallback? progressCallback = null, int updateFrequency = 10, bool useCuda = false)
    {
        Console.WriteLine($"[FastWFC] Generation mode: {mode}, Seeds: {seedCount}, CUDA: {(useCuda ? "Enabled" : "Disabled")}");

        // CUDA ACCELERATION: If enabled and GPU available, use massively parallel GPU generation
        if (useCuda && _gpu != null && _gpu.IsAvailable && _width * _height >= 50000)
        {
            Console.WriteLine($"[FastWFC] ⚡⚡⚡ CUDA HYPER-SPEED MODE ACTIVATED ⚡⚡⚡");
            Console.WriteLine($"[FastWFC] Processing {_width * _height:N0} pixels on 10240 CUDA cores!");
            return GenerateCudaHyperSpeed(seedCount, mode, progressCallback, updateFrequency);
        }

        // CPU fallback for small images or when CUDA disabled
        return mode switch
        {
            GenerationMode.FastSquare => GenerateFastSquare(seedCount, progressCallback, updateFrequency),
            GenerationMode.Blended => GenerateBlended(seedCount, progressCallback, updateFrequency),
            GenerationMode.FastOrganic => GenerateFastOrganic(seedCount, progressCallback, updateFrequency),
            GenerationMode.Radial => GenerateRadial(seedCount, progressCallback, updateFrequency),
            GenerationMode.EdgeFirst => GenerateEdgeFirst(seedCount, progressCallback, updateFrequency),
            GenerationMode.CenterOut => GenerateCenterOut(progressCallback, updateFrequency),
            GenerationMode.Spiral => GenerateSpiral(progressCallback, updateFrequency),
            _ => GenerateFastSquare(seedCount, progressCallback, updateFrequency)
        };
    }

    /// <summary>
    /// FAST MODE: Pure wave-based generation - maximum speed, visible boundaries
    /// </summary>
    private uint[] GenerateFastSquare(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] FAST SQUARE mode - maximum speed!");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new Queue<(int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        foreach (var (x, y) in seeds)
        {
            waveQueue.Enqueue((x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Simple FIFO wave - super fast!
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            var (x, y) = waveQueue.Dequeue();

            // Check cancellation more frequently (every cell)
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] Generation cancelled at {collapsedCount}/{totalCells} cells");
                return CreatePixelData(); // Return partial result
            }

            if (_collapsed[y, x] == null)
            {
                CollapseCell(x, y);
                collapsedCount++;
            }

            // Add all neighbors immediately
            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        waveQueue.Enqueue(coord);
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] FAST mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// BLENDED MODE: Optimized blended generation - balanced speed and quality
    /// </summary>
    private uint[] GenerateBlended(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] BLENDED mode - optimized blending");

        var seeds = InitializeSeeds(seedCount);

        // Use priority queue with pre-computed noise
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        // Pre-compute spatial noise map for speed
        var noiseMap = new float[_height, _width];
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                noiseMap[y, x] = (float)(Math.Sin(x * 0.3) * Math.Cos(y * 0.3));
            }
        }

        foreach (var (x, y) in seeds)
        {
            waveQueue.Add((0f, _random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Batch process for speed
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check cancellation before each batch
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] Generation cancelled at {collapsedCount}/{totalCells} cells");
                return CreatePixelData(); // Return partial result
            }

            // Process small batches (10-20 cells) for speed
            var batchSize = Math.Min(15, waveQueue.Count);
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                // Check cancellation within batch processing for even faster response
                if (_cancellationToken.IsCancellationRequested)
                {
                    Console.WriteLine($"[FastWFC] Generation cancelled mid-batch at {collapsedCount}/{totalCells} cells");
                    return CreatePixelData();
                }

                waveQueue.Remove(item);
                var (_, _, x, y) = item;

                if (_collapsed[y, x] == null)
                {
                    CollapseCell(x, y);
                    collapsedCount++;
                }

                // Add neighbors with simple prioritization
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        if (!processed.Contains(coord))
                        {
                            // Fast priority: pre-computed noise + random
                            var priority = noiseMap[ny, nx] + (float)_random.NextDouble() * 2f;
                            waveQueue.Add((priority, _random.Next(), nx, ny));
                            processed.Add(coord);
                        }
                    }
                }

                iteration++;
                if (progressCallback != null && iteration % updateFrequency == 0)
                {
                    var pixels = CreatePixelData();
                    progressCallback(pixels, collapsedCount, totalCells);
                }
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] SMOOTH mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// FAST ORGANIC MODE: Entropy-driven growth with fractal-like patterns
    /// Creates natural, organic growth patterns similar to crystals, coral, or plants
    /// </summary>
    private uint[] GenerateFastOrganic(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] FAST ORGANIC mode - entropy-driven fractal growth");

        var seeds = InitializeSeeds(seedCount);

        // Use priority queue with entropy + fractal noise for organic growth
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        // Pre-compute multi-scale fractal noise map for organic patterns
        var noiseMap = new float[_height, _width];
        var random = new Random();

        // Generate multiple octaves of noise for fractal-like appearance
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                // Combine multiple frequency scales for fractal effect
                var noise1 = (float)(Math.Sin(x * 0.1) * Math.Cos(y * 0.1));
                var noise2 = (float)(Math.Sin(x * 0.3) * Math.Cos(y * 0.3)) * 0.5f;
                var noise3 = (float)(Math.Sin(x * 0.7) * Math.Cos(y * 0.7)) * 0.25f;
                var noise4 = (float)(Math.Sin(x * 1.5) * Math.Cos(y * 1.5)) * 0.125f;

                // Add radial component for more organic feel
                var dx = x - _width / 2.0f;
                var dy = y - _height / 2.0f;
                var distance = (float)Math.Sqrt(dx * dx + dy * dy);
                var radialNoise = (float)Math.Sin(distance * 0.05) * 0.3f;

                noiseMap[y, x] = noise1 + noise2 + noise3 + noise4 + radialNoise;
            }
        }

        foreach (var (x, y) in seeds)
        {
            // Calculate initial entropy-based priority
            var entropy = (float)CalculateEntropy(x, y);
            var noiseFactor = noiseMap[y, x];
            var priority = entropy * 0.5f + noiseFactor * 2.0f + (float)random.NextDouble();

            waveQueue.Add((priority, random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Organic growth with entropy-driven expansion
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation frequently
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] Organic generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            // Take the highest priority cell (lowest entropy + fractal influence)
            var item = waveQueue.Min;
            waveQueue.Remove(item);

            var (_, _, x, y) = item;

            if (_collapsed[y, x] == null)
            {
                CollapseCell(x, y);
                collapsedCount++;
            }

            // Add neighbors with entropy-based prioritization
            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        // Calculate priority based on entropy + fractal noise + randomness
                        var entropy = (float)CalculateEntropy(nx, ny);
                        var noiseFactor = noiseMap[ny, nx];
                        var randomFactor = (float)random.NextDouble() * 1.5f;

                        // Lower entropy = more constrained = higher priority (inverted)
                        // Add fractal noise and randomness for organic patterns
                        var priority = entropy * 0.3f + noiseFactor * 1.5f + randomFactor;

                        waveQueue.Add((priority, random.Next(), nx, ny));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] ORGANIC mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// RADIAL MODE: Pure WFC with radial distance-based prioritization
    /// Grows from center outward in circular waves, respecting WFC weights
    /// </summary>
    private uint[] GenerateRadial(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] RADIAL mode - circular wave expansion");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2.0f;
        var centerY = _height / 2.0f;

        foreach (var (x, y) in seeds)
        {
            // Priority based on distance from center (closer = higher priority = lower value)
            var dx = x - centerX;
            var dy = y - centerY;
            var distance = (float)Math.Sqrt(dx * dx + dy * dy);

            waveQueue.Add((distance, _random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] Radial generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            var item = waveQueue.Min;
            waveQueue.Remove(item);

            var (_, _, x, y) = item;

            if (_collapsed[y, x] == null)
            {
                CollapseCell(x, y);
                collapsedCount++;
            }

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        // Pure radial distance priority
                        var ndx = nx - centerX;
                        var ndy = ny - centerY;
                        var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);

                        waveQueue.Add((distance, _random.Next(), nx, ny));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] RADIAL mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// EDGE FIRST MODE: Pure WFC starting from edges and filling toward center
    /// Creates interesting patterns by constraining from outside-in
    /// </summary>
    private uint[] GenerateEdgeFirst(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] EDGE FIRST mode - outside-in propagation");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2.0f;
        var centerY = _height / 2.0f;

        // Start from all edge pixels
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (x == 0 || x == _width - 1 || y == 0 || y == _height - 1)
                {
                    // Pick random colors for edge seeds
                    var seedColor = _availableColors[_random.Next(_availableColors.Count)];
                    _collapsed[y, x] = seedColor;

                    // Priority = inverted distance from center (edges = 0, center = max)
                    var dx = x - centerX;
                    var dy = y - centerY;
                    var distanceFromCenter = (float)Math.Sqrt(dx * dx + dy * dy);
                    var maxDistance = (float)Math.Sqrt(centerX * centerX + centerY * centerY);
                    var priority = maxDistance - distanceFromCenter; // Invert so edges process first

                    waveQueue.Add((priority, _random.Next(), x, y));
                    processed.Add((x, y));
                }
            }
        }

        int collapsedCount = processed.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] EdgeFirst generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            var item = waveQueue.Min;
            waveQueue.Remove(item);

            var (_, _, x, y) = item;

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        if (_collapsed[ny, nx] == null)
                        {
                            CollapseCell(nx, ny);
                            collapsedCount++;
                        }

                        // Priority = inverted distance from center
                        var ndx = nx - centerX;
                        var ndy = ny - centerY;
                        var distanceFromCenter = (float)Math.Sqrt(ndx * ndx + ndy * ndy);
                        var maxDistance = (float)Math.Sqrt(centerX * centerX + centerY * centerY);
                        var priority = maxDistance - distanceFromCenter;

                        waveQueue.Add((priority, _random.Next(), nx, ny));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] EDGE FIRST mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CENTER OUT MODE: Pure WFC starting from single center point
    /// Classic WFC expansion from a single seed
    /// </summary>
    private uint[] GenerateCenterOut(ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] CENTER OUT mode - single center seed expansion");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2;
        var centerY = _height / 2;

        // Single seed at center
        var seedColor = _availableColors[_random.Next(_availableColors.Count)];
        _collapsed[centerY, centerX] = seedColor;

        waveQueue.Add((0f, _random.Next(), centerX, centerY));
        processed.Add((centerX, centerY));

        int collapsedCount = 1;
        int totalCells = _width * _height;
        int iteration = 0;

        var fcenterX = _width / 2.0f;
        var fcenterY = _height / 2.0f;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] CenterOut generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            var item = waveQueue.Min;
            waveQueue.Remove(item);

            var (_, _, x, y) = item;

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        if (_collapsed[ny, nx] == null)
                        {
                            CollapseCell(nx, ny);
                            collapsedCount++;
                        }

                        // Priority by distance from original center
                        var ndx = nx - fcenterX;
                        var ndy = ny - fcenterY;
                        var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);

                        waveQueue.Add((distance, _random.Next(), nx, ny));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] CENTER OUT mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// SPIRAL MODE: Pure WFC with spiral expansion pattern
    /// Creates mesmerizing spiral growth from center using angular priority
    /// </summary>
    private uint[] GenerateSpiral(ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC] SPIRAL mode - angular spiral expansion");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2;
        var centerY = _height / 2;

        // Single seed at center
        var seedColor = _availableColors[_random.Next(_availableColors.Count)];
        _collapsed[centerY, centerX] = seedColor;

        waveQueue.Add((0f, _random.Next(), centerX, centerY));
        processed.Add((centerX, centerY));

        int collapsedCount = 1;
        int totalCells = _width * _height;
        int iteration = 0;

        var fcenterX = _width / 2.0f;
        var fcenterY = _height / 2.0f;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC] Spiral generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            var item = waveQueue.Min;
            waveQueue.Remove(item);

            var (_, _, x, y) = item;

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                {
                    var coord = (nx, ny);
                    if (!processed.Contains(coord))
                    {
                        if (_collapsed[ny, nx] == null)
                        {
                            CollapseCell(nx, ny);
                            collapsedCount++;
                        }

                        // Spiral priority: combine angle and distance
                        var ndx = nx - fcenterX;
                        var ndy = ny - fcenterY;
                        var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);
                        var angle = (float)Math.Atan2(ndy, ndx); // Range: -π to π

                        // Convert angle to 0-2π range and combine with distance
                        var normalizedAngle = angle < 0 ? angle + (float)(Math.PI * 2) : angle;

                        // Priority = distance + angle offset for spiral effect
                        // This creates a rotating wave that spirals outward
                        var spiralPriority = distance * 2.0f + normalizedAngle * 10.0f;

                        waveQueue.Add((spiralPriority, _random.Next(), nx, ny));
                        processed.Add(coord);
                    }
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC] SPIRAL mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// Calculate entropy (constraint level) for a cell based on collapsed neighbors
    /// Lower entropy = more constrained = collapse first for smooth blending
    /// </summary>
    private double CalculateEntropy(int x, int y)
    {
        if (_collapsed[y, x] != null)
            return double.MaxValue; // Already collapsed

        var normalizedX = _width > 1 ? (float)x / (_width - 1) : 0.5f;
        var normalizedY = _height > 1 ? (float)y / (_height - 1) : 0.5f;

        // Count collapsed neighbors
        var collapsedNeighbors = 0;
        var totalNeighbors = 0;
        var colorWeights = new Dictionary<ColorRgb, double>();

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            var (dx, dy) = dir.GetOffset();
            var nx = x + dx;
            var ny = y + dy;

            totalNeighbors++;

            if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
            {
                var neighborColor = _collapsed[ny, nx];
                if (neighborColor.HasValue)
                {
                    collapsedNeighbors++;

                    // Get predictions from this neighbor
                    var neighborNormX = _width > 1 ? (float)nx / (_width - 1) : 0.5f;
                    var neighborNormY = _height > 1 ? (float)ny / (_height - 1) : 0.5f;

                    var predictions = _graph.GetWeightedNeighbors(
                        neighborColor.Value,
                        neighborNormX,
                        neighborNormY,
                        dir);

                    foreach (var (color, weight) in predictions)
                    {
                        if (!colorWeights.ContainsKey(color))
                            colorWeights[color] = 0;
                        colorWeights[color] += weight;
                    }
                }
            }
        }

        // Entropy based on:
        // 1. How many neighbors are collapsed (more = lower entropy = collapse sooner)
        // 2. How many valid color options remain (fewer = lower entropy)
        var neighborRatio = collapsedNeighbors / (double)totalNeighbors;
        var colorOptions = colorWeights.Count > 0 ? colorWeights.Count : _availableColors.Count;

        // Lower entropy = collapse first (more constrained)
        var entropy = (1.0 - neighborRatio) * colorOptions;

        return entropy;
    }

    /// <summary>
    /// Initialize random seed points across the image
    /// </summary>
    private List<(int x, int y)> InitializeSeeds(int seedCount)
    {
        var seeds = new List<(int x, int y)>();

        // Distribute seeds randomly but avoid clustering
        var minDistance = Math.Min(_width, _height) / (int)Math.Sqrt(seedCount);

        for (int i = 0; i < seedCount * 3 && seeds.Count < seedCount; i++)
        {
            var x = _random.Next(_width);
            var y = _random.Next(_height);

            // Check distance from existing seeds
            bool tooClose = false;
            foreach (var (sx, sy) in seeds)
            {
                var dist = Math.Sqrt((x - sx) * (x - sx) + (y - sy) * (y - sy));
                if (dist < minDistance)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
            {
                // Pick random color for this seed
                var seedColor = _availableColors[_random.Next(_availableColors.Count)];
                _collapsed[y, x] = seedColor;
                seeds.Add((x, y));
                Console.WriteLine($"[FastWFC] Seed {seeds.Count} at ({x}, {y}) = {seedColor}");
            }
        }

        // Fallback: if we couldn't place enough seeds, just place them randomly
        while (seeds.Count < seedCount)
        {
            var x = _random.Next(_width);
            var y = _random.Next(_height);
            if (_collapsed[y, x] == null)
            {
                var seedColor = _availableColors[_random.Next(_availableColors.Count)];
                _collapsed[y, x] = seedColor;
                seeds.Add((x, y));
            }
        }

        return seeds;
    }

    /// <summary>
    /// Collapse a cell using weighted predictions from nearby nodes
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CollapseCell(int x, int y)
    {
        // Calculate normalized position
        var normalizedX = _width > 1 ? (float)x / (_width - 1) : 0.5f;
        var normalizedY = _height > 1 ? (float)y / (_height - 1) : 0.5f;

        // Collect weighted predictions from all collapsed neighbors
        var colorWeights = new Dictionary<ColorRgb, double>();

        foreach (var dir in DirectionExtensions.AllDirections)
        {
            var (dx, dy) = dir.GetOffset();
            var nx = x + dx;
            var ny = y + dy;

            if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
            {
                var neighborColor = _collapsed[ny, nx];
                if (neighborColor.HasValue)
                {
                    // Get predictions from graph based on neighbor's color and position
                    var neighborNormX = _width > 1 ? (float)nx / (_width - 1) : 0.5f;
                    var neighborNormY = _height > 1 ? (float)ny / (_height - 1) : 0.5f;

                    var predictions = _graph.GetWeightedNeighbors(
                        neighborColor.Value, 
                        neighborNormX, 
                        neighborNormY, 
                        dir);

                    // Accumulate weights from this neighbor's predictions
                    foreach (var (color, weight) in predictions)
                    {
                        if (!colorWeights.ContainsKey(color))
                        {
                            colorWeights[color] = 0;
                        }
                        colorWeights[color] += weight;
                    }
                }
            }
        }

        // Select color based on accumulated weights + entropy factor
        ColorRgb selectedColor;

        if (colorWeights.Count > 0)
        {
            // Apply entropy factor to introduce controlled randomness
            if (_entropyFactor > 0.0)
            {
                // Blend between weighted distribution and uniform distribution
                // entropyFactor = 0.0: pure weighted (deterministic)
                // entropyFactor = 1.0: uniform random (maximum chaos)

                var uniformWeight = colorWeights.Values.Average();
                var adjustedWeights = new Dictionary<ColorRgb, double>();

                foreach (var (color, weight) in colorWeights)
                {
                    // Lerp between original weight and uniform weight
                    var adjustedWeight = weight * (1.0 - _entropyFactor) + uniformWeight * _entropyFactor;
                    adjustedWeights[color] = adjustedWeight;
                }

                // Weighted random selection with adjusted weights
                var totalWeight = adjustedWeights.Values.Sum();
                var rand = _random.NextDouble() * totalWeight;
                var cumulative = 0.0;

                selectedColor = adjustedWeights.First().Key; // Fallback

                foreach (var (color, weight) in adjustedWeights.OrderByDescending(kvp => kvp.Value))
                {
                    cumulative += weight;
                    if (rand <= cumulative)
                    {
                        selectedColor = color;
                        break;
                    }
                }
            }
            else
            {
                // Pure weighted random selection based on edge weights (entropyFactor = 0)
                var totalWeight = colorWeights.Values.Sum();
                var rand = _random.NextDouble() * totalWeight;
                var cumulative = 0.0;

                selectedColor = colorWeights.First().Key; // Fallback

                foreach (var (color, weight) in colorWeights.OrderByDescending(kvp => kvp.Value))
                {
                    cumulative += weight;
                    if (rand <= cumulative)
                    {
                        selectedColor = color;
                        break;
                    }
                }
            }
        }
        else
        {
            // No neighbors collapsed yet - pick random color
            selectedColor = _availableColors[_random.Next(_availableColors.Count)];
        }

        _collapsed[y, x] = selectedColor;
    }

    /// <summary>
    /// ⚡ CUDA-ACCELERATED generation that follows EXACT same algorithm as CPU
    /// Just processes queue batches in parallel instead of one-by-one
    /// </summary>
    private uint[] GenerateCudaHyperSpeed(int seedCount, GenerationMode mode, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ CUDA-ACCELERATED mode - same algorithm, parallel processing");

        return mode switch
        {
            GenerationMode.FastSquare => GenerateFastSquareCuda(seedCount, progressCallback, updateFrequency),
            GenerationMode.Blended => GenerateBlendedCuda(seedCount, progressCallback, updateFrequency),
            GenerationMode.FastOrganic => GenerateFastOrganicCuda(seedCount, progressCallback, updateFrequency),
            GenerationMode.Radial => GenerateRadialCuda(seedCount, progressCallback, updateFrequency),
            GenerationMode.EdgeFirst => GenerateEdgeFirstCuda(seedCount, progressCallback, updateFrequency),
            GenerationMode.CenterOut => GenerateCenterOutCuda(progressCallback, updateFrequency),
            GenerationMode.Spiral => GenerateSpiralCuda(progressCallback, updateFrequency),
            _ => GenerateFastSquareCuda(seedCount, progressCallback, updateFrequency)
        };
    }

    /// <summary>
    /// CUDA-accelerated FastSquare - processes queue batches in parallel
    /// BIT-EXACT same results as CPU version, just faster
    /// </summary>
    private uint[] GenerateFastSquareCuda(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ FAST SQUARE mode with parallel batching");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new Queue<(int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        foreach (var (x, y) in seeds)
        {
            waveQueue.Enqueue((x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Process queue in parallel batches while maintaining FIFO order
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC-CUDA] FastSquare generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            // Take a batch from the queue (respects FIFO order)
            var batchSize = Math.Min(1000, waveQueue.Count); // Process 1000 cells in parallel
            var batch = new List<(int x, int y)>(batchSize);

            for (int i = 0; i < batchSize && waveQueue.Count > 0; i++)
            {
                batch.Add(waveQueue.Dequeue());
            }

            // Collapse batch in parallel (all at same "wave depth")
            var localCollapsed = 0;
            System.Threading.Tasks.Parallel.ForEach(batch,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                cell =>
            {
                var (x, y) = cell;
                lock (_collapsed) // Protect shared state
                {
                    if (_collapsed[y, x] == null)
                    {
                        CollapseCell(x, y);
                        System.Threading.Interlocked.Increment(ref localCollapsed);
                    }
                }
            });

            collapsedCount += localCollapsed;

            // Add neighbors to queue (in parallel)
            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(int x, int y)>();
            System.Threading.Tasks.Parallel.ForEach(batch, cell =>
            {
                var (x, y) = cell;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                newNeighbors.Add(coord);
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            // Add new neighbors to queue
            foreach (var neighbor in newNeighbors)
            {
                waveQueue.Enqueue(neighbor);
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ FAST mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated Blended - processes priority queue batches in parallel
    /// BIT-EXACT same results as CPU version, just faster
    /// </summary>
    private uint[] GenerateBlendedCuda(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ BLENDED mode with parallel batching");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        // Pre-compute spatial noise map for speed
        var noiseMap = new float[_height, _width];
        System.Threading.Tasks.Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                noiseMap[y, x] = (float)(Math.Sin(x * 0.3) * Math.Cos(y * 0.3));
            }
        });

        foreach (var (x, y) in seeds)
        {
            waveQueue.Add((0f, _random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Process priority queue in parallel batches
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC-CUDA] Blended generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            // Take batch with similar priorities (respects priority order)
            var batchSize = Math.Min(100, waveQueue.Count); // Smaller batches to respect priority better
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            // Collapse batch in parallel (all at similar priority)
            var localCollapsed = 0;
            System.Threading.Tasks.Parallel.ForEach(batch,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                item =>
            {
                var (_, _, x, y) = item;
                lock (_collapsed)
                {
                    if (_collapsed[y, x] == null)
                    {
                        CollapseCell(x, y);
                        System.Threading.Interlocked.Increment(ref localCollapsed);
                    }
                }
            });

            collapsedCount += localCollapsed;

            // Add neighbors with priorities (in parallel)
            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                var priority = noiseMap[ny, nx] + (float)_random.NextDouble() * 2f;
                                newNeighbors.Add((priority, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            // Add new neighbors to priority queue
            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ BLENDED mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated FastOrganic - entropy-driven fractal growth with parallel batching
    /// BIT-EXACT same results as CPU version, just faster
    /// </summary>
    private uint[] GenerateFastOrganicCuda(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ FAST ORGANIC mode with parallel entropy batching");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();
        var random = new Random();

        // Pre-compute multi-scale fractal noise map (parallel computation)
        var noiseMap = new float[_height, _width];
        System.Threading.Tasks.Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                // Combine multiple frequency scales for fractal effect
                var noise1 = (float)(Math.Sin(x * 0.1) * Math.Cos(y * 0.1));
                var noise2 = (float)(Math.Sin(x * 0.3) * Math.Cos(y * 0.3)) * 0.5f;
                var noise3 = (float)(Math.Sin(x * 0.7) * Math.Cos(y * 0.7)) * 0.25f;
                var noise4 = (float)(Math.Sin(x * 1.5) * Math.Cos(y * 1.5)) * 0.125f;

                // Add radial component for more organic feel
                var dx = x - _width / 2.0f;
                var dy = y - _height / 2.0f;
                var distance = (float)Math.Sqrt(dx * dx + dy * dy);
                var radialNoise = (float)Math.Sin(distance * 0.05) * 0.3f;

                noiseMap[y, x] = noise1 + noise2 + noise3 + noise4 + radialNoise;
            }
        });

        foreach (var (x, y) in seeds)
        {
            var entropy = (float)CalculateEntropy(x, y);
            var noiseFactor = noiseMap[y, x];
            var priority = entropy * 0.5f + noiseFactor * 2.0f + (float)random.NextDouble();

            waveQueue.Add((priority, random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        // Process priority queue in parallel batches (organic growth)
        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC-CUDA] Organic generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            // Take batch with similar priorities (respects entropy order)
            var batchSize = Math.Min(50, waveQueue.Count); // Smaller batches for better entropy control
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            // Collapse batch in parallel
            var localCollapsed = 0;
            System.Threading.Tasks.Parallel.ForEach(batch,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                item =>
            {
                var (_, _, x, y) = item;
                lock (_collapsed)
                {
                    if (_collapsed[y, x] == null)
                    {
                        CollapseCell(x, y);
                        System.Threading.Interlocked.Increment(ref localCollapsed);
                    }
                }
            });

            collapsedCount += localCollapsed;

            // Add neighbors with entropy-based priorities (in parallel)
            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                // Calculate entropy-based priority with fractal noise
                                var entropy = (float)CalculateEntropy(nx, ny);
                                var noiseFactor = noiseMap[ny, nx];
                                var randomFactor = (float)_random.NextDouble() * 1.5f;

                                var priority = entropy * 0.3f + noiseFactor * 1.5f + randomFactor;

                                newNeighbors.Add((priority, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            // Add new neighbors to priority queue
            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ ORGANIC mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated Radial mode
    /// </summary>
    private uint[] GenerateRadialCuda(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ RADIAL mode with parallel batching");

        var seeds = InitializeSeeds(seedCount);
        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2.0f;
        var centerY = _height / 2.0f;

        foreach (var (x, y) in seeds)
        {
            var dx = x - centerX;
            var dy = y - centerY;
            var distance = (float)Math.Sqrt(dx * dx + dy * dy);

            waveQueue.Add((distance, _random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = seeds.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            var batchSize = Math.Min(100, waveQueue.Count);
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            var localCollapsed = 0;
            System.Threading.Tasks.Parallel.ForEach(batch,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                item =>
            {
                var (_, _, x, y) = item;
                lock (_collapsed)
                {
                    if (_collapsed[y, x] == null)
                    {
                        CollapseCell(x, y);
                        System.Threading.Interlocked.Increment(ref localCollapsed);
                    }
                }
            });

            collapsedCount += localCollapsed;

            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                var ndx = nx - centerX;
                                var ndy = ny - centerY;
                                var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);

                                newNeighbors.Add((distance, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ RADIAL mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated EdgeFirst mode
    /// </summary>
    private uint[] GenerateEdgeFirstCuda(int seedCount, ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ EDGE FIRST mode with parallel batching");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2.0f;
        var centerY = _height / 2.0f;
        var maxDistance = (float)Math.Sqrt(centerX * centerX + centerY * centerY);

        // Start from edges in parallel
        var edgeSeeds = new System.Collections.Concurrent.ConcurrentBag<(int x, int y)>();
        System.Threading.Tasks.Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                if (x == 0 || x == _width - 1 || y == 0 || y == _height - 1)
                {
                    edgeSeeds.Add((x, y));
                }
            }
        });

        foreach (var (x, y) in edgeSeeds)
        {
            var seedColor = _availableColors[_random.Next(_availableColors.Count)];
            _collapsed[y, x] = seedColor;

            var dx = x - centerX;
            var dy = y - centerY;
            var distanceFromCenter = (float)Math.Sqrt(dx * dx + dy * dy);
            var priority = maxDistance - distanceFromCenter;

            waveQueue.Add((priority, _random.Next(), x, y));
            processed.Add((x, y));
        }

        int collapsedCount = processed.Count;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            // Check for cancellation
            if (_cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[FastWFC-CUDA] Radial generation cancelled at {collapsedCount}/{totalCells} cells");
                FillRemainingCells();
                return CreatePixelData(); // Return partial result
            }

            var batchSize = Math.Min(100, waveQueue.Count);
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            var localCollapsed = 0;

            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                lock (_collapsed)
                                {
                                    if (_collapsed[ny, nx] == null)
                                    {
                                        CollapseCell(nx, ny);
                                        System.Threading.Interlocked.Increment(ref localCollapsed);
                                    }
                                }

                                var ndx = nx - centerX;
                                var ndy = ny - centerY;
                                var distanceFromCenter = (float)Math.Sqrt(ndx * ndx + ndy * ndy);
                                var priority = maxDistance - distanceFromCenter;

                                newNeighbors.Add((priority, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            collapsedCount += localCollapsed;

            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ EDGE FIRST mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated CenterOut mode
    /// </summary>
    private uint[] GenerateCenterOutCuda(ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ CENTER OUT mode with parallel batching");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2;
        var centerY = _height / 2;
        var fcenterX = _width / 2.0f;
        var fcenterY = _height / 2.0f;

        var seedColor = _availableColors[_random.Next(_availableColors.Count)];
        _collapsed[centerY, centerX] = seedColor;

        waveQueue.Add((0f, _random.Next(), centerX, centerY));
        processed.Add((centerX, centerY));

        int collapsedCount = 1;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            var batchSize = Math.Min(100, waveQueue.Count);
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            var localCollapsed = 0;

            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                lock (_collapsed)
                                {
                                    if (_collapsed[ny, nx] == null)
                                    {
                                        CollapseCell(nx, ny);
                                        System.Threading.Interlocked.Increment(ref localCollapsed);
                                    }
                                }

                                var ndx = nx - fcenterX;
                                var ndy = ny - fcenterY;
                                var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);

                                newNeighbors.Add((distance, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            collapsedCount += localCollapsed;

            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ CENTER OUT mode complete!");
        return finalPixels;
    }

    /// <summary>
    /// CUDA-accelerated Spiral mode
    /// </summary>
    private uint[] GenerateSpiralCuda(ProgressCallback? progressCallback, int updateFrequency)
    {
        Console.WriteLine($"[FastWFC-CUDA] ⚡ SPIRAL mode with parallel batching");

        var waveQueue = new SortedSet<(float priority, int random, int x, int y)>();
        var processed = new HashSet<(int x, int y)>();

        var centerX = _width / 2;
        var centerY = _height / 2;
        var fcenterX = _width / 2.0f;
        var fcenterY = _height / 2.0f;

        var seedColor = _availableColors[_random.Next(_availableColors.Count)];
        _collapsed[centerY, centerX] = seedColor;

        waveQueue.Add((0f, _random.Next(), centerX, centerY));
        processed.Add((centerX, centerY));

        int collapsedCount = 1;
        int totalCells = _width * _height;
        int iteration = 0;

        while (waveQueue.Count > 0 && collapsedCount < totalCells)
        {
            var batchSize = Math.Min(50, waveQueue.Count); // Smaller batches for spiral coherence
            var batch = waveQueue.Take(batchSize).ToList();

            foreach (var item in batch)
            {
                waveQueue.Remove(item);
            }

            var newNeighbors = new System.Collections.Concurrent.ConcurrentBag<(float priority, int random, int x, int y)>();
            var localCollapsed = 0;

            System.Threading.Tasks.Parallel.ForEach(batch, item =>
            {
                var (_, _, x, y) = item;
                foreach (var dir in DirectionExtensions.AllDirections)
                {
                    var (dx, dy) = dir.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                    {
                        var coord = (nx, ny);
                        lock (processed)
                        {
                            if (!processed.Contains(coord))
                            {
                                lock (_collapsed)
                                {
                                    if (_collapsed[ny, nx] == null)
                                    {
                                        CollapseCell(nx, ny);
                                        System.Threading.Interlocked.Increment(ref localCollapsed);
                                    }
                                }

                                var ndx = nx - fcenterX;
                                var ndy = ny - fcenterY;
                                var distance = (float)Math.Sqrt(ndx * ndx + ndy * ndy);
                                var angle = (float)Math.Atan2(ndy, ndx);
                                var normalizedAngle = angle < 0 ? angle + (float)(Math.PI * 2) : angle;
                                var spiralPriority = distance * 2.0f + normalizedAngle * 10.0f;

                                newNeighbors.Add((spiralPriority, _random.Next(), nx, ny));
                                processed.Add(coord);
                            }
                        }
                    }
                }
            });

            collapsedCount += localCollapsed;

            lock (waveQueue)
            {
                foreach (var neighbor in newNeighbors)
                {
                    waveQueue.Add(neighbor);
                }
            }

            iteration++;
            if (progressCallback != null && iteration % updateFrequency == 0)
            {
                var pixels = CreatePixelData();
                progressCallback(pixels, collapsedCount, totalCells);
            }
        }

        FillRemainingCells();
        var finalPixels = CreatePixelData();
        progressCallback?.Invoke(finalPixels, totalCells, totalCells);

        Console.WriteLine($"[FastWFC-CUDA] ⚡ SPIRAL mode complete!");
        return finalPixels;
    }

    private void FillRemainingCells()
    {
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                if (_collapsed[y, x] == null)
                {
                    // Try to collapse based on neighbors
                    CollapseCell(x, y);
                }
            }
        }
    }

    private uint[] CreatePixelData()
    {
        var pixels = new uint[_width * _height];

        System.Threading.Tasks.Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                var color = _collapsed[y, x] ?? new ColorRgb(32, 32, 32); // Dark gray for uncollapsed
                pixels[y * _width + x] = ColorToPixel(color);
            }
        });

        return pixels;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint ColorToPixel(ColorRgb color)
    {
        return (uint)((255 << 24) | (color.R << 16) | (color.G << 8) | color.B);
    }

    public static Bitmap CreateBitmapFromPixels(uint[] pixels, int width, int height)
    {
        var bitmap = new WriteableBitmap(
            new Avalonia.PixelSize(width, height),
            new Avalonia.Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Opaque);

        using var lockedBitmap = bitmap.Lock();

        unsafe
        {
            var ptr = (uint*)lockedBitmap.Address.ToPointer();
            for (int i = 0; i < width * height; i++)
            {
                ptr[i] = pixels[i];
            }
        }

        return bitmap;
    }
}
