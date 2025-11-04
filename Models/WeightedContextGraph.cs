using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;

namespace NNImage.Models;

/// <summary>
/// Context-aware weighted graph that considers neighborhood patterns with spatial location tracking
/// Much more powerful than simple adjacency - tracks where colors appear in images
/// </summary>
public class WeightedContextGraph
{
    // pattern -> direction -> target_color -> weight
    private readonly ConcurrentDictionary<NeighborhoodPattern, Dictionary<Direction, Dictionary<ColorRgb, double>>> _contextPatterns = new();

    // Fallback: Simple adjacency for when we don't have enough context
    private readonly ConcurrentDictionary<ColorRgb, ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>> _simpleGraph = new();

    // Spatial index: center_color -> list of patterns with that center
    private readonly ConcurrentDictionary<ColorRgb, List<NeighborhoodPattern>> _patternIndex = new();

    // Spatial location tracking: color -> list of (x, y) positions where it appears
    private readonly ConcurrentDictionary<ColorRgb, List<(float x, float y)>> _colorPositions = new();

    // Color co-occurrence tracking: color -> nearby_color -> count (tracks which colors appear near each other)
    private readonly ConcurrentDictionary<ColorRgb, ConcurrentDictionary<ColorRgb, int>> _colorCooccurrence = new();

    // Cache for weighted neighbor lookups
    private readonly ConcurrentDictionary<(NeighborhoodPattern, Direction), List<(ColorRgb, double)>> _weightedNeighborCache = new();

    // PRECOMPUTED similarity index: pattern -> list of (similar_pattern, similarity_score)
    // This is computed ONCE during training, not during generation
    private readonly ConcurrentDictionary<NeighborhoodPattern, List<(NeighborhoodPattern pattern, double similarity)>> _precomputedSimilarities = new();

    private readonly object _normalizeLock = new object();
    private bool _isNormalized;
    private bool _similaritiesPrecomputed = false;
    private Services.GpuAccelerator? _gpu;

    // Ultra-fast mode: skip similarity searches during generation and use simple adjacency fallback
    // Default: enabled unless NNIMAGE_FAST is explicitly set to 0/false
    private static readonly bool UltraFast = !string.Equals(Environment.GetEnvironmentVariable("NNIMAGE_FAST"), "0", StringComparison.OrdinalIgnoreCase)
                                            && !string.Equals(Environment.GetEnvironmentVariable("NNIMAGE_FAST"), "false", StringComparison.OrdinalIgnoreCase);

    private List<(ColorRgb color, double weight)> GetSimpleAdjacencyDistribution(ColorRgb center, Direction direction)
    {
        if (_simpleGraph.TryGetValue(center, out var dirDict) && dirDict.TryGetValue(direction, out var counts))
        {
            double total = 0;
            foreach (var v in counts.Values) total += v;
            if (total > 0)
            {
                var list = new List<(ColorRgb color, double weight)>(counts.Count);
                foreach (var kv in counts)
                    list.Add((kv.Key, kv.Value / total));
                return list;
            }
        }
        return new List<(ColorRgb color, double weight)>();
    }

    public void SetGpuAccelerator(Services.GpuAccelerator? gpu)
    {
        _gpu = gpu;
        Console.WriteLine($"[WeightedContextGraph] GPU acceleration: {(gpu?.IsAvailable == true ? "ENABLED" : "DISABLED")}");
    }

    public void AddPattern(NeighborhoodPattern pattern, Direction outputDirection, ColorRgb targetColor, double weight = 1.0)
    {
        var patternDict = _contextPatterns.GetOrAdd(pattern, _ => 
        {
            // Add to spatial index when creating new pattern
            var centerColor = pattern.Center;
            var indexList = _patternIndex.GetOrAdd(centerColor, _ => new List<NeighborhoodPattern>());
            lock (indexList)
            {
                indexList.Add(pattern);
            }

            // Note: Position tracking now handled by GraphNode in FastContextGraph
            // WeightedContextGraph retained for legacy compatibility only

            // Track color co-occurrence with neighbors
            var cooccurDict = _colorCooccurrence.GetOrAdd(centerColor, _ => new ConcurrentDictionary<ColorRgb, int>());
            foreach (var neighbor in pattern.Neighbors.Values)
            {
                if (neighbor.HasValue)
                {
                    cooccurDict.AddOrUpdate(neighbor.Value, 1, (_, count) => count + 1);
                }
            }

            return new Dictionary<Direction, Dictionary<ColorRgb, double>>();
        });

        lock (patternDict)
        {
            if (!patternDict.ContainsKey(outputDirection))
                patternDict[outputDirection] = new Dictionary<ColorRgb, double>();

            if (!patternDict[outputDirection].ContainsKey(targetColor))
                patternDict[outputDirection][targetColor] = 0;

            patternDict[outputDirection][targetColor] += weight;
        }

        _isNormalized = false;
    }

    public void AddSimpleAdjacency(ColorRgb centerColor, Direction direction, ColorRgb neighborColor)
    {
        var colorDict = _simpleGraph.GetOrAdd(centerColor, _ => new ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>());
        var directionDict = colorDict.GetOrAdd(direction, _ => new ConcurrentDictionary<ColorRgb, int>());
        directionDict.AddOrUpdate(neighborColor, 1, (_, count) => count + 1);

        _isNormalized = false;
    }

    public void Normalize()
    {
        lock (_normalizeLock)
        {
            if (_isNormalized)
                return;

            Console.WriteLine("[WeightedContextGraph] Normalizing weights...");

            // Clear cache since we're changing weights
            _weightedNeighborCache.Clear();

            // Normalize context patterns in parallel
            System.Threading.Tasks.Parallel.ForEach(_contextPatterns.Keys.ToList(), pattern =>
            {
                var directions = _contextPatterns[pattern];
                foreach (var dir in directions.Keys.ToList())
                {
                    var colors = directions[dir];
                    var total = colors.Values.Sum();

                    if (total > 0)
                    {
                        lock (colors)
                        {
                            foreach (var color in colors.Keys.ToList())
                            {
                                colors[color] /= total;
                            }
                        }
                    }
                }
            });

            _isNormalized = true;
            Console.WriteLine($"[WeightedContextGraph] Normalized {_contextPatterns.Count} context patterns");
            Console.WriteLine($"[WeightedContextGraph] Spatial index contains {_patternIndex.Count} color groups");

            // Fast training: skip heavy precompute by default unless explicitly enabled
            var precomputeEnv = Environment.GetEnvironmentVariable("NNIMAGE_PRECOMPUTE");
            var doPrecompute = string.Equals(precomputeEnv, "1", StringComparison.OrdinalIgnoreCase) 
                               || string.Equals(precomputeEnv, "true", StringComparison.OrdinalIgnoreCase);
            if (doPrecompute)
            {
                // PRECOMPUTE pattern similarities to avoid runtime GPU calls during generation
                PrecomputePatternSimilarities();
            }
            else
            {
                Console.WriteLine("[WeightedContextGraph] Skipping pattern similarity precompute (NNIMAGE_PRECOMPUTE not set to 1/true) — fastest training mode");
                _similaritiesPrecomputed = true;
            }
        }
    }

    private void PrecomputePatternSimilarities()
    {
        if (_similaritiesPrecomputed)
            return;

        Console.WriteLine("[WeightedContextGraph] Precomputing pattern similarities for fast generation...");
        Console.WriteLine("[WeightedContextGraph] This happens ONCE during training, not during generation");

        var allPatterns = _contextPatterns.Keys.ToArray();
        var totalComparisons = 0L;

        // Group patterns by center color for efficient processing
        var colorGroups = _patternIndex.ToArray();

        Console.WriteLine($"[WeightedContextGraph] Processing {colorGroups.Length} color groups with {allPatterns.Length} total patterns");

        var processedGroups = 0;
        var startTime = DateTime.Now;

        // OPTIMIZED: Use aggressive parallelization with more threads
        var parallelOptions = new System.Threading.Tasks.ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount * 2 // Hyper-threading
        };

        // Process each color group in parallel with optimized settings
        System.Threading.Tasks.Parallel.ForEach(colorGroups, parallelOptions, colorGroup =>
        {
            var centerColor = colorGroup.Key;
            var patternsWithThisCenter = colorGroup.Value.ToArray();

            // OPTIMIZED: Prepare candidate data once for all queries in this group
            var candidateCenters = patternsWithThisCenter.Select(p => p.Center).ToArray();
            var candidateNeighbors = patternsWithThisCenter.Select(p =>
            {
                var neighbors = new ColorRgb?[8];
                for (int i = 0; i < 8; i++)
                {
                    neighbors[i] = p.Neighbors.GetValueOrDefault((Direction)i);
                }
                return neighbors;
            }).ToArray();

            // OPTIMIZED: Use GPU batching for massive speedup
            var useGpu = _gpu?.IsAvailable == true && patternsWithThisCenter.Length > 50;
            var validPatterns = patternsWithThisCenter.Where(p => _contextPatterns.ContainsKey(p)).ToArray();

            if (useGpu && validPatterns.Length > 10)
            {
                // BATCHED GPU PROCESSING - Calculate all similarities at once
                try
                {
                    var batchResults = _gpu.CalculateBatchedPatternSimilarities(
                        validPatterns, candidateCenters, candidateNeighbors);

                    if (batchResults != null)
                    {
                        // Process results in parallel
                        System.Threading.Tasks.Parallel.For(0, (int)validPatterns.Length, queryIdx =>
                        {
                            var queryPattern = validPatterns[queryIdx];
                            var similarities = batchResults[queryIdx];

                            var similarPatterns = new List<(NeighborhoodPattern pattern, double similarity)>(10);
                            for (int i = 0; i < patternsWithThisCenter.Length; i++)
                            {
                                var sim = (double)similarities[i];
                                if (sim > 0.5)
                                {
                                    similarPatterns.Add((patternsWithThisCenter[i], sim));
                                }
                            }

                            // Sort and take top 10
                            similarPatterns.Sort((a, b) => b.similarity.CompareTo(a.similarity));
                            if (similarPatterns.Count > 10)
                            {
                                similarPatterns.RemoveRange(10, similarPatterns.Count - 10);
                            }

                            _precomputedSimilarities[queryPattern] = similarPatterns;
                        });

                        System.Threading.Interlocked.Add(ref totalComparisons, 
                            (long)validPatterns.Length * patternsWithThisCenter.Length);
                    }
                    else
                    {
                        // GPU failed, fall back to parallel CPU
                        ProcessPatternsParallelCpu(validPatterns, patternsWithThisCenter, ref totalComparisons);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WeightedContextGraph] Batched GPU failed: {ex.Message}, using parallel CPU");
                    ProcessPatternsParallelCpu(validPatterns, patternsWithThisCenter, ref totalComparisons);
                }
            }
            else
            {
                // OPTIMIZED: Parallel CPU for small sets or no GPU
                ProcessPatternsParallelCpu(validPatterns, patternsWithThisCenter, ref totalComparisons);
            }

            // Progress reporting
            var completed = System.Threading.Interlocked.Increment(ref processedGroups);
            if (completed % 10 == 0 || completed == colorGroups.Length)
            {
                var elapsed = (DateTime.Now - startTime).TotalSeconds;
                var rate = completed / Math.Max(elapsed, 0.001);
                var remaining = (colorGroups.Length - completed) / Math.Max(rate, 0.001);
                var percentComplete = completed * 100.0 / colorGroups.Length;
                Console.WriteLine($"[WeightedContextGraph] Precomputed {completed}/{colorGroups.Length} color groups ({percentComplete:F1}%) - ETA: {remaining:F0}s");
            }
        });

        var totalTime = (DateTime.Now - startTime).TotalSeconds;
        Console.WriteLine($"[WeightedContextGraph] Pattern similarity precomputation complete!");
        Console.WriteLine($"[WeightedContextGraph] Processed {allPatterns.Length} patterns in {totalTime:F2}s");
        Console.WriteLine($"[WeightedContextGraph] Total comparisons: {totalComparisons:N0}");
        Console.WriteLine($"[WeightedContextGraph] Generation will now be MUCH faster (no runtime GPU calls)");

        _similaritiesPrecomputed = true;
    }

        private void ProcessPatternsParallelCpu(
            NeighborhoodPattern[] queryPatterns, 
            NeighborhoodPattern[] candidates, 
            ref long totalComparisons)
        {
            long localComparisons = 0;
            System.Threading.Tasks.Parallel.ForEach(queryPatterns, queryPattern =>
            {
                var similarPatterns = ComputeSimilaritiesCpuOptimized(queryPattern, candidates);
                _precomputedSimilarities[queryPattern] = similarPatterns;
                System.Threading.Interlocked.Add(ref localComparisons, candidates.Length);
            });
            System.Threading.Interlocked.Add(ref totalComparisons, localComparisons);
        }

        private List<(NeighborhoodPattern pattern, double similarity)> ComputeSimilaritiesCpu(
            NeighborhoodPattern queryPattern, NeighborhoodPattern[] candidates)
        {
            return ComputeSimilaritiesCpuOptimized(queryPattern, candidates);
        }

        private List<(NeighborhoodPattern pattern, double similarity)> ComputeSimilaritiesCpuOptimized(
            NeighborhoodPattern queryPattern, NeighborhoodPattern[] candidates)
        {
            // OPTIMIZED: Use thread-local storage to avoid allocations
            var results = new List<(NeighborhoodPattern pattern, double similarity)>(candidates.Length);

            // OPTIMIZED: Compute all similarities first (better for vectorization)
            for (int i = 0; i < candidates.Length; i++)
            {
                var sim = queryPattern.CalculateSimilarity(candidates[i]);
                if (sim > 0.5)
                {
                    results.Add((candidates[i], sim));
                }
            }

            // OPTIMIZED: Use partial sort - only sort top 10
            if (results.Count <= 10)
            {
                results.Sort((a, b) => b.similarity.CompareTo(a.similarity));
                return results;
            }

            // Partial sort for top 10 (faster than full sort)
            var top10 = new List<(NeighborhoodPattern pattern, double similarity)>(10);
            for (int i = 0; i < results.Count; i++)
            {
                if (top10.Count < 10)
                {
                    top10.Add(results[i]);
                    if (top10.Count == 10)
                    {
                        top10.Sort((a, b) => b.similarity.CompareTo(a.similarity));
                    }
                }
                else if (results[i].similarity > top10[9].similarity)
                {
                    top10[9] = results[i];
                    // Bubble up the new entry
                    for (int j = 8; j >= 0 && top10[j + 1].similarity > top10[j].similarity; j--)
                    {
                        (top10[j], top10[j + 1]) = (top10[j + 1], top10[j]);
                    }
                }
            }

            return top10;
    }

    public List<(ColorRgb color, double weight)> GetWeightedNeighbors(NeighborhoodPattern currentPattern, Direction direction)
    {
        if (!_isNormalized)
            Normalize();

        // OPTIMIZED: Fast path - check cache first with TryGetValue (single lookup)
        var cacheKey = (currentPattern, direction);
        if (_weightedNeighborCache.TryGetValue(cacheKey, out var cached))
            return cached;

        // Prepare result collector for fallback paths
        var results = new List<(ColorRgb color, double weight)>();

        // Try exact pattern match first (fastest path)
        if (_contextPatterns.TryGetValue(currentPattern, out var patternDirs) &&
            patternDirs.TryGetValue(direction, out var colors))
        {
            // Avoid LINQ, use direct iteration
            var exact = new List<(ColorRgb color, double weight)>(colors.Count);
            foreach (var kvp in colors)
            {
                // Apply spatial position boost based on color co-occurrence
                var weight = kvp.Value;
                if (_colorCooccurrence.TryGetValue(currentPattern.Center, out var cooccur) &&
                    cooccur.TryGetValue(kvp.Key, out var cooccurCount))
                {
                    // Boost colors that frequently appear together (up to 20% boost)
                    var boost = Math.Min(cooccurCount / 100.0, 0.2);
                    weight *= (1.0 + boost);
                }
                exact.Add((kvp.Key, weight));
            }

            // Re-normalize after spatial boosting
            var total = exact.Sum(x => x.weight);
            if (total > 0)
            {
                for (int i = 0; i < exact.Count; i++)
                {
                    exact[i] = (exact[i].color, exact[i].weight / total);
                }
            }

            _weightedNeighborCache[cacheKey] = exact;
            return exact;
        }

        // Ultra-fast: skip similarity computations and use simple adjacency distribution
        if (UltraFast)
        {
            var simple = GetSimpleAdjacencyDistribution(currentPattern.Center, direction);
            _weightedNeighborCache[cacheKey] = simple;
            return simple;
        }

        // Use spatial index to only check patterns with matching center color
        if (_patternIndex.TryGetValue(currentPattern.Center, out var candidatePatterns))
        {
            List<(NeighborhoodPattern pattern, double similarity)> similarPatterns;

            // Try GPU acceleration for large pattern sets (only when not UltraFast)
            if (_gpu?.IsAvailable == true && candidatePatterns.Count > 100)
            {
                try
                {
                    // Prepare data for GPU
                    var queryNeighbors = new ColorRgb?[8];
                    for (int i = 0; i < 8; i++)
                    {
                        queryNeighbors[i] = currentPattern.Neighbors.GetValueOrDefault((Direction)i);
                    }

                    var candidateCenters = candidatePatterns.Select(p => p.Center).ToArray();
                    var candidateNeighbors = candidatePatterns.Select(p =>
                    {
                        var neighbors = new ColorRgb?[8];
                        for (int i = 0; i < 8; i++)
                        {
                            neighbors[i] = p.Neighbors.GetValueOrDefault((Direction)i);
                        }
                        return neighbors;
                    }).ToArray();

                    var similarities = _gpu.CalculatePatternSimilarities(
                        currentPattern.Center, queryNeighbors,
                        candidateCenters, candidateNeighbors);

                    if (similarities != null)
                    {
                        similarPatterns = candidatePatterns
                            .Zip(similarities, (p, s) => (pattern: p, similarity: (double)s))
                            .Where(x => x.similarity > 0.5)
                            .OrderByDescending(x => x.similarity)
                            .Take(5)
                            .ToList();
                    }
                    else
                    {
                        // GPU failed, fallback to CPU
                        similarPatterns = candidatePatterns
                            .AsParallel()
                            .Select(p => (pattern: p, similarity: currentPattern.CalculateSimilarity(p)))
                            .Where(x => x.similarity > 0.5)
                            .OrderByDescending(x => x.similarity)
                            .Take(5)
                            .ToList();
                    }
                }
                catch
                {
                    // GPU failed, fallback to CPU
                    similarPatterns = candidatePatterns
                        .AsParallel()
                        .Select(p => (pattern: p, similarity: currentPattern.CalculateSimilarity(p)))
                        .Where(x => x.similarity > 0.5)
                        .OrderByDescending(x => x.similarity)
                        .Take(5)
                        .ToList();
                }
            }
            else
            {
                // CPU path for small pattern sets
                similarPatterns = candidatePatterns
                    .AsParallel()
                    .Select(p => (pattern: p, similarity: currentPattern.CalculateSimilarity(p)))
                    .Where(x => x.similarity > 0.5)
                    .OrderByDescending(x => x.similarity)
                    .Take(5)
                    .ToList();
            }

            if (similarPatterns.Any())
            {
                var weightedColors = new Dictionary<ColorRgb, double>();

                foreach (var (pattern, similarity) in similarPatterns)
                {
                    if (_contextPatterns[pattern].TryGetValue(direction, out var patternColors))
                    {
                        foreach (var (color, weight) in patternColors)
                        {
                            if (!weightedColors.ContainsKey(color))
                                weightedColors[color] = 0;

                            weightedColors[color] += weight * similarity;
                        }
                    }
                }

                // Normalize combined weights
                var total = weightedColors.Values.Sum();
                if (total > 0)
                {
                    results.AddRange(weightedColors.Select(kvp => (kvp.Key, kvp.Value / total)));
                }
            }
        }

        // Fallback to simple adjacency if no context available
        if (results.Count == 0 && _simpleGraph.TryGetValue(currentPattern.Center, out var centerDict) &&
            centerDict.TryGetValue(direction, out var neighbors))
        {
            var total = neighbors.Values.Sum();
            results.AddRange(neighbors.Select(kvp => (kvp.Key, (double)kvp.Value / total)));
        }

        // Cache the result
        _weightedNeighborCache[cacheKey] = results;
        return results;
    }

    public List<ColorRgb> GetPossibleNeighbors(NeighborhoodPattern pattern, Direction direction)
    {
        var weighted = GetWeightedNeighbors(pattern, direction);
        return weighted.Select(x => x.color).ToList();
    }

    public List<ColorRgb> GetAllColors()
    {
        var colors = new HashSet<ColorRgb>();

        foreach (var pattern in _contextPatterns.Keys)
        {
            colors.Add(pattern.Center);
            foreach (var neighbor in pattern.Neighbors.Values)
            {
                if (neighbor.HasValue)
                    colors.Add(neighbor.Value);
            }
        }

        foreach (var color in _simpleGraph.Keys)
        {
            colors.Add(color);
        }

        return colors.ToList();
    }

    public int GetColorCount()
    {
        return GetAllColors().Count;
    }

    public int GetPatternCount()
    {
        return _contextPatterns.Count;
    }

    /// <summary>
    /// Get the spatial distribution of where a color appears in images
    /// Returns list of (x, y) normalized positions
    /// </summary>
    public List<(float x, float y)> GetColorPositions(ColorRgb color)
    {
        if (_colorPositions.TryGetValue(color, out var positions))
        {
            return new List<(float x, float y)>(positions);
        }
        return new List<(float x, float y)>();
    }

    /// <summary>
    /// Get colors that frequently appear near the given color
    /// Returns list of (color, count) ordered by frequency
    /// </summary>
    public List<(ColorRgb color, int count)> GetNearbyColors(ColorRgb color)
    {
        if (_colorCooccurrence.TryGetValue(color, out var cooccur))
        {
            return cooccur.OrderByDescending(kvp => kvp.Value)
                         .Select(kvp => (kvp.Key, kvp.Value))
                         .ToList();
        }
        return new List<(ColorRgb color, int count)>();
    }

    // --- Persistence mapping ---
    public WeightedContextGraphSnapshot ToSnapshot()
    {
        var snap = new WeightedContextGraphSnapshot();

        // Patterns with normalized direction weights (already normalized in _contextPatterns)
        foreach (var patternKvp in _contextPatterns)
        {
            var pattern = patternKvp.Key;
            var directions = patternKvp.Value;

            var entry = new PatternEntry
            {
                Center = pattern.Center,
                Neighbors = new ColorRgb?[8]
            };

            // neighbors in fixed order 0..7
            for (int i = 0; i < 8; i++)
            {
                entry.Neighbors[i] = pattern.Neighbors.GetValueOrDefault((Direction)i);
            }

            foreach (var dir in directions.Keys)
            {
                var colors = directions[dir];
                var dirEntry = new DirectionWeightsEntry
                {
                    Direction = dir,
                    Colors = colors.Select(kv => new ColorWeight { Color = kv.Key, Weight = kv.Value }).ToList()
                };
                entry.DirectionWeights.Add(dirEntry);
            }

            snap.Patterns.Add(entry);
        }

        // Simple adjacency as normalized weights
        foreach (var center in _simpleGraph.Keys)
        {
            var dirDict = _simpleGraph[center];
            foreach (var dir in dirDict.Keys)
            {
                var counts = dirDict[dir];
                double total = 0;
                foreach (var v in counts.Values) total += v;
                if (total <= 0) continue;

                var adjEntry = new SimpleAdjacencyEntry
                {
                    Center = center,
                    Direction = dir,
                    Colors = counts.Select(kv => new ColorWeight { Color = kv.Key, Weight = kv.Value / total }).ToList()
                };
                snap.Adjacency.Add(adjEntry);
            }
        }

        snap.Colors = GetAllColors();
        return snap;
    }

    public static WeightedContextGraph FromSnapshot(WeightedContextGraphSnapshot snap)
    {
        var graph = new WeightedContextGraph();

        // Rebuild patterns with weights
        foreach (var p in snap.Patterns)
        {
            var neighbors = new Dictionary<Direction, ColorRgb?>();
            for (int i = 0; i < 8; i++)
            {
                neighbors[(Direction)i] = p.Neighbors.Length > i ? p.Neighbors[i] : null;
            }
            var pattern = new NeighborhoodPattern(p.Center, neighbors);

            foreach (var dirEntry in p.DirectionWeights)
            {
                foreach (var cw in dirEntry.Colors)
                {
                    graph.AddPattern(pattern, dirEntry.Direction, cw.Color, cw.Weight);
                }
            }
        }

        // Rebuild simple adjacency approximately from probabilities
        foreach (var adj in snap.Adjacency)
        {
            foreach (var cw in adj.Colors)
            {
                // Convert probability to approximate counts for stability
                var repeats = Math.Clamp((int)Math.Round(cw.Weight * 100), 1, 100);
                for (int i = 0; i < repeats; i++)
                    graph.AddSimpleAdjacency(adj.Center, adj.Direction, cw.Color);
            }
        }

        // Normalize to finalize weights and caches
        graph.Normalize();
        return graph;
    }
}
