using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;

namespace NNImage.Models;

public class AdjacencyGraph
{
    // adjacency_graph[color][direction][neighbor_color] = count
    private readonly ConcurrentDictionary<ColorRgb, ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>> _graph = new();
    private readonly Dictionary<ColorRgb, Dictionary<Direction, Dictionary<ColorRgb, double>>> _probabilities = new();
    private readonly object _normalizeLock = new object();
    private bool _isNormalized;

    public void AddAdjacency(ColorRgb centerColor, Direction direction, ColorRgb neighborColor)
    {
        var colorDict = _graph.GetOrAdd(centerColor, _ => new ConcurrentDictionary<Direction, ConcurrentDictionary<ColorRgb, int>>());
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

            Console.WriteLine("[AdjacencyGraph] Starting normalization...");
            _probabilities.Clear();

            System.Threading.Tasks.Parallel.ForEach(_graph, colorPair =>
            {
                var (color, directions) = (colorPair.Key, colorPair.Value);
                var colorProbs = new Dictionary<Direction, Dictionary<ColorRgb, double>>();

                foreach (var (direction, neighbors) in directions)
                {
                    var total = neighbors.Values.Sum();
                    colorProbs[direction] = new Dictionary<ColorRgb, double>();

                    foreach (var (neighborColor, count) in neighbors)
                    {
                        colorProbs[direction][neighborColor] = (double)count / total;
                    }
                }

                lock (_probabilities)
                {
                    _probabilities[color] = colorProbs;
                }
            });

            _isNormalized = true;
            Console.WriteLine("[AdjacencyGraph] Normalization complete");
        }
    }

    public ColorRgb? GetRandomNeighbor(ColorRgb centerColor, Direction direction, Random random)
    {
        if (!_isNormalized)
            Normalize();

        if (!_probabilities.ContainsKey(centerColor) || 
            !_probabilities[centerColor].ContainsKey(direction))
            return null;

        var neighbors = _probabilities[centerColor][direction];
        if (neighbors.Count == 0)
            return null;

        var rand = random.NextDouble();
        var cumulative = 0.0;

        foreach (var (neighborColor, probability) in neighbors)
        {
            cumulative += probability;
            if (rand <= cumulative)
                return neighborColor;
        }

        // Fallback to last color
        return neighbors.Keys.Last();
    }

    public List<ColorRgb> GetPossibleNeighbors(ColorRgb centerColor, Direction direction)
    {
        if (!_graph.ContainsKey(centerColor) || 
            !_graph[centerColor].ContainsKey(direction))
            return new List<ColorRgb>();

        return _graph[centerColor][direction].Keys.ToList();
    }

    public List<ColorRgb> GetAllColors()
    {
        return _graph.Keys.ToList();
    }

    public int GetColorCount()
    {
        return _graph.Keys.Count;
    }

    public bool HasColor(ColorRgb color)
    {
        return _graph.ContainsKey(color);
    }
}
