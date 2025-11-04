using System;
using System.Collections.Generic;
using System.Linq;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// High-performance WFC using fixed-size bitsets for palette sizes up to 256.
/// This keeps the visual behavior equivalent to the classic implementation while
/// greatly accelerating propagation and entropy evaluation. Intended as a drop-in
/// CPU acceleration step before full GPU offload.
/// </summary>
public class BitsetWaveFunctionCollapse
{
    private readonly AdjacencyGraph _adj;
    private readonly int _width;
    private readonly int _height;
    private readonly Random _random;

    // Palette and mappings
    private readonly List<ColorRgb> _palette;
    private readonly Dictionary<ColorRgb, int> _indexOf;

    // Per direction (0..7), per color index (0..P-1) → BitMask of allowed neighbor colors
    private readonly BitMask256[,] _compat; // [8, P]

    // State
    private readonly BitMask256[] _masks;   // per-cell possibility set as bit mask
    private readonly int[] _collapsed;      // -1 if not collapsed; otherwise palette index

    public delegate void ProgressCallback(uint[] pixels, int iteration, int totalIterations);

    public BitsetWaveFunctionCollapse(AdjacencyGraph adjacencyGraph, int width, int height, int? seed = null)
    {
        _adj = adjacencyGraph;
        _width = width;
        _height = height;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        _palette = _adj.GetAllColors();
        if (_palette.Count == 0)
            throw new InvalidOperationException("Adjacency graph has no colors");
        if (_palette.Count > 256)
            throw new NotSupportedException("Bitset WFC supports up to 256 colors");

        _indexOf = new Dictionary<ColorRgb, int>(_palette.Count);
        for (int i = 0; i < _palette.Count; i++) _indexOf[_palette[i]] = i;

        _compat = BuildCompatibilityMasks();

        _masks = new BitMask256[_width * _height];
        var full = BitMask256.Full(_palette.Count);
        for (int i = 0; i < _masks.Length; i++) _masks[i] = full;

        _collapsed = Enumerable.Repeat(-1, _width * _height).ToArray();
    }

    private BitMask256[,] BuildCompatibilityMasks()
    {
        int P = _palette.Count;
        var compat = new BitMask256[8, P];

        for (int dir = 0; dir < 8; dir++)
        {
            for (int ci = 0; ci < P; ci++)
            {
                var center = _palette[ci];
                var allowed = new BitMask256();
                // Query allowed neighbors from adjacency graph
                var neighbors = _adj.GetPossibleNeighbors(center, (Direction)dir);
                foreach (var n in neighbors)
                {
                    if (_indexOf.TryGetValue(n, out int ni))
                        allowed.SetBit(ni);
                }
                compat[dir, ci] = allowed;
            }
        }
        return compat;
    }

    private int Idx(int x, int y) => y * _width + x;

    public uint[] Generate(ProgressCallback? progress = null, int updateFrequency = 50)
    {
        int totalCells = _width * _height;
        int collapsedCount = 0;
        int iteration = 0;

        while (collapsedCount < totalCells)
        {
            var (cx, cy) = FindMinimumEntropyCell();
            if (cx < 0) break; // nothing left

            // Collapse
            int chosen = ChooseColorIndex(cx, cy);
            SetCollapsed(cx, cy, chosen);
            collapsedCount++;

            // Propagate from this collapse
            Propagate(cx, cy);

            iteration++;
            if (progress != null && iteration % updateFrequency == 0)
            {
                progress(CreatePixelData(), collapsedCount, totalCells);
            }
        }

        // Fill any remaining
        FillRemaining();
        var finalPixels = CreatePixelData();
        progress?.Invoke(finalPixels, totalCells, totalCells);
        return finalPixels;
    }

    private (int x, int y) FindMinimumEntropyCell()
    {
        int bestX = -1, bestY = -1;
        int bestEntropy = int.MaxValue;

        for (int y = 0; y < _height; y++)
        {
            int rowBase = y * _width;
            for (int x = 0; x < _width; x++)
            {
                int i = rowBase + x;
                if (_collapsed[i] >= 0) continue;
                int entropy = _masks[i].PopCount();
                if (entropy == 0) continue;
                if (entropy < bestEntropy)
                {
                    bestEntropy = entropy;
                    bestX = x; bestY = y;
                }
                else if (entropy == bestEntropy && bestEntropy != int.MaxValue)
                {
                    // tie-break randomly to keep similar variety
                    if (_random.NextDouble() < 0.5)
                    {
                        bestX = x; bestY = y;
                    }
                }
            }
        }

        return (bestX, bestY);
    }

    private int ChooseColorIndex(int x, int y)
    {
        int i = Idx(x, y);
        var mask = _masks[i];

        // Weight selection by neighbor compatibility similar to classic version
        // Iterate all set bits and accumulate weights
        var weights = new List<(int ci, double w)>();

        for (int ci = 0; ci < _palette.Count; ci++)
        {
            if (!mask.TestBit(ci)) continue;
            double w = 1.0;

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;
                int ni = Idx(nx, ny);
                int ncol = _collapsed[ni];
                if (ncol >= 0)
                {
                    // Check if neighbor color allowed
                    var allowed = _compat[(int)dir, ci];
                    if (allowed.TestBit(ncol)) w *= 2.0;
                }
            }
            weights.Add((ci, w));
        }

        if (weights.Count == 0)
        {
            // fallback random from palette
            return _random.Next(_palette.Count);
        }

        double total = 0;
        foreach (var t in weights) total += t.w;
        double r = _random.NextDouble() * total;
        double c = 0;
        foreach (var (ci, w) in weights)
        {
            c += w;
            if (r <= c) return ci;
        }
        return weights[0].ci; // fallback
    }

    private void SetCollapsed(int x, int y, int colorIndex)
    {
        int i = Idx(x, y);
        _collapsed[i] = colorIndex;
        _masks[i] = BitMask256.FromSingle(colorIndex);
    }

    private void Propagate(int startX, int startY)
    {
        var q = new Queue<(int x, int y)>();
        var seen = new HashSet<int>();
        q.Enqueue((startX, startY));
        seen.Add(Idx(startX, startY));

        while (q.Count > 0)
        {
            var (x, y) = q.Dequeue();
            int i = Idx(x, y);
            int centerColor = _collapsed[i];
            if (centerColor < 0) continue;

            foreach (var dir in DirectionExtensions.AllDirections)
            {
                var (dx, dy) = dir.GetOffset();
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;
                int ni = Idx(nx, ny);
                if (_collapsed[ni] >= 0) continue;

                // neighbor must be in set allowed by center's color in this direction
                var allowed = _compat[(int)dir, centerColor];
                var old = _masks[ni];
                var now = BitMask256.And(old, allowed);
                if (!now.Equals(old) && !now.IsZero)
                {
                    _masks[ni] = now;
                    if (seen.Add(ni)) q.Enqueue((nx, ny));
                }
            }
        }
    }

    private void FillRemaining()
    {
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                int i = Idx(x, y);
                if (_collapsed[i] >= 0) continue;
                int ci = ChooseColorIndex(x, y);
                SetCollapsed(x, y, ci);
            }
        }
    }

    private uint[] CreatePixelData()
    {
        var pixels = new uint[_width * _height];
        for (int y = 0; y < _height; y++)
        {
            int rowBase = y * _width;
            for (int x = 0; x < _width; x++)
            {
                int i = rowBase + x;
                int ci = _collapsed[i];
                if (ci < 0)
                {
                    // pick first set bit as preview
                    var m = _masks[i];
                    for (int k = 0; k < _palette.Count; k++)
                        if (m.TestBit(k)) { ci = k; break; }
                }
                var c = (ci >= 0) ? _palette[ci] : new ColorRgb(0,0,0);
                pixels[i] = c.ToUInt32();
            }
        }
        return pixels;
    }
}