using System;
using System.Collections.Generic;
using System.Linq;

namespace NNImage.Models;

/// <summary>
/// Represents a neighborhood pattern around a pixel
/// Used for context-aware pattern matching
/// Position tracking moved to GraphNode for ultra-fast training
/// </summary>
public readonly struct NeighborhoodPattern : IEquatable<NeighborhoodPattern>
{
    public ColorRgb Center { get; }
    public Dictionary<Direction, ColorRgb?> Neighbors { get; }

    // Store hash code as readonly to avoid mutation
    private readonly int _hashCode;

    public NeighborhoodPattern(ColorRgb center, Dictionary<Direction, ColorRgb?> neighbors)
    {
        Center = center;
        Neighbors = neighbors;
        _hashCode = ComputeHashCode(center, neighbors);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public bool Equals(NeighborhoodPattern other)
    {
        // Quick hash check first
        if (_hashCode != other._hashCode)
            return false;

        if (!Center.Equals(other.Center))
            return false;

        // Fast path: check if same dictionary reference (common during training)
        if (ReferenceEquals(Neighbors, other.Neighbors))
            return true;

        // Unrolled loop for 8 directions - much faster than foreach
        for (int i = 0; i < 8; i++)
        {
            var dir = (Direction)i;
            var thisColor = Neighbors.GetValueOrDefault(dir);
            var otherColor = other.Neighbors.GetValueOrDefault(dir);

            if (!Equals(thisColor, otherColor))
                return false;
        }

        return true;
    }

    public override bool Equals(object? obj)
    {
        return obj is NeighborhoodPattern other && Equals(other);
    }

    public override int GetHashCode()
    {
        return _hashCode;
    }

    private static int ComputeHashCode(ColorRgb center, Dictionary<Direction, ColorRgb?> neighbors)
    {
        // Faster hashing using direct bit manipulation
        unchecked
        {
            int hash = center.GetHashCode();

            // Unrolled for performance
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.North)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.NorthEast)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.East)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.SouthEast)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.South)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.SouthWest)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.West)?.GetHashCode() ?? 0);
            hash = (hash * 397) ^ (neighbors.GetValueOrDefault(Direction.NorthWest)?.GetHashCode() ?? 0);

            return hash;
        }
    }

    /// <summary>
    /// Calculate similarity score between this pattern and another
    /// Returns 0.0-1.0 where 1.0 is identical
    /// Optimized with early exit and unrolled loops
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public double CalculateSimilarity(NeighborhoodPattern other)
    {
        // Centers must match
        if (!Center.Equals(other.Center))
            return 0.0;

        // Fast path for identical patterns
        if (Equals(other))
            return 1.0;

        int matches = 0;
        int total = 0;

        // Unrolled for performance - check all 8 directions
        for (int i = 0; i < 8; i++)
        {
            var dir = (Direction)i;
            var thisColor = Neighbors.GetValueOrDefault(dir);
            var otherColor = other.Neighbors.GetValueOrDefault(dir);

            if (thisColor.HasValue && otherColor.HasValue)
            {
                total++;
                if (thisColor.Value.Equals(otherColor.Value))
                {
                    matches++;
                }
                else if (total - matches > 4) // Early exit if we can't reach 0.5
                {
                    return 0.0;
                }
            }
        }

        return total > 0 ? (double)matches / total : 0.0;
    }

    public override string ToString()
    {
        return $"Pattern[Center={Center}, Neighbors={Neighbors.Count}]";
    }
}
