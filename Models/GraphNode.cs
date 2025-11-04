using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NNImage.Models;

/// <summary>
/// Ultra-fast graph node for super fast training
/// Stores color, position, and weighted edges to other nodes
/// </summary>
public class GraphNode
{
    public ColorRgb Color { get; }
    public float NormalizedX { get; }
    public float NormalizedY { get; }

    // Weighted edges to neighbor nodes in each direction
    // Direction -> (target_node, weight)
    public Dictionary<Direction, List<(GraphNode node, float weight)>> Edges { get; }

    // Fast lookup: which nodes connect TO this node
    public List<GraphNode> IncomingNodes { get; }

    // Observation count for this node (field for Interlocked operations)
    public int ObservationCount;

    public GraphNode(ColorRgb color, float normalizedX, float normalizedY)
    {
        Color = color;
        NormalizedX = normalizedX;
        NormalizedY = normalizedY;
        Edges = new Dictionary<Direction, List<(GraphNode, float)>>(8);
        IncomingNodes = new List<GraphNode>();
        ObservationCount = 1;
    }

    /// <summary>
    /// Add or update weighted edge to another node
    /// Thread-safe with edge-level locking
    /// Weight accumulates with each observation - more frequent = higher weight!
    /// </summary>
    public void AddEdge(Direction direction, GraphNode targetNode, float weight = 1.0f)
    {
        lock (Edges) // Lock on edges dictionary for thread-safety
        {
            if (!Edges.ContainsKey(direction))
            {
                Edges[direction] = new List<(GraphNode, float)>(8); // Pre-allocate for common case
            }

            var edges = Edges[direction];

            // Check if edge already exists - accumulate weight for frequency learning
            for (int i = 0; i < edges.Count; i++)
            {
                if (ReferenceEquals(edges[i].node, targetNode))
                {
                    // ACCUMULATE weight - the more we see this connection, the stronger it gets!
                    edges[i] = (targetNode, edges[i].weight + weight);
                    return;
                }
            }

            // Add new edge
            edges.Add((targetNode, weight));
        }

        // Track incoming connection (only once per unique edge)
        lock (targetNode.IncomingNodes)
        {
            if (!targetNode.IncomingNodes.Contains(this))
            {
                targetNode.IncomingNodes.Add(this);
            }
        }
    }

    /// <summary>
    /// Get weighted neighbor predictions for a direction
    /// Returns normalized probability distribution
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public List<(ColorRgb color, double weight)> GetWeightedNeighbors(Direction direction)
    {
        if (!Edges.TryGetValue(direction, out var edges) || edges.Count == 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Calculate total weight
        float totalWeight = 0;
        for (int i = 0; i < edges.Count; i++)
        {
            totalWeight += edges[i].weight;
        }

        if (totalWeight <= 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Normalize and return
        var result = new List<(ColorRgb, double)>(edges.Count);
        for (int i = 0; i < edges.Count; i++)
        {
            var (node, weight) = edges[i];
            result.Add((node.Color, weight / totalWeight));
        }

        return result;
    }

    /// <summary>
    /// Calculate spatial distance to another node
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float SpatialDistance(GraphNode other)
    {
        var dx = NormalizedX - other.NormalizedX;
        var dy = NormalizedY - other.NormalizedY;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>
    /// Get all connected nodes across all directions with their total weights
    /// Useful for learning which colors frequently appear together
    /// </summary>
    public List<(GraphNode node, float totalWeight)> GetAllConnectedNodes()
    {
        var nodeWeights = new Dictionary<GraphNode, float>();

        foreach (var edgeList in Edges.Values)
        {
            foreach (var (node, weight) in edgeList)
            {
                if (!nodeWeights.ContainsKey(node))
                {
                    nodeWeights[node] = 0;
                }
                nodeWeights[node] += weight;
            }
        }

        return nodeWeights
            .OrderByDescending(kvp => kvp.Value)
            .Select(kvp => (kvp.Key, kvp.Value))
            .ToList();
    }

    /// <summary>
    /// Get strongest edge weight in a direction (most frequent co-occurrence)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float GetStrongestEdgeWeight(Direction direction)
    {
        if (Edges.TryGetValue(direction, out var edges) && edges.Count > 0)
        {
            float maxWeight = 0;
            for (int i = 0; i < edges.Count; i++)
            {
                if (edges[i].weight > maxWeight)
                {
                    maxWeight = edges[i].weight;
                }
            }
            return maxWeight;
        }
        return 0;
    }

    public override string ToString()
    {
        var totalEdges = Edges.Values.Sum(e => e.Count);
        var totalWeight = Edges.Values.SelectMany(e => e).Sum(e => e.weight);
        return $"Node[{Color} @ ({NormalizedX:F2}, {NormalizedY:F2}), Edges: {totalEdges}, TotalWeight: {totalWeight:F1}]";
    }
}
