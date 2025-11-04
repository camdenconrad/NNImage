using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using NNImage.Services;

namespace NNImage.Models;

// DTOs for persisting the trained model (single-scale 9x9 at the moment)
public class ModelSnapshot
{
    [JsonPropertyName("version")] public int Version { get; set; } = 2; // v2 = ultra-fast node-based graph
    [JsonPropertyName("node_count")] public int NodeCount { get; set; }
    [JsonPropertyName("edge_count")] public int EdgeCount { get; set; }
    [JsonPropertyName("created")] public DateTime Created { get; set; } = DateTime.Now;
    [JsonPropertyName("notes")] public string? Notes { get; set; }

    // Metadata for training session
    [JsonPropertyName("quantization_level")] public int QuantizationLevel { get; set; }
    [JsonPropertyName("total_images")] public int TotalImages { get; set; }
    [JsonPropertyName("first_training_date")] public DateTime? FirstTrainingDate { get; set; }
    [JsonPropertyName("last_training_date")] public DateTime? LastTrainingDate { get; set; }
    [JsonPropertyName("training_image_paths")] public List<string> TrainingImagePaths { get; set; } = new();

    // v2 fast-graph data (preferred)
    [JsonPropertyName("nodes")] public List<FastNodeDto>? Nodes { get; set; }
    [JsonPropertyName("edges")] public List<FastEdgeDto>? Edges { get; set; }

    // Legacy fields for backward compatibility
    [JsonPropertyName("scales")] public int[]? Scales { get; set; }
    [JsonPropertyName("graphs")] public Dictionary<int, WeightedContextGraphSnapshot>? Graphs { get; set; }
}

public class FastNodeDto
{
    [JsonPropertyName("color")] public uint Color { get; set; }
    [JsonPropertyName("x")] public float X { get; set; }
    [JsonPropertyName("y")] public float Y { get; set; }
    [JsonPropertyName("obs")] public int ObservationCount { get; set; }
}

public class FastEdgeDto
{
    [JsonPropertyName("from")] public int From { get; set; }
    [JsonPropertyName("dir")] public int Direction { get; set; }
    [JsonPropertyName("to")] public int To { get; set; }
    [JsonPropertyName("w")] public float Weight { get; set; }
}

public class WeightedContextGraphSnapshot
{
    // Normalized pattern distributions: each entry is one NeighborhoodPattern with per-direction color weights
    [JsonPropertyName("patterns")] public List<PatternEntry> Patterns { get; set; } = new();

    // Simple adjacency normalized distributions (fallback)
    [JsonPropertyName("adjacency")] public List<SimpleAdjacencyEntry> Adjacency { get; set; } = new();

    // Known distinct colors (optional; speeds up GetAllColors)
    [JsonPropertyName("colors")] public List<ColorRgb> Colors { get; set; } = new();
}

public class PatternEntry
{
    [JsonPropertyName("center")] public ColorRgb Center { get; set; }
    // neighbors length 8, null means out-of-bounds
    [JsonPropertyName("neighbors")] public ColorRgb?[] Neighbors { get; set; } = new ColorRgb?[8];
    // Direction -> color weights
    [JsonPropertyName("dir")] public List<DirectionWeightsEntry> DirectionWeights { get; set; } = new();
}

public class DirectionWeightsEntry
{
    [JsonPropertyName("direction")] public Direction Direction { get; set; }
    [JsonPropertyName("colors")] public List<ColorWeight> Colors { get; set; } = new();
}

public class ColorWeight
{
    [JsonPropertyName("color")] public ColorRgb Color { get; set; }
    [JsonPropertyName("w")] public double Weight { get; set; }
}

public class SimpleAdjacencyEntry
{
    [JsonPropertyName("center")] public ColorRgb Center { get; set; }
    [JsonPropertyName("direction")] public Direction Direction { get; set; }
    [JsonPropertyName("colors")] public List<ColorWeight> Colors { get; set; } = new();
}

public static class ModelRepository
{
    private static string DefaultPath
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("NNIMAGE_MODEL_PATH");
            if (!string.IsNullOrWhiteSpace(env)) return env!;
            var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "NNImage");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "model_v1.json");
        }
    }

    public static string GetModelsDirectory()
    {
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "NNImage", "models");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static List<string> GetAllSavedModels()
    {
        try
        {
            var modelsDir = GetModelsDirectory();
            if (!Directory.Exists(modelsDir))
                return new List<string>();

            var files = Directory.GetFiles(modelsDir, "*.json")
                .Select(Path.GetFileNameWithoutExtension)
                .Where(name => !string.IsNullOrEmpty(name))
                .OrderByDescending(name => name)
                .ToList();

            return files!;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Failed to list models: {ex.Message}");
            return new List<string>();
        }
    }

    public static string GetModelPath(string modelName)
    {
        var modelsDir = GetModelsDirectory();
        return Path.Combine(modelsDir, $"{modelName}.json");
    }

    public static bool Save(MultiScaleContextGraph graph, string? path = null, string? notes = null, 
        int quantizationLevel = 128, int totalImages = 0, DateTime? firstTrainingDate = null, 
        List<string>? trainingImagePaths = null)
    {
        var file = path ?? DefaultPath;
        var dir = Path.GetDirectoryName(file);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);

        try
        {
            Console.WriteLine($"[ModelRepository] Attempting memory-efficient streaming save...");

            // Try streaming save first (memory-efficient for large models)
            var success = SaveStreaming(graph, file, notes, quantizationLevel, totalImages, firstTrainingDate, trainingImagePaths);
            if (success)
            {
                Console.WriteLine($"[ModelRepository] Streaming save successful to {file} (patterns: {graph.GetTotalPatternCount()}, images: {totalImages})");
                return true;
            }

            Console.WriteLine($"[ModelRepository] Streaming save failed, attempting standard save...");
        }
        catch (OutOfMemoryException)
        {
            Console.WriteLine($"[ModelRepository] Out of memory during streaming save, trying minimal save...");
            return SaveMinimal(graph, file, notes, quantizationLevel, totalImages, firstTrainingDate, trainingImagePaths);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Streaming save error: {ex.Message}, trying standard save...");
        }

        // Fallback to standard save
        try
        {
            var snapshot = graph.ToSnapshot();
            snapshot.Notes = notes;
            snapshot.QuantizationLevel = quantizationLevel;
            snapshot.TotalImages = totalImages;
            snapshot.FirstTrainingDate = firstTrainingDate;
            snapshot.LastTrainingDate = DateTime.Now;
            snapshot.TrainingImagePaths = trainingImagePaths ?? new List<string>();

            var options = new JsonSerializerOptions
            {
                WriteIndented = false, // Compact to save memory
                IncludeFields = false
            };
            string json = System.Text.Json.JsonSerializer.Serialize<ModelSnapshot>(snapshot, options);
            File.WriteAllText(file, json);
            Console.WriteLine($"[ModelRepository] Standard save successful to {file} (patterns: {graph.GetTotalPatternCount()}, images: {totalImages})");
            return true;
        }
        catch (OutOfMemoryException)
        {
            Console.WriteLine($"[ModelRepository] Out of memory during standard save, trying minimal save...");
            return SaveMinimal(graph, file, notes, quantizationLevel, totalImages, firstTrainingDate, trainingImagePaths);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Save failed: {ex.Message}");
            return false;
        }
    }

    private static bool SaveStreaming(MultiScaleContextGraph graph, string file, string? notes, 
        int quantizationLevel, int totalImages, DateTime? firstTrainingDate, List<string>? trainingImagePaths)
    {
        try
        {
            // Use larger buffer for faster I/O with less memory overhead
            using var fileStream = new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.None, 256 * 1024);
            using var jsonWriter = new System.Text.Json.Utf8JsonWriter(fileStream, new System.Text.Json.JsonWriterOptions 
            { 
                Indented = false // Compact format
            });

            var fastGraph = graph.GetFastGraph();
            var nodes = fastGraph.GetAllNodes();

            Console.WriteLine($"[ModelRepository] RAM-aware streaming save: {nodes.Count:N0} nodes");

            // Write JSON in memory-efficient chunks
            jsonWriter.WriteStartObject();

            // Metadata
            jsonWriter.WriteNumber("version", 2);
            jsonWriter.WriteNumber("node_count", nodes.Count);
            jsonWriter.WriteNumber("quantization_level", quantizationLevel);
            jsonWriter.WriteNumber("total_images", totalImages);
            jsonWriter.WriteString("created", DateTime.Now);
            jsonWriter.WriteString("last_training_date", DateTime.Now);
            if (firstTrainingDate.HasValue)
                jsonWriter.WriteString("first_training_date", firstTrainingDate.Value);
            if (!string.IsNullOrEmpty(notes))
                jsonWriter.WriteString("notes", notes);

            // Training paths (limited)
            jsonWriter.WriteStartArray("training_image_paths");
            var pathsToSave = trainingImagePaths?.Take(500).ToList() ?? new List<string>();
            foreach (var imagePath in pathsToSave)
            {
                jsonWriter.WriteStringValue(imagePath);
            }
            jsonWriter.WriteEndArray();

            // Write nodes in larger batches with less frequent GC
            jsonWriter.WriteStartArray("nodes");
            var nodeIndexMap = new Dictionary<GraphNode, int>(nodes.Count);

            const int nodeBatchSize = 50000; // 5x larger batches = 5x faster
            var lastFlush = 0;

            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                nodeIndexMap[node] = i;

                jsonWriter.WriteStartObject();
                jsonWriter.WriteNumber("color", node.Color.ToUInt32());
                jsonWriter.WriteNumber("x", node.NormalizedX);
                jsonWriter.WriteNumber("y", node.NormalizedY);
                jsonWriter.WriteNumber("obs", node.ObservationCount);
                jsonWriter.WriteEndObject();

                // Flush less frequently for speed
                if (i - lastFlush >= nodeBatchSize)
                {
                    jsonWriter.Flush();
                    lastFlush = i;
                    Console.WriteLine($"[ModelRepository] Streamed {i:N0}/{nodes.Count:N0} nodes ({i * 100 / nodes.Count}%)");

                    // Light GC only when needed
                    if (i % (nodeBatchSize * 4) == 0)
                    {
                        GC.Collect(1, GCCollectionMode.Optimized, blocking: false);
                    }
                }
            }
            jsonWriter.WriteEndArray();
            jsonWriter.Flush();

            Console.WriteLine($"[ModelRepository] Nodes complete, streaming edges...");

            // Count edges efficiently
            int totalEdges = 0;
            foreach (var node in nodes)
            {
                foreach (var edgeList in node.Edges.Values)
                {
                    totalEdges += edgeList.Count;
                }
            }
            jsonWriter.WriteNumber("edge_count", totalEdges);

            // Write edges in larger batches
            jsonWriter.WriteStartArray("edges");
            int edgeCount = 0;
            const int edgeBatchSize = 100000; // 2x larger batches
            lastFlush = 0;

            for (int i = 0; i < nodes.Count; i++)
            {
                var fromNode = nodes[i];
                foreach (var kvp in fromNode.Edges)
                {
                    var direction = (int)kvp.Key;
                    foreach (var (toNode, weight) in kvp.Value)
                    {
                        if (!nodeIndexMap.TryGetValue(toNode, out var toIndex))
                            continue;

                        jsonWriter.WriteStartObject();
                        jsonWriter.WriteNumber("from", i);
                        jsonWriter.WriteNumber("dir", direction);
                        jsonWriter.WriteNumber("to", toIndex);
                        jsonWriter.WriteNumber("w", weight);
                        jsonWriter.WriteEndObject();

                        edgeCount++;

                        // Less frequent flushing
                        if (edgeCount - lastFlush >= edgeBatchSize)
                        {
                            jsonWriter.Flush();
                            lastFlush = edgeCount;
                            var pct = totalEdges > 0 ? edgeCount * 100 / totalEdges : 0;
                            Console.WriteLine($"[ModelRepository] Streamed {edgeCount:N0}/{totalEdges:N0} edges ({pct}%)");

                            // Light GC occasionally
                            if (edgeCount % (edgeBatchSize * 4) == 0)
                            {
                                GC.Collect(1, GCCollectionMode.Optimized, blocking: false);
                            }
                        }
                    }
                }
            }
            jsonWriter.WriteEndArray();

            jsonWriter.WriteEndObject();
            jsonWriter.Flush();

            Console.WriteLine($"[ModelRepository] ⚡ ULTRA-FAST streaming save complete: {nodes.Count:N0} nodes, {edgeCount:N0} edges");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Streaming save failed: {ex.Message}");
            return false;
        }
    }

    private static bool SaveMinimal(MultiScaleContextGraph graph, string file, string? notes,
        int quantizationLevel, int totalImages, DateTime? firstTrainingDate, List<string>? trainingImagePaths)
    {
        try
        {
            Console.WriteLine($"[ModelRepository] Attempting minimal save (metadata only)...");

            // Save only essential metadata without full graph data
            var minimalSnapshot = new ModelSnapshot
            {
                Version = 2,
                NodeCount = graph.GetTotalPatternCount(),
                EdgeCount = 0,
                Notes = $"[MINIMAL SAVE] {notes}",
                QuantizationLevel = quantizationLevel,
                TotalImages = totalImages,
                FirstTrainingDate = firstTrainingDate,
                LastTrainingDate = DateTime.Now,
                TrainingImagePaths = trainingImagePaths?.Take(100).ToList() ?? new List<string>(),
                Nodes = null, // Don't save node/edge data to conserve memory
                Edges = null
            };

            var options = new JsonSerializerOptions
            {
                WriteIndented = false,
                IncludeFields = false
            };

            string json = JsonSerializer.Serialize(minimalSnapshot, options);
            File.WriteAllText(file, json);

            Console.WriteLine($"[ModelRepository] Minimal save successful (metadata only) to {file}");
            Console.WriteLine($"[ModelRepository] WARNING: This is a metadata-only save. The model cannot be loaded for generation.");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Minimal save failed: {ex.Message}");
            return false;
        }
    }

    public static bool Clear(string? path = null)
    {
        try
        {
            var file = path ?? DefaultPath;
            if (File.Exists(file))
            {
                File.Delete(file);
                Console.WriteLine($"[ModelRepository] Deleted model file: {file}");
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Clear failed: {ex.Message}");
            return false;
        }
    }

    public static MultiScaleContextGraph? Load(string? path = null, GpuAccelerator? gpu = null)
    {
        var result = LoadWithMetadata(path, gpu);
        return result?.graph;
    }

    public static (MultiScaleContextGraph graph, ModelSnapshot snapshot)? LoadWithMetadata(string? path = null, GpuAccelerator? gpu = null)
    {
        var file = path ?? DefaultPath;
        if (!File.Exists(file))
        {
            Console.WriteLine($"[ModelRepository] No model file at {file}");
            return null;
        }

        // Try streaming load first (memory-efficient for large models)
        try
        {
            Console.WriteLine($"[ModelRepository] Attempting memory-efficient streaming load...");
            var result = LoadStreaming(file, gpu);
            if (result.HasValue)
            {
                Console.WriteLine($"[ModelRepository] Streaming load successful!");
                return result;
            }
            Console.WriteLine($"[ModelRepository] Streaming load failed, attempting standard load...");
        }
        catch (OutOfMemoryException)
        {
            Console.WriteLine($"[ModelRepository] Out of memory during streaming load");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Streaming load error: {ex.Message}, trying standard load...");
        }

        // Fallback to standard load
        try
        {
            var json = File.ReadAllText(file);
            var options = new JsonSerializerOptions();
            var snapshot = JsonSerializer.Deserialize<ModelSnapshot>(json, options);
            if (snapshot == null)
            {
                Console.WriteLine("[ModelRepository] Snapshot deserialized to null");
                return null;
            }
            if (snapshot.Version != 1 && snapshot.Version != 2)
            {
                Console.WriteLine($"[ModelRepository] Unsupported snapshot version: {snapshot.Version}");
                return null;
            }
            // Support single-scale radius 8 (17x17) for large context learning
            if (snapshot.Scales == null || snapshot.Scales.Length == 0 || snapshot.Scales.Any(s => s != 8 && s != 4))
            {
                Console.WriteLine("[ModelRepository] Snapshot scales not strictly matching current single-scale (8); attempting best-effort load");
            }

            var graph = MultiScaleContextGraph.FromSnapshot(snapshot, gpu);
            return (graph, snapshot);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Load failed: {ex.Message}");
            return null;
        }
    }

    private static (MultiScaleContextGraph graph, ModelSnapshot snapshot)? LoadStreaming(string file, GpuAccelerator? gpu)
    {
        try
        {
            var snapshot = new ModelSnapshot();
            var graph = new MultiScaleContextGraph();
            graph.SetGpuAccelerator(gpu);

            using var fileStream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024);

            // Read file in chunks with proper state management
            const int bufferSize = 1024 * 1024; // 1MB chunks
            var buffer = new byte[bufferSize];
            var leftoverBuffer = new List<byte>();

            var nodes = new List<GraphNode>();
            int nodesProcessed = 0;
            int edgesProcessed = 0;
            string? currentProperty = null;
            bool inNodesArray = false;
            bool inEdgesArray = false;
            int depth = 0;

            // Current node/edge being parsed
            uint currentColor = 0;
            float currentX = 0, currentY = 0;
            int currentObs = 0;
            int currentFrom = 0, currentDir = 0, currentTo = 0;
            float currentWeight = 0;

            const int nodeBatchSize = 10000;
            const int edgeBatchSize = 50000;

            var readerState = new JsonReaderState(new JsonReaderOptions { AllowTrailingCommas = true, MaxDepth = 64 });
            bool isComplete = false;

            while (!isComplete)
            {
                int bytesRead = fileStream.Read(buffer, 0, buffer.Length);
                bool isFinalBlock = bytesRead == 0;

                // Combine leftover with new data
                ReadOnlySpan<byte> dataToProcess;
                byte[]? combinedBuffer = null;

                if (leftoverBuffer.Count > 0 && bytesRead > 0)
                {
                    combinedBuffer = new byte[leftoverBuffer.Count + bytesRead];
                    leftoverBuffer.CopyTo(combinedBuffer, 0);
                    Array.Copy(buffer, 0, combinedBuffer, leftoverBuffer.Count, bytesRead);
                    dataToProcess = combinedBuffer;
                    leftoverBuffer.Clear();
                }
                else if (bytesRead > 0)
                {
                    dataToProcess = new ReadOnlySpan<byte>(buffer, 0, bytesRead);
                }
                else if (leftoverBuffer.Count > 0)
                {
                    dataToProcess = leftoverBuffer.ToArray();
                    leftoverBuffer.Clear();
                    isFinalBlock = true;
                }
                else
                {
                    break;
                }

                var reader = new Utf8JsonReader(dataToProcess, isFinalBlock, readerState);

                try
                {
                    while (reader.Read())
                    {
                        switch (reader.TokenType)
                        {
                            case JsonTokenType.PropertyName:
                                currentProperty = reader.GetString();
                                break;

                            case JsonTokenType.Number:
                                if (inNodesArray && depth == 3)
                                {
                                    switch (currentProperty)
                                    {
                                        case "color": currentColor = reader.GetUInt32(); break;
                                        case "x": currentX = reader.GetSingle(); break;
                                        case "y": currentY = reader.GetSingle(); break;
                                        case "obs": currentObs = reader.GetInt32(); break;
                                    }
                                }
                                else if (inEdgesArray && depth == 3)
                                {
                                    switch (currentProperty)
                                    {
                                        case "from": currentFrom = reader.GetInt32(); break;
                                        case "dir": currentDir = reader.GetInt32(); break;
                                        case "to": currentTo = reader.GetInt32(); break;
                                        case "w": currentWeight = reader.GetSingle(); break;
                                    }
                                }
                                else if (depth == 1)
                                {
                                    switch (currentProperty)
                                    {
                                        case "version": snapshot.Version = reader.GetInt32(); break;
                                        case "node_count": snapshot.NodeCount = reader.GetInt32(); break;
                                        case "edge_count": snapshot.EdgeCount = reader.GetInt32(); break;
                                        case "quantization_level": snapshot.QuantizationLevel = reader.GetInt32(); break;
                                        case "total_images": snapshot.TotalImages = reader.GetInt32(); break;
                                    }
                                }
                                break;

                            case JsonTokenType.String:
                                if (depth == 1)
                                {
                                    switch (currentProperty)
                                    {
                                        case "notes": snapshot.Notes = reader.GetString(); break;
                                        case "created": 
                                            if (DateTime.TryParse(reader.GetString(), out var created))
                                                snapshot.Created = created;
                                            break;
                                        case "first_training_date":
                                            if (DateTime.TryParse(reader.GetString(), out var firstDate))
                                                snapshot.FirstTrainingDate = firstDate;
                                            break;
                                        case "last_training_date":
                                            if (DateTime.TryParse(reader.GetString(), out var lastDate))
                                                snapshot.LastTrainingDate = lastDate;
                                            break;
                                    }
                                }
                                break;

                            case JsonTokenType.StartArray:
                                depth++;
                                if (currentProperty == "nodes" && depth == 2)
                                {
                                    inNodesArray = true;
                                    Console.WriteLine($"[ModelRepository] Streaming load: Processing {snapshot.NodeCount:N0} nodes...");
                                }
                                else if (currentProperty == "edges" && depth == 2)
                                {
                                    inEdgesArray = true;
                                    Console.WriteLine($"[ModelRepository] All nodes loaded. Processing {snapshot.EdgeCount:N0} edges...");
                                }
                                break;

                            case JsonTokenType.EndArray:
                                if (inNodesArray && depth == 2)
                                {
                                    inNodesArray = false;
                                    Console.WriteLine($"[ModelRepository] Completed loading {nodes.Count:N0} nodes");
                                }
                                else if (inEdgesArray && depth == 2)
                                {
                                    inEdgesArray = false;
                                    Console.WriteLine($"[ModelRepository] Completed loading {edgesProcessed:N0} edges");
                                }
                                depth--;
                                break;

                            case JsonTokenType.StartObject:
                                depth++;
                                if ((inNodesArray || inEdgesArray) && depth == 3)
                                {
                                    currentColor = 0; currentX = 0; currentY = 0; currentObs = 0;
                                    currentFrom = 0; currentDir = 0; currentTo = 0; currentWeight = 0;
                                }
                                break;

                            case JsonTokenType.EndObject:
                                if (inNodesArray && depth == 3)
                                {
                                    var color = ColorRgb.FromUInt32(currentColor);
                                    var node = graph.GetFastGraph().CreateNodeExact(color, currentX, currentY, currentObs);
                                    nodes.Add(node);
                                    nodesProcessed++;

                                    if (nodesProcessed % nodeBatchSize == 0)
                                    {
                                        var progress = snapshot.NodeCount > 0 ? nodesProcessed * 100 / snapshot.NodeCount : 0;
                                        Console.WriteLine($"[ModelRepository] Loaded {nodesProcessed:N0}/{snapshot.NodeCount:N0} nodes ({progress}%)");
                                        GC.Collect(0, GCCollectionMode.Optimized);
                                    }
                                }
                                else if (inEdgesArray && depth == 3)
                                {
                                    if (currentFrom >= 0 && currentFrom < nodes.Count && currentTo >= 0 && currentTo < nodes.Count)
                                    {
                                        graph.GetFastGraph().AddEdge(nodes[currentFrom], (Direction)currentDir, nodes[currentTo], currentWeight);
                                        edgesProcessed++;

                                        if (edgesProcessed % edgeBatchSize == 0)
                                        {
                                            var progress = snapshot.EdgeCount > 0 ? edgesProcessed * 100 / snapshot.EdgeCount : 0;
                                            Console.WriteLine($"[ModelRepository] Loaded {edgesProcessed:N0}/{snapshot.EdgeCount:N0} edges ({progress}%)");
                                            GC.Collect(0, GCCollectionMode.Optimized);
                                        }
                                    }
                                }
                                depth--;
                                break;
                        }
                    }

                    // Save reader state for next chunk
                    readerState = reader.CurrentState;

                    // Save unconsumed bytes
                    long consumed = reader.BytesConsumed;
                    if (consumed < dataToProcess.Length)
                    {
                        var remaining = dataToProcess.Slice((int)consumed);
                        leftoverBuffer.AddRange(remaining.ToArray());
                    }

                    if (isFinalBlock && leftoverBuffer.Count == 0)
                    {
                        isComplete = true;
                    }
                }
                catch (JsonException ex)
                {
                    // Incomplete token - save remaining data and continue
                    long consumed = reader.BytesConsumed;
                    if (consumed < dataToProcess.Length)
                    {
                        var remaining = dataToProcess.Slice((int)consumed);
                        leftoverBuffer.Clear();
                        leftoverBuffer.AddRange(remaining.ToArray());
                    }

                    readerState = reader.CurrentState;

                    if (isFinalBlock)
                    {
                        Console.WriteLine($"[ModelRepository] JSON parse error at end: {ex.Message}");
                        break;
                    }
                }
            }

            Console.WriteLine($"[ModelRepository] Streaming load complete: {nodes.Count:N0} nodes, {edgesProcessed:N0} edges");
            return (graph, snapshot);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ModelRepository] Streaming load failed: {ex.Message}");
            return null;
        }
    }

}
