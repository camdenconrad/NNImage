using System;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using NNImage.Models;

namespace NNImage.Services;

/// <summary>
/// GPU-accelerated Wave Function Collapse using ILGPU. Keeps output selection
/// behavior equivalent by performing the final color choice on CPU with the
/// same weighting as BitsetWaveFunctionCollapse, while offloading propagation
/// and entropy computation to the GPU.
/// </summary>
public sealed class GpuWaveFunctionCollapse : IDisposable
{
    private readonly int _width;
    private readonly int _height;
    private readonly List<ColorRgb> _palette;
    private readonly Dictionary<ColorRgb, int> _indexOf;
    private readonly Random _random;
    private readonly AdjacencyGraph _adj;

    // ILGPU state
    private readonly Context _context;
    private readonly Accelerator _accel;

    // Device buffers
    private MemoryBuffer1D<DeviceBitMask256, Stride1D.Dense> _masks;
    private MemoryBuffer1D<int, Stride1D.Dense> _collapsed;
    private MemoryBuffer1D<int, Stride1D.Dense> _entropy; // entropy per cell
    private MemoryBuffer1D<DeviceBitMask256, Stride1D.Dense> _compat; // 8*P
    private MemoryBuffer1D<int, Stride1D.Dense> _changeCounter; // size 1

    // Kernels
    private readonly Action<Index1D, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, int> _propagateKernel;
    private readonly Action<Index1D, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> _entropyKernel;
    private readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, int> _applyCollapseKernel;
    private readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>> _clearIntsKernel;

    public delegate void ProgressCallback(uint[] pixels, int iteration, int totalIterations);

    public GpuWaveFunctionCollapse(AdjacencyGraph adjacency, int width, int height, int? seed = null)
    {
        _width = width;
        _height = height;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _adj = adjacency;

        _palette = adjacency.GetAllColors();
        if (_palette.Count == 0) throw new InvalidOperationException("Adjacency graph has no colors");
        if (_palette.Count > 256) throw new NotSupportedException("GPU WFC supports up to 256 colors");

        _indexOf = new Dictionary<ColorRgb, int>(_palette.Count);
        for (int i = 0; i < _palette.Count; i++) _indexOf[_palette[i]] = i;

        // Build compat masks CPU-side once
        var compatHost = BuildCompatibilityMasks(adjacency);

        // Init ILGPU
        try
        {
            _context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
            var dev = _context.GetPreferredDevice(preferCPU: false);
            _accel = dev.CreateAccelerator(_context);
        }
        catch
        {
            // Fallback to CPU accelerator (still correct; mainly for debugging)
            _context = Context.Create(builder => builder.CPU().EnableAlgorithms());
            var dev = _context.GetPreferredDevice(preferCPU: true);
            _accel = dev.CreateAccelerator(_context);
        }

        // Allocate buffers
        int cells = _width * _height;
        _masks = _accel.Allocate1D<DeviceBitMask256>(cells);
        _collapsed = _accel.Allocate1D<int>(cells);
        _entropy = _accel.Allocate1D<int>(cells);
        _compat = _accel.Allocate1D<DeviceBitMask256>(8 * _palette.Count);
        _changeCounter = _accel.Allocate1D<int>(1);

        // Initialize masks to full and collapsed to -1
        var initMasks = new DeviceBitMask256[cells];
        var full = DeviceBitMask256.Full(_palette.Count);
        for (int i = 0; i < cells; i++) initMasks[i] = full;
        _masks.CopyFromCPU(initMasks);
        var negOnes = new int[cells];
        for (int i = 0; i < cells; i++) negOnes[i] = -1;
        _collapsed.CopyFromCPU(negOnes);

        _compat.CopyFromCPU(compatHost);

        // Compile kernels
        _propagateKernel = _accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, int>(PropagateKernel);
        _entropyKernel = _accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(ComputeEntropyKernel);
        _applyCollapseKernel = _accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<DeviceBitMask256, Stride1D.Dense>, int>(ApplyCollapseKernel);
        _clearIntsKernel = _accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ClearIntsKernel);
    }

    private DeviceBitMask256[] BuildCompatibilityMasks(AdjacencyGraph adj)
    {
        // Build inverse compatibility: for each direction d and neighbor color n,
        // produce a mask of center colors c such that adj allows n as a neighbor
        // when looking from c in direction d.
        int P = _palette.Count;
        var compat = new DeviceBitMask256[8 * P];
        for (int dir = 0; dir < 8; dir++)
        {
            for (int ni = 0; ni < P; ni++)
            {
                var neighborColor = _palette[ni];
                var allowedCenters = DeviceBitMask256.Empty;
                for (int ci = 0; ci < P; ci++)
                {
                    var center = _palette[ci];
                    var neighbors = adj.GetPossibleNeighbors(center, (Direction)dir);
                    // if neighborColor is allowed at direction dir from center
                    // then center is allowed for current cell when neighbor at dir is fixed to neighborColor
                    if (neighbors.Contains(neighborColor))
                        allowedCenters.SetBit(ci);
                }
                compat[dir * P + ni] = allowedCenters;
            }
        }
        return compat;
    }

    public uint[] Generate(ProgressCallback? progress = null, int updateFrequency = 50)
    {
        int cells = _width * _height;
        int collapsedCount = 0;
        int iteration = 0;

        // Host-side copies to make decisions
        var collapsedHost = new int[cells];
        for (int i = 0; i < cells; i++) collapsedHost[i] = -1;

        while (collapsedCount < cells)
        {
            // Propagate until stable
            int passes = 0;
            while (true)
            {
                // clear change counter
                _clearIntsKernel((int)1, _changeCounter.View);
                _accel.Synchronize();
                // run propagate
                _propagateKernel(cells, _masks.View, _collapsed.View, _width, _height, _compat.View, _palette.Count);
                _accel.Synchronize();
                // For simplicity of change detection on CPU: recompute entropy and check if any zero? We'll just limit passes
                passes++;
                if (passes >= 4) break; // simple cap to avoid long loops; frontier is small typically
            }

            // Compute entropy and find min on CPU
            _entropyKernel(cells, _masks.View, _collapsed.View, _entropy.View);
            _accel.Synchronize();
            var entropyHost = _entropy.GetAsArray1D();

            int bestIdx = -1;
            int bestEnt = int.MaxValue;
            for (int i = 0; i < cells; i++)
            {
                if (collapsedHost[i] >= 0) continue;
                int e = entropyHost[i];
                if (e <= 0) continue;
                if (e < bestEnt)
                {
                    bestEnt = e; bestIdx = i;
                }
                else if (e == bestEnt && bestEnt != int.MaxValue)
                {
                    if (_random.NextDouble() < 0.5) bestIdx = i;
                }
            }

            if (bestIdx < 0) break; // done

            int bx = bestIdx % _width; int by = bestIdx / _width;
            int chosen = ChooseColorIndexCpu(bx, by, bestIdx, collapsedHost);

            // Apply collapse on device (set collapsed[idx]=chosen and mask to singleton)
            _collapsed.View.SubView(bestIdx, 1).CopyFromCPU(new[] { chosen });
            var singleMask = DeviceBitMask256.MakeSingle(chosen);
            _masks.View.SubView(bestIdx, 1).CopyFromCPU(new DeviceBitMask256[] { singleMask });
            _accel.Synchronize();

            collapsedHost[bestIdx] = chosen;
            collapsedCount++;
            iteration++;

            if (progress != null && iteration % updateFrequency == 0)
            {
                progress(CreatePixelDataHost(collapsedHost), collapsedCount, cells);
            }
        }

        // Fill remaining on CPU for determinism
        for (int i = 0; i < collapsedHost.Length; i++)
        {
            if (collapsedHost[i] < 0)
            {
                int x = i % _width; int y = i / _width;
                collapsedHost[i] = ChooseColorIndexCpu(x, y, i, collapsedHost);
            }
        }
        return CreatePixelDataHost(collapsedHost);
    }

    private uint[] CreatePixelDataHost(int[] collapsedHost)
    {
        var pixels = new uint[_width * _height];
        for (int i = 0; i < pixels.Length; i++)
        {
            int ci = collapsedHost[i];
            if (ci < 0) { pixels[i] = 0xFF000000u; continue; }
            pixels[i] = _palette[ci].ToUInt32();
        }
        return pixels;
    }

    private int ChooseColorIndexCpu(int x, int y, int idx, int[] collapsedHost)
    {
        // Fetch current mask for idx to limit choices
        var maskArr = _masks.GetAsArray1D();
        var mask = maskArr[idx];
        var options = new List<(int ci, double w)>();
        for (int ci = 0; ci < _palette.Count; ci++)
        {
            if (!mask.TestBit(ci)) continue;
            double w = 1.0;
            var candidateColor = _palette[ci];
            for (int d = 0; d < 8; d++)
            {
                var (dx, dy) = ((Direction)d).GetOffset();
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= _width || ny < 0 || ny >= _height) continue;
                int nidx = ny * _width + nx;
                int ncol = collapsedHost[nidx];
                if (ncol >= 0)
                {
                    var neighborColor = _palette[ncol];
                    var possibleNeighbors = _adj.GetPossibleNeighbors(candidateColor, (Direction)d);
                    if (possibleNeighbors.Contains(neighborColor))
                        w *= 2.0;
                }
            }
            options.Add((ci, w));
        }
        if (options.Count == 0) return _random.Next(_palette.Count);
        double total = 0; foreach (var o in options) total += o.w;
        double r = _random.NextDouble() * total; double c = 0;
        foreach (var (ci, w) in options) { c += w; if (r <= c) return ci; }
        return options[0].ci;
    }

    public void Dispose()
    {
        _changeCounter.Dispose();
        _compat.Dispose();
        _entropy.Dispose();
        _collapsed.Dispose();
        _masks.Dispose();
        _accel.Dispose();
        _context.Dispose();
    }

    // Device types and kernels
    public struct DeviceBitMask256
    {
        public ulong W0, W1, W2, W3;

        public static DeviceBitMask256 Empty => default;
        public static DeviceBitMask256 Full(int paletteSize)
        {
            DeviceBitMask256 m = new DeviceBitMask256 { W0 = ~0UL, W1 = ~0UL, W2 = ~0UL, W3 = ~0UL };
            if (paletteSize < 256)
            {
                int extra = 256 - paletteSize;
                for (int i = 0; i < extra; i++) UnsetBit(ref m, 255 - i);
            }
            return m;
        }

        public static void SetBit(ref DeviceBitMask256 m, int idx)
        {
            if (idx < 64) m.W0 |= 1UL << idx;
            else if (idx < 128) m.W1 |= 1UL << (idx - 64);
            else if (idx < 192) m.W2 |= 1UL << (idx - 128);
            else m.W3 |= 1UL << (idx - 192);
        }
        public void SetBit(int idx) => SetBit(ref this, idx);

        public static void UnsetBit(ref DeviceBitMask256 m, int idx)
        {
            if (idx < 64) m.W0 &= ~(1UL << idx);
            else if (idx < 128) m.W1 &= ~(1UL << (idx - 64));
            else if (idx < 192) m.W2 &= ~(1UL << (idx - 128));
            else m.W3 &= ~(1UL << (idx - 192));
        }
        public bool TestBit(int idx)
        {
            if (idx < 64) return (W0 & (1UL << idx)) != 0;
            if (idx < 128) return (W1 & (1UL << (idx - 64))) != 0;
            if (idx < 192) return (W2 & (1UL << (idx - 128))) != 0;
            return (W3 & (1UL << (idx - 192))) != 0;
        }
        public static DeviceBitMask256 And(DeviceBitMask256 a, DeviceBitMask256 b)
        {
            return new DeviceBitMask256 { W0 = a.W0 & b.W0, W1 = a.W1 & b.W1, W2 = a.W2 & b.W2, W3 = a.W3 & b.W3 };
        }
        public static DeviceBitMask256 MakeSingle(int idx)
        {
            DeviceBitMask256 m = default;
            SetBit(ref m, idx);
            return m;
        }
        public int PopCount()
        {
            return XMath.PopCount((uint)(W0 & 0xFFFFFFFF)) + XMath.PopCount((uint)(W0 >> 32))
                 + XMath.PopCount((uint)(W1 & 0xFFFFFFFF)) + XMath.PopCount((uint)(W1 >> 32))
                 + XMath.PopCount((uint)(W2 & 0xFFFFFFFF)) + XMath.PopCount((uint)(W2 >> 32))
                 + XMath.PopCount((uint)(W3 & 0xFFFFFFFF)) + XMath.PopCount((uint)(W3 >> 32));
        }
    }

    private static void PropagateKernel(
        Index1D index,
        ArrayView1D<DeviceBitMask256, Stride1D.Dense> masks,
        ArrayView1D<int, Stride1D.Dense> collapsed,
        int width,
        int height,
        ArrayView1D<DeviceBitMask256, Stride1D.Dense> compat,
        int paletteSize)
    {
        int i = index;
        int cells = width * height;
        if (i >= cells) return;
        int x = i % width; int y = i / width;

        // If already collapsed, ensure mask is singleton
        int col = collapsed[i];
        if (col >= 0)
        {
            DeviceBitMask256 single = DeviceBitMask256.Empty;
            DeviceBitMask256.SetBit(ref single, col);
            masks[i] = single;
            return;
        }

        DeviceBitMask256 current = masks[i];
        // Intersect with constraints from each neighbor that is collapsed
        // N, NE, E, SE, S, SW, W, NW
        // For each neighbor with color c, allowed = compat[dir, c]
        // dir indexing must match Direction enum ordering used in host

        // North (0)
        if (y > 0)
        {
            int ni = i - width; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 0 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // NorthEast (1)
        if (y > 0 && x + 1 < width)
        {
            int ni = i - width + 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 1 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // East (2)
        if (x + 1 < width)
        {
            int ni = i + 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 2 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // SouthEast (3)
        if (y + 1 < height && x + 1 < width)
        {
            int ni = i + width + 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 3 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // South (4)
        if (y + 1 < height)
        {
            int ni = i + width; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 4 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // SouthWest (5)
        if (y + 1 < height && x > 0)
        {
            int ni = i + width - 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 5 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // West (6)
        if (x > 0)
        {
            int ni = i - 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 6 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }
        // NorthWest (7)
        if (y > 0 && x > 0)
        {
            int ni = i - width - 1; int c = collapsed[ni];
            if (c >= 0)
            {
                var allowed = compat[c + 7 * paletteSize];
                current = DeviceBitMask256.And(current, allowed);
            }
        }

        masks[i] = current;
    }

    private static void ComputeEntropyKernel(
        Index1D index,
        ArrayView1D<DeviceBitMask256, Stride1D.Dense> masks,
        ArrayView1D<int, Stride1D.Dense> collapsed,
        ArrayView1D<int, Stride1D.Dense> entropy)
    {
        int i = index;
        if (i >= masks.Length) return;
        if (collapsed[i] >= 0) { entropy[i] = int.MaxValue; return; }
        entropy[i] = masks[i].PopCount();
    }

    private static void ApplyCollapseKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> collapsed,
        ArrayView1D<DeviceBitMask256, Stride1D.Dense> masks,
        int paletteSize)
    {
        // Expects a 2-int buffer in global memory before this call: [idx, color]
        // For simplicity in this kernel signature, we assume single element launch and fetch from the end of collapsed array not possible here.
        // In practice this kernel is a placeholder; collapse is handled by host-side copies to collapsed array in this version.
    }

    private static void ClearIntsKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
    {
        if (index < data.Length) data[index] = 0;
    }
}
