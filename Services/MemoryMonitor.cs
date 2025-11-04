using System;
using System.Diagnostics;
using System.Threading;

namespace NNImage.Services;

/// <summary>
/// Monitors system memory usage and triggers streaming/pausing when approaching RAM limit
/// Prevents system-wide freezes by managing memory proactively
/// Uses cross-platform .NET Core GC memory info for accurate RAM tracking
/// </summary>
public class MemoryMonitor
{
    private readonly long _maxRamBytes;
    private readonly long _streamingThresholdBytes;
    private readonly long _pauseThresholdBytes;
    private long _lastMemoryCheck = 0;
    private readonly long _checkIntervalTicks;

    public MemoryMonitor(int maxRamGB = 30)
    {
        _maxRamBytes = (long)maxRamGB * 1024 * 1024 * 1024;
        _streamingThresholdBytes = (long)(_maxRamBytes * 0.85); // Start streaming at 85%
        _pauseThresholdBytes = (long)(_maxRamBytes * 0.95); // Pause at 95%
        _checkIntervalTicks = Stopwatch.Frequency / 10; // Check 10 times per second max

        Console.WriteLine($"[MemoryMonitor] Initialized with {maxRamGB}GB limit");
        Console.WriteLine($"[MemoryMonitor] Streaming threshold: {_streamingThresholdBytes / (1024 * 1024 * 1024)}GB ({(_streamingThresholdBytes * 100.0 / _maxRamBytes):F1}%)");
        Console.WriteLine($"[MemoryMonitor] Pause threshold: {_pauseThresholdBytes / (1024 * 1024 * 1024)}GB ({(_pauseThresholdBytes * 100.0 / _maxRamBytes):F1}%)");
        Console.WriteLine("[MemoryMonitor] Using cross-platform GC memory info for RAM tracking");
    }

    /// <summary>
    /// Check current memory usage. Returns status based on thresholds.
    /// </summary>
    public MemoryStatus CheckMemory()
    {
        var now = Stopwatch.GetTimestamp();

        // Rate limit checks for performance
        if (now - _lastMemoryCheck < _checkIntervalTicks)
        {
            return MemoryStatus.Normal;
        }

        _lastMemoryCheck = now;

        // Use GC memory info for cross-platform accurate memory tracking
        var gcInfo = GC.GetGCMemoryInfo();

        // Get process working set for process-specific RAM usage
        long usedBytes;
        using (var process = Process.GetCurrentProcess())
        {
            usedBytes = process.WorkingSet64;
        }

        // Also check heap size for managed allocations
        var heapSize = gcInfo.HeapSizeBytes;
        var totalLoad = gcInfo.MemoryLoadBytes;

        // Use the maximum of process working set, heap size, or memory load
        usedBytes = Math.Max(usedBytes, Math.Max(heapSize, totalLoad));

        var usedPercent = (usedBytes * 100.0) / _maxRamBytes;

        if (usedBytes >= _pauseThresholdBytes)
        {
            Console.WriteLine($"[MemoryMonitor] ⚠ CRITICAL RAM: {usedBytes / (1024 * 1024 * 1024):F2}GB ({usedPercent:F1}%) - PAUSING");
            return MemoryStatus.Critical;
        }
        else if (usedBytes >= _streamingThresholdBytes)
        {
            Console.WriteLine($"[MemoryMonitor] ⚠ High RAM: {usedBytes / (1024 * 1024 * 1024):F2}GB ({usedPercent:F1}%) - STREAMING");
            return MemoryStatus.Streaming;
        }

        return MemoryStatus.Normal;
    }

    /// <summary>
    /// Wait for memory to drop below streaming threshold
    /// </summary>
    public void WaitForMemoryRecovery(Action<string>? statusCallback = null)
    {
        Console.WriteLine("[MemoryMonitor] Waiting for memory recovery...");
        statusCallback?.Invoke("⚠ High memory - waiting for GC...");

        var startTime = Stopwatch.GetTimestamp();
        var gcAttempts = 0;

        while (true)
        {
            // Force aggressive GC
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            gcAttempts++;

            Thread.Sleep(500); // Give system time to release memory

            var status = CheckMemory();
            if (status == MemoryStatus.Normal)
            {
                var elapsedSec = (Stopwatch.GetTimestamp() - startTime) / (double)Stopwatch.Frequency;
                Console.WriteLine($"[MemoryMonitor] Memory recovered after {elapsedSec:F1}s ({gcAttempts} GC cycles)");
                statusCallback?.Invoke($"✓ Memory recovered ({elapsedSec:F1}s)");
                return;
            }

            if (status == MemoryStatus.Streaming)
            {
                Console.WriteLine($"[MemoryMonitor] Memory at streaming level - continuing with caution");
                statusCallback?.Invoke("⚠ Memory at streaming level");
                return;
            }

            // Continue waiting if still critical
            var elapsed = (Stopwatch.GetTimestamp() - startTime) / (double)Stopwatch.Frequency;
            if (elapsed > 30) // Max wait time
            {
                Console.WriteLine($"[MemoryMonitor] ⚠ Memory recovery timeout after {elapsed:F1}s");
                statusCallback?.Invoke("⚠ Memory recovery timeout - continuing with risk");
                return;
            }

            statusCallback?.Invoke($"⚠ Waiting for memory... ({elapsed:F0}s)");
        }
    }

    public void Dispose()
    {
        // Nothing to dispose - using built-in GC memory info
    }
}

public enum MemoryStatus
{
    Normal,      // < 85% - normal operation
    Streaming,   // 85-95% - enable streaming, reduce batch sizes
    Critical     // > 95% - pause and wait for GC
}
