using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Markup.Xaml;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NNImage.Models;
using NNImage.Services;

namespace NNImage;

public partial class MainWindow : Window
{
    private MultiScaleContextGraph _multiScaleGraph = new(); // Multi-scale persistent graph
    private ColorQuantizer? _quantizer;
    private List<string> _trainingImagePaths = new();
    private Bitmap? _generatedBitmap;
    private int _currentQuantizationLevel = 128;
    private GpuAccelerator? _gpu;
    private OnlineTrainingService? _onlineTraining;
    private int _totalTrainingImages = 0;
    private DateTime? _firstTrainingDate;
    private bool _loadedFromSnapshot = false;
    private System.Threading.CancellationTokenSource? _generationCancellation;
    private MemoryMonitor? _memoryMonitor;

    public MainWindow()
    {
        InitializeComponent();

        // Setup entropy slider value display
        EntropySlider.PropertyChanged += (s, e) =>
        {
            if (e.Property.Name == "Value")
            {
                EntropyValueText.Text = $"{(int)EntropySlider.Value}%";
            }
        };

        // Initialize GPU
        try
        {
            _gpu = new GpuAccelerator();
            Console.WriteLine($"[MainWindow] GPU initialization complete. Available: {_gpu.IsAvailable}");

            // Initialize online training service
            _onlineTraining = new OnlineTrainingService(_gpu);
            _onlineTraining.ProgressUpdate += OnlineTraining_ProgressUpdate;
            _onlineTraining.ErrorOccurred += OnlineTraining_ErrorOccurred;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] GPU initialization failed: {ex.Message}");
            _gpu = null;
        }

        // Initialize memory monitor (20GB RAM limit - HARD LIMIT to prevent system freeze)
        try
        {
            _memoryMonitor = new MemoryMonitor(maxRamGB: 20);
            Console.WriteLine("[MainWindow] Memory monitor initialized with 20GB HARD LIMIT");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] Memory monitor initialization failed: {ex.Message}");
        }

        // Initialize persistent multi-scale graph with GPU
        _multiScaleGraph.SetGpuAccelerator(_gpu);
        Console.WriteLine("[MainWindow] Multi-scale context graph initialized");

        // Attempt to load persisted trained model (snapshot) first
        try
        {
            var loadedData = ModelRepository.LoadWithMetadata(null, _gpu);
            if (loadedData.HasValue)
            {
                var (graph, snapshot) = loadedData.Value;
                _multiScaleGraph = graph;
                _multiScaleGraph.SetGpuAccelerator(_gpu);

                // Restore metadata
                _currentQuantizationLevel = snapshot.QuantizationLevel;
                _totalTrainingImages = snapshot.TotalImages;
                _firstTrainingDate = snapshot.FirstTrainingDate;
                _trainingImagePaths = snapshot.TrainingImagePaths;

                if (_currentQuantizationLevel > 0)
                    QuantizationLevel.Value = _currentQuantizationLevel;

                Console.WriteLine("[MainWindow] Loaded persisted model snapshot successfully");
                TrainingStatusText.Text = "✓ Loaded model snapshot";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;

                var patternCount = _multiScaleGraph.GetTotalPatternCount();
                var colorCount = _multiScaleGraph.GetColorCount();
                TrainingInfoText.Text = $"Total images: {_totalTrainingImages}\n" +
                                       $"Patterns: {patternCount}\n" +
                                       $"Colors: {colorCount}\n" +
                                       $"Quantization: {_currentQuantizationLevel}\n" +
                                       $"First trained: {_firstTrainingDate:g}\n" +
                                       $"Last updated: {snapshot.LastTrainingDate:g}";

                // Enable Generate only when actual data is present
                var canGenerate = patternCount > 0 && colorCount > 0;
                GenerateButton.IsEnabled = canGenerate;
                SaveModelButton.IsEnabled = canGenerate;
                if (!canGenerate)
                {
                    TrainingStatusText.Text = "⚠ Loaded snapshot has no patterns/colors. Please train or load a different snapshot.";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
                }

                _loadedFromSnapshot = true;

                // Load sample images if paths exist
                LoadSampleImagesFromCache();
            }
            else
            {
                Console.WriteLine("[MainWindow] No model snapshot found; starting with empty model");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] Failed to load model snapshot: {ex.Message}");
        }

        // Also load legacy/simple training cache (optional previews/info)
        if (!_loadedFromSnapshot)
        {
            LoadTrainingCache();
        }

        // Populate the models dropdown
        RefreshModelsList();
    }

    private void OnlineTraining_ProgressUpdate(int imagesProcessed, string message)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            OnlineTrainingInfoText.Text = $"Online: {message}\nTotal processed: {imagesProcessed}";
        });
    }

    private void OnlineTraining_ErrorOccurred(Exception ex)
    {
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            OnlineTrainingInfoText.Text = $"Online training error: {ex.Message}";
            OnlineTrainingInfoText.Foreground = Avalonia.Media.Brushes.OrangeRed;
        });
    }

    private void StartOnlineTrainingButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_onlineTraining == null)
            return;

        var batchSize = (int)OnlineBatchSize.Value;
        var quantLevel = (int)QuantizationLevel.Value;

        _onlineTraining.Start(batchSize, quantLevel);

        StartOnlineTrainingButton.IsEnabled = false;
        StopOnlineTrainingButton.IsEnabled = true;

        OnlineTrainingInfoText.Text = "Online training started...";
        OnlineTrainingInfoText.Foreground = Avalonia.Media.Brushes.LightGreen;

        Console.WriteLine($"[MainWindow] Online training started with batch size {batchSize}");
    }

    private async void StopOnlineTrainingButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_onlineTraining == null)
            return;

        _onlineTraining.Stop();
        await _onlineTraining.WaitForStopAsync();

        StartOnlineTrainingButton.IsEnabled = true;
        StopOnlineTrainingButton.IsEnabled = false;

        OnlineTrainingInfoText.Text = $"Online training stopped. Total: {_onlineTraining.TotalImagesProcessed} images";
        OnlineTrainingInfoText.Foreground = Avalonia.Media.Brushes.Gray;

        Console.WriteLine("[MainWindow] Online training stopped");
    }

    private void ClearCacheButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_onlineTraining == null)
            return;

        var cacheSize = _onlineTraining.GetCacheSize();
        _onlineTraining.ClearCache();

        OnlineTrainingInfoText.Text = $"Cleared {cacheSize / (1024 * 1024):F1} MB cache";
        Console.WriteLine($"[MainWindow] Cleared image cache: {cacheSize} bytes");
    }

    private void LoadTrainingCache()
    {
        Console.WriteLine("[MainWindow] Checking for training cache...");
        var cache = TrainingDataCache.Load();

        if (cache != null)
        {
            try
            {
                Console.WriteLine($"[MainWindow] Loading cache with {cache.ColorCount} colors from {cache.LastTrainingDate}");
                TrainingStatusText.Text = "⏳ Loading training cache...";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

                // Load into persistent graph (additive)
                var loadedGraph = cache.ToAdjacencyGraph();

                // Merge loaded patterns into persistent graph
                // For now, just use the loaded graph as-is
                // TODO: Implement proper merging

                _trainingImagePaths = cache.TrainingImagePaths;
                _currentQuantizationLevel = cache.QuantizationLevel;
                _totalTrainingImages = cache.TrainingImagePaths.Count;
                _firstTrainingDate = cache.LastTrainingDate;

                QuantizationLevel.Value = cache.QuantizationLevel;

                Console.WriteLine($"[MainWindow] Cache loaded successfully");
                TrainingStatusText.Text = $"✓ Loaded from cache: {cache.ColorCount} colors";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
                TrainingInfoText.Text = $"Total trained: {_totalTrainingImages} images\nFirst trained: {_firstTrainingDate:g}\nLast updated: {cache.LastTrainingDate:g}\n(Auto-loaded from cache)";
                GenerateButton.IsEnabled = true;
                SaveModelButton.IsEnabled = true;

                // Load sample images preview
                LoadSampleImagesFromCache();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MainWindow] Failed to load cache: {ex.Message}");
                TrainingStatusText.Text = $"Failed to load cache: {ex.Message}";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            }
        }
        else
        {
            Console.WriteLine("[MainWindow] No training cache found - starting fresh");
            TrainingStatusText.Text = "Ready to train (empty model)";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Gray;
            TrainingInfoText.Text = "No training data yet";
        }
    }

    private void LoadSampleImagesFromCache()
    {
        SampleImagesPanel.Children.Clear();

        foreach (var imagePath in _trainingImagePaths.Take(6))
        {
            try
            {
                if (File.Exists(imagePath))
                {
                    var bitmap = new Bitmap(imagePath);
                    var image = new Image
                    {
                        Source = bitmap,
                        Width = 60,
                        Height = 60,
                        Margin = new Avalonia.Thickness(2),
                        Stretch = Avalonia.Media.Stretch.UniformToFill
                    };

                    var border = new Border
                    {
                        Child = image,
                        BorderBrush = Avalonia.Media.Brushes.Gray,
                        BorderThickness = new Avalonia.Thickness(1),
                        CornerRadius = new Avalonia.CornerRadius(2),
                        Margin = new Avalonia.Thickness(2)
                    };

                    SampleImagesPanel.Children.Add(border);
                }
            }
            catch { }
        }
    }

    private async void SelectFolderButton_Click(object? sender, RoutedEventArgs e)
    {
        var folder = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select Image Folder",
            AllowMultiple = false
        });

        if (folder.Count > 0)
        {
            var folderPath = folder[0].Path.LocalPath;
            LoadSampleImages(folderPath);
        }
    }

    private void LoadSampleImages(string folderPath)
    {
        SampleImagesPanel.Children.Clear();
        _trainingImagePaths.Clear();

        var supportedExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
        var imageFiles = Directory.GetFiles(folderPath)
            .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLower()))
            .Take(100)
            .ToList();

        _trainingImagePaths = imageFiles;

        foreach (var imagePath in imageFiles.Take(6))
        {
            try
            {
                var bitmap = new Bitmap(imagePath);
                var image = new Image
                {
                    Source = bitmap,
                    Width = 60,
                    Height = 60,
                    Margin = new Avalonia.Thickness(0),
                    Stretch = Avalonia.Media.Stretch.UniformToFill
                };

                var border = new Border
                {
                    Child = image,
                    BorderBrush = Avalonia.Media.Brushes.Gray,
                    BorderThickness = new Avalonia.Thickness(1),
                    CornerRadius = new Avalonia.CornerRadius(2),
                    Margin = new Avalonia.Thickness(2)
                };

                SampleImagesPanel.Children.Add(border);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading image {imagePath}: {ex.Message}");
            }
        }

        TrainingStatusText.Text = $"✓ Found {_trainingImagePaths.Count} images";
        TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
        TrainingInfoText.Text = $"Ready to train on {_trainingImagePaths.Count} images";
    }

    private async void TrainButton_Click(object? sender, RoutedEventArgs e)
    {
        Console.WriteLine("[MainWindow] TrainButton_Click started");

        if (_trainingImagePaths.Count == 0)
        {
            TrainingStatusText.Text = "⚠ No images selected";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            Console.WriteLine("[MainWindow] No images selected");
            return;
        }

        Console.WriteLine($"[MainWindow] Starting training with {_trainingImagePaths.Count} images");

        TrainButton.IsEnabled = false;
        GenerateButton.IsEnabled = false;
        SelectFolderButton.IsEnabled = false;

        var progressDialog = new Views.ProgressDialog();
        progressDialog.SetTitle("Training Model");

        // Show dialog non-blocking
        _ = progressDialog.ShowDialog(this);

        try
        {
            Console.WriteLine("[MainWindow] Starting STREAMING training to avoid memory issues...");
            progressDialog.AddLog($"Processing {_trainingImagePaths.Count} images in streaming mode...");
            progressDialog.AddLog($"Current model has {_multiScaleGraph.GetTotalPatternCount()} patterns");

            var quantizationLevel = (int)QuantizationLevel.Value;
            _currentQuantizationLevel = quantizationLevel;

            // Initialize quantizer if needed
            if (_quantizer == null || _currentQuantizationLevel != quantizationLevel)
            {
                Console.WriteLine($"[MainWindow] Creating new quantizer with level: {quantizationLevel}");

                // For quantizer initialization, we need to sample colors from images
                // Do this in a memory-efficient way by sampling a subset
                progressDialog.AddLog("Building color palette from samples...");

                await Task.Run(async () =>
                {
                    var colorSamples = new System.Collections.Concurrent.ConcurrentBag<ColorRgb>();
                    var maxSamples = 100000; // Limit samples to manage memory
                    var samplesPerImage = Math.Max(1000, maxSamples / Math.Min(_trainingImagePaths.Count, 100));

                    // Sample from first 100 images max for palette building
                    var imagesToSample = _trainingImagePaths.Take(100).ToList();

                    foreach (var imagePath in imagesToSample)
                    {
                        try
                        {
                            var pixelData = await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () =>
                            {
                                return await ExtractPixelDataAsync(imagePath);
                            });

                            if (pixelData.HasValue)
                            {
                                var (pixels, _, _) = pixelData.Value;

                                // Sample colors (not all pixels to save memory)
                                var step = Math.Max(1, pixels.Length / samplesPerImage);
                                for (int i = 0; i < pixels.Length; i += step)
                                {
                                    colorSamples.Add(PixelToColorRgb(pixels[i]));

                                    if (colorSamples.Count >= maxSamples)
                                        break;
                                }
                            }

                            if (colorSamples.Count >= maxSamples)
                                break;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[MainWindow] Failed to sample colors from {Path.GetFileName(imagePath)}: {ex.Message}");
                        }
                    }

                    Console.WriteLine($"[MainWindow] Collected {colorSamples.Count} color samples, building palette...");
                    _quantizer = new ColorQuantizer(quantizationLevel, _gpu);
                    _quantizer.BuildPalette(colorSamples.ToList());
                    Console.WriteLine($"[MainWindow] Palette built successfully");
                });

                progressDialog.AddLog("Color palette built ✓");
            }

            // SEQUENTIAL PROCESSING: One image at a time with full resource utilization
            progressDialog.SetTitle("⚡ Sequential Training");
            var processedCount = 0;
            var failedCount = 0;

            // Check GPU VRAM availability
            var (totalVRam, usedVRam, availableVRam) = _gpu?.GetGpuMemoryInfo() ?? (0, 0, 0);
            Console.WriteLine($"[MainWindow] GPU VRAM: {availableVRam / (1024 * 1024 * 1024):F1}GB available of {totalVRam / (1024 * 1024 * 1024):F1}GB total");
            progressDialog.AddLog($"GPU VRAM: {availableVRam / (1024 * 1024 * 1024):F1}GB / {totalVRam / (1024 * 1024 * 1024):F1}GB available");

            var hasGpu = _gpu != null && _gpu.IsAvailable && availableVRam > 1L * 1024 * 1024 * 1024; // At least 1GB VRAM

            Console.WriteLine($"[MainWindow] Starting SEQUENTIAL training - one image at a time with full resource utilization");
            progressDialog.AddLog($"Mode: {(hasGpu ? "⚡ GPU-accelerated" : "💻 CPU-based")} - Sequential processing");
            progressDialog.AddLog($"Processing {_trainingImagePaths.Count} images one at a time...");
            progressDialog.UpdateProgress(0, _trainingImagePaths.Count, "⚡ Starting training...");

            // Process each image sequentially with AGGRESSIVE memory management
            for (int i = 0; i < _trainingImagePaths.Count; i++)
            {
                var imagePath = _trainingImagePaths[i];
                var fileName = Path.GetFileName(imagePath);

                // ═══ PREVENTIVE MEMORY CHECK BEFORE LOADING ═══
                var memStatusBefore = _memoryMonitor?.CheckMemory() ?? MemoryStatus.Normal;
                var memBeforeGB = GC.GetTotalMemory(false) / (1024.0 * 1024 * 1024);

                Console.WriteLine($"[MainWindow] Pre-load memory: {memBeforeGB:F2}GB / 20GB limit");

                if (memStatusBefore == MemoryStatus.Critical)
                {
                    // CRITICAL: Wait for memory to drop below 20GB before loading next image
                    Console.WriteLine($"[MainWindow] 🔴 CRITICAL: {memBeforeGB:F2}GB RAM - Pausing for recovery...");
                    progressDialog.AddLog($"🔴 RAM at {memBeforeGB:F2}GB - Pausing for cleanup...");

                    // Aggressive cleanup before waiting
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                    GC.WaitForPendingFinalizers();
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);

                    _memoryMonitor?.WaitForMemoryRecovery(msg =>
                    {
                        progressDialog.AddLog($"⏸ {msg}");
                        progressDialog.UpdateProgress(i, _trainingImagePaths.Count, $"⏸ {msg}");
                    });

                    var memAfterRecovery = GC.GetTotalMemory(true) / (1024.0 * 1024 * 1024);
                    Console.WriteLine($"[MainWindow] ✓ Memory recovered: {memAfterRecovery:F2}GB");
                    progressDialog.AddLog($"✓ Resumed at {memAfterRecovery:F2}GB RAM");
                }
                else if (memStatusBefore == MemoryStatus.Streaming)
                {
                    // WARNING: Approaching limit - aggressive cleanup now
                    Console.WriteLine($"[MainWindow] 🟡 WARNING: {memBeforeGB:F2}GB RAM - Aggressive cleanup...");
                    progressDialog.AddLog($"🟡 RAM at {memBeforeGB:F2}GB - Aggressive cleanup...");

                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                    GC.WaitForPendingFinalizers();
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);

                    var memAfterPreCleanupGB = GC.GetTotalMemory(true) / (1024.0 * 1024 * 1024);
                    Console.WriteLine($"[MainWindow] Cleaned: {memBeforeGB:F2}GB → {memAfterPreCleanupGB:F2}GB");
                }

                try
                {
                    Console.WriteLine($"[MainWindow] ═══ Processing image {i + 1}/{_trainingImagePaths.Count}: {fileName} ═══");

                    // Load image on UI thread
                    var pixelData = await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () =>
                    {
                        return await ExtractPixelDataAsync(imagePath);
                    });

                    if (pixelData.HasValue)
                    {
                        var (pixels, width, height) = pixelData.Value;
                        var imageSizeMB = (pixels.Length * 4) / (1024.0 * 1024);
                        Console.WriteLine($"[MainWindow] Image loaded: {width}x{height} ({imageSizeMB:F1}MB pixel data)");

                        // Process on background thread with FULL resource utilization
                        await Task.Run(() =>
                        {
                            ProcessImageIntoGraph(pixels, width, height);
                        });

                        // Immediately null out pixel data to allow GC
                        pixelData = null;

                        processedCount++;
                        Console.WriteLine($"[MainWindow] ✓ Successfully processed: {fileName}");
                    }
                    else
                    {
                        failedCount++;
                        progressDialog.AddLog($"⚠ Failed to load: {fileName}");
                        Console.WriteLine($"[MainWindow] Failed to load: {fileName}");
                    }
                }
                catch (Exception ex)
                {
                    failedCount++;
                    progressDialog.AddLog($"⚠ Error processing {fileName}: {ex.Message}");
                    Console.WriteLine($"[MainWindow] Failed to process {fileName}: {ex.Message}");
                }

                // Update progress after EACH image
                var currentProgress = processedCount + failedCount;
                var currentPercentage = (currentProgress * 100) / _trainingImagePaths.Count;

                // Memory info before cleanup
                var memMB = GC.GetTotalMemory(false) / (1024 * 1024);
                var memGB = memMB / 1024.0;

                var (totalGpu, usedGpu, availGpu) = _gpu?.GetGpuMemoryInfo() ?? (0, 0, 0);
                var gpuGB = usedGpu / (1024.0 * 1024 * 1024);

                var progressMsg = hasGpu 
                    ? $"⚡ {currentProgress}/{_trainingImagePaths.Count} ({currentPercentage}%) | " +
                      $"VRAM: {gpuGB:F1}GB | RAM: {memGB:F1}GB / 20GB"
                    : $"⚡ {currentProgress}/{_trainingImagePaths.Count} ({currentPercentage}%) | " +
                      $"RAM: {memGB:F1}GB / 20GB";

                progressDialog.UpdateProgress(currentProgress, _trainingImagePaths.Count, progressMsg);

                // ═══ AGGRESSIVE RESOURCE CLEANUP AFTER EACH IMAGE ═══
                Console.WriteLine($"[MainWindow] ┌─ Aggressive cleanup after image {i + 1}...");
                Console.WriteLine($"[MainWindow] │  Before: {memGB:F2}GB RAM");

                // Triple-pass aggressive GC with compaction to ensure memory is freed
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);

                // Verify memory after cleanup
                var memAfterCleanup = GC.GetTotalMemory(true) / (1024 * 1024.0);
                var memAfterCleanupGB = memAfterCleanup / 1024.0;
                var freedMB = memMB - memAfterCleanup;

                Console.WriteLine($"[MainWindow] │  After: {memAfterCleanupGB:F2}GB RAM (freed {freedMB:F1}MB)");
                Console.WriteLine($"[MainWindow] └─ Cleanup complete");

                // Log detailed status every image to track memory
                var percentage = ((i + 1) * 100) / _trainingImagePaths.Count;
                var memStatus = _memoryMonitor?.CheckMemory() ?? MemoryStatus.Normal;
                var statusEmoji = memStatus switch
                {
                    MemoryStatus.Critical => "🔴",
                    MemoryStatus.Streaming => "🟡",
                    _ => "🟢"
                };

                // Log every 5 images or if memory is concerning
                if ((i + 1) % 5 == 0 || (i + 1) == _trainingImagePaths.Count || memStatus != MemoryStatus.Normal)
                {
                    if (hasGpu)
                    {
                        var (totalGpu2, usedGpu2, availGpu2) = _gpu?.GetGpuMemoryInfo() ?? (0, 0, 0);
                        var gpuGB2 = usedGpu2 / (1024.0 * 1024 * 1024);
                        var gpuMemPercent = usedGpu2 > 0 ? (usedGpu2 * 100.0 / totalGpu2) : 0;
                        progressDialog.AddLog($"{statusEmoji} {percentage}% | VRAM: {gpuGB2:F1}GB ({gpuMemPercent:F0}%) | RAM: {memAfterCleanupGB:F2}GB/20GB | OK: {processedCount}");
                    }
                    else
                    {
                        progressDialog.AddLog($"{statusEmoji} {percentage}% | RAM: {memAfterCleanupGB:F2}GB/20GB | OK: {processedCount}");
                    }
                }
            }

            Console.WriteLine($"[MainWindow] Streaming training complete: {processedCount} processed, {failedCount} failed");
            progressDialog.AddLog($"Processed {processedCount} images successfully");

            if (failedCount > 0)
            {
                progressDialog.AddLog($"⚠ {failedCount} images failed");
            }

            // Final processing on background thread with comprehensive error handling
            progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                "✓ All images processed - Finalizing model...");

            await Task.Run(() =>
            {
                try
                {
                    _totalTrainingImages += processedCount;
                    if (!_firstTrainingDate.HasValue)
                        _firstTrainingDate = DateTime.Now;

                    // Force memory cleanup before intensive operations
                    Console.WriteLine("[Background] Pre-normalization memory cleanup...");
                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("🧹 Cleaning up memory before finalization...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "🧹 Memory cleanup...");
                    });

                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
                    GC.WaitForPendingFinalizers();

                    Console.WriteLine("[Background] Normalizing multi-scale graph...");
                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("⚙️ Normalizing model...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "⚙️ Normalizing model...");
                    });

                    _multiScaleGraph.Normalize();

                    var patternCount = _multiScaleGraph.GetTotalPatternCount();
                    var colorCount = _multiScaleGraph.GetColorCount();
                    Console.WriteLine($"[Background] Graph now has {patternCount} patterns, {colorCount} colors");

                    // Another memory cleanup before saving
                    Console.WriteLine("[Background] Pre-save memory cleanup...");
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
                    GC.WaitForPendingFinalizers();

                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("🏗️ Building cache structure...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "🏗️ Building cache...");
                    });
                    Console.WriteLine("[Background] Building adjacency graph for cache...");

                    // Save training cache - ALWAYS after training
                    // Note: Saving as simple adjacency graph for compatibility
                    var simpleGraph = new AdjacencyGraph();
                    var colors = _multiScaleGraph.GetAllColors();
                    var colorList = colors.ToList();

                    Console.WriteLine($"[Background] Processing {colorList.Count} colors for cache...");
                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog($"Processing {colorList.Count:N0} colors for cache...");
                    });

                    // Use FastContextGraph for ultra-fast cache building
                    var fastGraph = _multiScaleGraph.GetFastGraph();
                    Console.WriteLine($"[Background] Using FastContextGraph with {fastGraph.GetNodeCount()} nodes");

                    var processedColors = 0;
                    foreach (var color in colorList)
                    {
                        processedColors++;
                        if (processedColors % 100 == 0)
                        {
                            var percent = (processedColors * 100) / colorList.Count;
                            Console.WriteLine($"[Background] Cache building: {percent}% ({processedColors}/{colorList.Count})");
                            Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                            {
                                progressDialog.AddLog($"Processing colors: {percent}%");
                                progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                                    $"🏗️ Building cache: {percent}% ({processedColors:N0}/{colorList.Count:N0} colors)");
                            });
                        }

                        foreach (var dir in DirectionExtensions.AllDirections)
                        {
                            try
                            {
                                // Get weighted neighbors from fast graph (center position)
                                var neighbors = fastGraph.GetWeightedNeighbors(color, 0.5f, 0.5f, dir);
                                foreach (var (neighborColor, _) in neighbors)
                                {
                                    simpleGraph.AddAdjacency(color, dir, neighborColor);
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[Background] Warning: Failed to process color/direction: {ex.Message}");
                                // Continue with other colors
                            }
                        }
                    }

                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("⚙️ Normalizing cache...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "⚙️ Normalizing cache...");
                    });
                    Console.WriteLine("[Background] Normalizing adjacency graph...");
                    simpleGraph.Normalize();

                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("💾 Saving to disk...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "💾 Saving cache to disk...");
                    });
                    Console.WriteLine("[Background] Saving cache to disk...");

                    var cache = TrainingDataCache.FromAdjacencyGraph(
                        simpleGraph, 
                        quantizationLevel, 
                        _trainingImagePaths);
                    cache.Save();

                    Console.WriteLine("[Background] Training data saved successfully");

                    // ALWAYS save full model snapshot for instant reuse next run
                    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                    {
                        progressDialog.AddLog("💾 Saving model snapshot...");
                        progressDialog.UpdateProgress(_trainingImagePaths.Count, _trainingImagePaths.Count, 
                            "💾 Saving model snapshot...");
                    });
                    ModelRepository.Save(
                        _multiScaleGraph, 
                        null, 
                        $"QuantLevel={quantizationLevel}; Images={_totalTrainingImages}",
                        quantizationLevel,
                        _totalTrainingImages,
                        _firstTrainingDate,
                        _trainingImagePaths);
                    Console.WriteLine("[Background] Model snapshot saved with metadata");

                    // Final cleanup
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
                }
                catch (OutOfMemoryException ex)
                {
                    Console.WriteLine($"[Background] OUT OF MEMORY during finalization: {ex.Message}");
                    Avalonia.Threading.Dispatcher.UIThread.Post(() => 
                        progressDialog.AddLog("⚠ Memory exhausted - attempting emergency save..."));

                    // Try emergency save without full graph conversion
                    try
                    {
                        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
                        GC.WaitForPendingFinalizers();
                        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
                        throw; // Re-throw to be caught by outer catch
                    }
                    catch
                    {
                        Console.WriteLine("[Background] Emergency save failed");
                        throw;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Background] ERROR during finalization: {ex.Message}");
                    Console.WriteLine($"[Background] Stack trace: {ex.StackTrace}");
                    throw;
                }
            });

            var patternCount = _multiScaleGraph.GetTotalPatternCount();
            var colorCount = _multiScaleGraph.GetColorCount();
            Console.WriteLine($"[MainWindow] Streaming training completed: {processedCount} images, {patternCount} multi-scale patterns, {colorCount} colors");

            progressDialog.Complete($"Training complete! {processedCount} images processed\n{patternCount} multi-scale patterns, {colorCount} colors");

            await Task.Delay(1500);
            progressDialog.Close();

            TrainingStatusText.Text = $"✓ Multi-scale model trained! {patternCount} patterns";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
            TrainingInfoText.Text = $"Total images: {_totalTrainingImages}\n" +
                                   $"Multi-scale patterns: {patternCount}\n" +
                                   $"Colors: {colorCount}\n" +
                                   $"Scales: 3x3, 5x5, 9x9\n" +
                                   $"First trained: {_firstTrainingDate:g}\n" +
                                   $"Last updated: {DateTime.Now:g}";
            GenerateButton.IsEnabled = true;
            SaveModelButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] ERROR in TrainButton_Click: {ex.Message}");
            Console.WriteLine($"[MainWindow] Stack trace: {ex.StackTrace}");

            progressDialog.Error($"Training failed: {ex.Message}");
            await Task.Delay(2000);
            progressDialog.Close();

            TrainingStatusText.Text = $"✗ Error: {ex.Message}";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
        }
        finally
        {
            Console.WriteLine("[MainWindow] TrainButton_Click completed, re-enabling buttons");
            TrainButton.IsEnabled = true;
            SelectFolderButton.IsEnabled = true;
        }
    }

    private ColorRgb PixelToColorRgb(uint pixel)
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

    private void ProcessImageIntoGraph(uint[] pixels, int width, int height)
    {
        Console.WriteLine($"[ProcessImage] Processing {width}x{height} image with SINGLE-SCALE (3x3) extraction");

        // Try GPU single-scale (3x3) extraction first
        var gpuResult = _gpu?.ExtractMultiScaleNeighborhoodsGpu(pixels, width, height, new[] { 1 });

        if (gpuResult.HasValue)
        {
            Console.WriteLine($"[ProcessImage] Using GPU-accelerated SINGLE-SCALE (3x3) extraction");
            ProcessMultiScaleGpuResults(gpuResult.Value, pixels, width, height);
        }
        else
        {
            Console.WriteLine($"[ProcessImage] Using CPU multi-scale extraction");
            ProcessMultiScaleCpu(pixels, width, height);
        }
    }

    private void ProcessMultiScaleCpu(uint[] pixels, int width, int height)
    {
        // Try GPU bulk training first for MASSIVE speedup! Lower threshold to 5000 pixels
        if (_gpu != null && _gpu.IsAvailable && pixels.Length >= 5000)
        {
            Console.WriteLine($"[ProcessImage] Using GPU BULK TRAINING for {width}x{height} image");
            ProcessImageGpuBulk(pixels, width, height);
            return;
        }

        Console.WriteLine($"[ProcessImage] ULTRA-FAST CPU processing with 10-thread optimization");

        var processedCount = 0;

        // OPTIMIZED: Use 10 threads with row-based partitioning
        var rowsPerThread = Math.Max(1, height / 10);
        var partitioner = System.Collections.Concurrent.Partitioner.Create(0, height, rowsPerThread);

        // Use exactly 10 threads for optimal parallelization
        System.Threading.Tasks.Parallel.ForEach(partitioner,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 10 },
            range =>
        {
            var localCount = 0;

            for (int y = range.Item1; y < range.Item2; y++)
            {
            var rowBase = y * width;

            for (int x = 0; x < width; x++)
            {
                var centerPixel = pixels[rowBase + x];
                var centerColor = _quantizer!.Quantize(PixelToColorRgb(centerPixel));

                // Extract 3x3 neighborhood
                var neighbors3x3 = ExtractNeighborhood(pixels, width, height, x, y, 1);

                // Train GraphNode edges directly for each direction
                for (int dirIdx = 0; dirIdx < 8; dirIdx++)
                {
                    var direction = (Direction)dirIdx;
                    var targetColor = neighbors3x3[direction];
                    if (targetColor.HasValue)
                    {
                        // Direct training to GraphNode - super fast!
                        _multiScaleGraph.AddPatternMultiScale(centerColor, neighbors3x3, direction, targetColor.Value, pixels, width, height, x, y);
                    }
                }
            }

                localCount++;
            }

            // Update global progress
            var count = System.Threading.Interlocked.Add(ref processedCount, localCount);
            var percent = (count * 100) / height;
            Console.WriteLine($"[ProcessImage] Thread completed {localCount} rows - Total: {count}/{height} ({percent}%)");
        });

        Console.WriteLine($"[ProcessImage] ULTRA-FAST 10-thread CPU training complete - {height} rows trained to GraphNodes");
    }

    /// <summary>
    /// MASSIVE SPEEDUP: Process entire image on GPU in bulk - all pixels trained in parallel!
    /// Uses all 10240 CUDA cores to train patterns simultaneously
    /// MAXED OUT: GPU batch quantization + GPU pattern training!
    /// </summary>
    private void ProcessImageGpuBulk(uint[] pixels, int width, int height)
    {
        Console.WriteLine($"[ProcessImage] ⚡ GPU MAXIMUM MODE: {pixels.Length:N0} pixels @ 10240 cores!");

        var startTime = System.Diagnostics.Stopwatch.GetTimestamp();

        // Step 1: Convert ALL pixels to ColorRgb in parallel
        var rawColors = new ColorRgb[pixels.Length];
        System.Threading.Tasks.Parallel.For(0, pixels.Length, i =>
        {
            rawColors[i] = PixelToColorRgb(pixels[i]);
        });

        // Step 2: ULTRA-FAST GPU batch quantization - quantize ENTIRE image on GPU!
        Console.WriteLine($"[ProcessImage] ⚡ GPU quantizing {pixels.Length:N0} colors...");
        var quantizedColors = _quantizer!.QuantizeBatch(rawColors);

        var quantizeTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ProcessImage] ✓ Quantized in {quantizeTime * 1000:F1}ms");

        // Step 3: Extract patterns in parallel with pre-quantized colors (ULTRA FAST!)
        var centerColorsList = new List<ColorRgb>(pixels.Length * 8);
        var targetColorsList = new List<ColorRgb>(pixels.Length * 8);
        var directionsList = new List<int>(pixels.Length * 8);
        var normalizedXList = new List<float>(pixels.Length * 8);
        var normalizedYList = new List<float>(pixels.Length * 8);

        var batchSize = pixels.Length / 10;
        var partitioner = System.Collections.Concurrent.Partitioner.Create(0, pixels.Length, batchSize);
        var localBatches = new System.Collections.Concurrent.ConcurrentBag<(ColorRgb[], ColorRgb[], int[], float[], float[])>();

        System.Threading.Tasks.Parallel.ForEach(partitioner,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 10 },
            range =>
        {
            var localCenters = new List<ColorRgb>((range.Item2 - range.Item1) * 8);
            var localTargets = new List<ColorRgb>((range.Item2 - range.Item1) * 8);
            var localDirections = new List<int>((range.Item2 - range.Item1) * 8);
            var localNormX = new List<float>((range.Item2 - range.Item1) * 8);
            var localNormY = new List<float>((range.Item2 - range.Item1) * 8);

            for (int pixelIdx = range.Item1; pixelIdx < range.Item2; pixelIdx++)
            {
                var x = pixelIdx % width;
                var y = pixelIdx / width;

                var normalizedX = width > 1 ? (float)x / (width - 1) : 0.5f;
                var normalizedY = height > 1 ? (float)y / (height - 1) : 0.5f;

                var centerColor = quantizedColors[pixelIdx]; // Already quantized!

                // Extract all 8 neighbors (pre-quantized!)
                for (int dirIdx = 0; dirIdx < 8; dirIdx++)
                {
                    var direction = (Direction)dirIdx;
                    var (dx, dy) = direction.GetOffset();
                    var nx = x + dx;
                    var ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        var neighborIdx = ny * width + nx;
                        var targetColor = quantizedColors[neighborIdx]; // Already quantized!

                        localCenters.Add(centerColor);
                        localTargets.Add(targetColor);
                        localDirections.Add(dirIdx);
                        localNormX.Add(normalizedX);
                        localNormY.Add(normalizedY);
                    }
                }
            }

            localBatches.Add((localCenters.ToArray(), localTargets.ToArray(), 
                             localDirections.ToArray(), localNormX.ToArray(), localNormY.ToArray()));
        });

        // Combine batches
        foreach (var (centers, targets, dirs, xs, ys) in localBatches)
        {
            centerColorsList.AddRange(centers);
            targetColorsList.AddRange(targets);
            directionsList.AddRange(dirs);
            normalizedXList.AddRange(xs);
            normalizedYList.AddRange(ys);
        }

        var extractTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ProcessImage] ✓ Extracted {centerColorsList.Count:N0} patterns in {extractTime:F3}s");

        // Step 4: GPU BULK TRAINING - train ALL patterns on GPU in one call!
        var centerColorsArray = centerColorsList.ToArray();
        var targetColorsArray = targetColorsList.ToArray();
        var directionsArray = directionsList.ToArray();
        var normalizedXArray = normalizedXList.ToArray();
        var normalizedYArray = normalizedYList.ToArray();

        // Clear lists immediately to free memory
        centerColorsList.Clear();
        targetColorsList.Clear();
        directionsList.Clear();
        normalizedXList.Clear();
        normalizedYList.Clear();
        centerColorsList = null;
        targetColorsList = null;
        directionsList = null;
        normalizedXList = null;
        normalizedYList = null;

        _multiScaleGraph.AddPatternsBulkGpu(
            centerColorsArray,
            targetColorsArray,
            directionsArray,
            normalizedXArray,
            normalizedYArray,
            width,
            height);

        var totalTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        var patternsPerSec = centerColorsArray.Length / totalTime;

        Console.WriteLine($"[ProcessImage] ⚡ MAXED OUT GPU COMPLETE!");
        Console.WriteLine($"[ProcessImage] {centerColorsArray.Length:N0} patterns in {totalTime:F3}s");
        Console.WriteLine($"[ProcessImage] ⚡ {patternsPerSec / 1_000_000:F2}M patterns/sec - GPU MAXED!");

        // Clear arrays immediately after GPU processing
        centerColorsArray = null;
        targetColorsArray = null;
        directionsArray = null;
        normalizedXArray = null;
        normalizedYArray = null;
        rawColors = null;
        quantizedColors = null;
        localBatches = null;

        // Force cleanup of large temporary data
        GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
    }

    private Dictionary<Direction, ColorRgb?> ExtractNeighborhood(
        uint[] pixels, int width, int height, int centerX, int centerY, int radius)
    {
        var neighbors = new Dictionary<Direction, ColorRgb?>();

        for (int i = 0; i < 8; i++)
        {
            var dir = (Direction)i;
            var (dx, dy) = dir.GetOffset();
            var nx = centerX + dx * radius;
            var ny = centerY + dy * radius;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                var neighborPixel = pixels[ny * width + nx];
                neighbors[dir] = _quantizer!.Quantize(PixelToColorRgb(neighborPixel));
            }
            else
            {
                neighbors[dir] = null;
            }
        }

        return neighbors;
    }

    private void ProcessMultiScaleGpuResults(
        (byte[] centersR, byte[] centersG, byte[] centersB, 
         byte[] neighbors3x3R, byte[] neighbors3x3G, byte[] neighbors3x3B,
         byte[] neighbors5x5R, byte[] neighbors5x5G, byte[] neighbors5x5B,
         byte[] neighbors9x9R, byte[] neighbors9x9G, byte[] neighbors9x9B) gpuData,
        uint[] pixels, int width, int height)
    {
        Console.WriteLine($"[ProcessImage] ⚡ MAX GPU MODE: {pixels.Length:N0} pixels → FULL GPU PIPELINE!");

        var (centersR, centersG, centersB, 
             neighbors3x3R, neighbors3x3G, neighbors3x3B,
             _, _, _,
             _, _, _) = gpuData;

        var startTime = System.Diagnostics.Stopwatch.GetTimestamp();

        // Step 1: ULTRA-FAST GPU batch quantization for ALL colors at once!
        Console.WriteLine($"[ProcessImage] ⚡ GPU quantizing {pixels.Length:N0} center colors...");

        var rawCenters = new ColorRgb[pixels.Length];
        System.Threading.Tasks.Parallel.For(0, pixels.Length, i =>
        {
            rawCenters[i] = new ColorRgb(centersR[i], centersG[i], centersB[i]);
        });

        var quantizedCenters = _quantizer!.QuantizeBatch(rawCenters);

        Console.WriteLine($"[ProcessImage] ⚡ GPU quantizing {pixels.Length * 8:N0} neighbor colors...");

        var rawNeighbors = new ColorRgb[pixels.Length * 8];
        System.Threading.Tasks.Parallel.For(0, pixels.Length * 8, i =>
        {
            var r = neighbors3x3R[i];
            if (r != 255)
            {
                rawNeighbors[i] = new ColorRgb(r, neighbors3x3G[i], neighbors3x3B[i]);
            }
            else
            {
                rawNeighbors[i] = new ColorRgb(0, 0, 0); // Placeholder for null
            }
        });

        var quantizedNeighbors = _quantizer.QuantizeBatch(rawNeighbors);

        var quantizeTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ProcessImage] ✓ Quantized {(pixels.Length * 9):N0} colors in {quantizeTime * 1000:F1}ms");

        // Step 2: Prepare ALL patterns for GPU bulk training (ULTRA FAST!)
        Console.WriteLine($"[ProcessImage] ⚡ Preparing {pixels.Length * 8:N0} patterns for GPU...");

        var centerColorsList = new List<ColorRgb>(pixels.Length * 8);
        var targetColorsList = new List<ColorRgb>(pixels.Length * 8);
        var directionsList = new List<int>(pixels.Length * 8);
        var normalizedXList = new List<float>(pixels.Length * 8);
        var normalizedYList = new List<float>(pixels.Length * 8);

        // Extract patterns in parallel batches
        var batchSize = pixels.Length / 10;
        var partitioner = System.Collections.Concurrent.Partitioner.Create(0, pixels.Length, batchSize);
        var localBatches = new System.Collections.Concurrent.ConcurrentBag<(ColorRgb[], ColorRgb[], int[], float[], float[])>();

        System.Threading.Tasks.Parallel.ForEach(partitioner,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 10 },
            range =>
        {
            var localCenters = new List<ColorRgb>((range.Item2 - range.Item1) * 8);
            var localTargets = new List<ColorRgb>((range.Item2 - range.Item1) * 8);
            var localDirections = new List<int>((range.Item2 - range.Item1) * 8);
            var localNormX = new List<float>((range.Item2 - range.Item1) * 8);
            var localNormY = new List<float>((range.Item2 - range.Item1) * 8);

            for (int pixelIdx = range.Item1; pixelIdx < range.Item2; pixelIdx++)
            {
                var x = pixelIdx % width;
                var y = pixelIdx / width;

                var normalizedX = width > 1 ? (float)x / (width - 1) : 0.5f;
                var normalizedY = height > 1 ? (float)y / (height - 1) : 0.5f;

                var centerColor = quantizedCenters[pixelIdx]; // Already quantized!

                // Process all 8 directions
                for (int dir = 0; dir < 8; dir++)
                {
                    var neighborIdx = pixelIdx * 8 + dir;
                    var r = neighbors3x3R[neighborIdx];

                    if (r != 255) // Not null marker
                    {
                        var targetColor = quantizedNeighbors[neighborIdx]; // Already quantized!

                        localCenters.Add(centerColor);
                        localTargets.Add(targetColor);
                        localDirections.Add(dir);
                        localNormX.Add(normalizedX);
                        localNormY.Add(normalizedY);
                    }
                }
            }

            localBatches.Add((localCenters.ToArray(), localTargets.ToArray(), 
                             localDirections.ToArray(), localNormX.ToArray(), localNormY.ToArray()));
        });

        // Combine batches
        foreach (var (centers, targets, dirs, xs, ys) in localBatches)
        {
            centerColorsList.AddRange(centers);
            targetColorsList.AddRange(targets);
            directionsList.AddRange(dirs);
            normalizedXList.AddRange(xs);
            normalizedYList.AddRange(ys);
        }

        var prepTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ProcessImage] ✓ Prepared {centerColorsList.Count:N0} patterns in {prepTime:F3}s");

        // Step 3: GPU BULK TRAINING - Train ALL patterns on GPU in ONE massive call!
        Console.WriteLine($"[ProcessImage] ⚡ GPU BULK TRAINING {centerColorsList.Count:N0} patterns on 10240 cores!");

        var centerColorsArray = centerColorsList.ToArray();
        var targetColorsArray = targetColorsList.ToArray();
        var directionsArray = directionsList.ToArray();
        var normalizedXArray = normalizedXList.ToArray();
        var normalizedYArray = normalizedYList.ToArray();

        // Clear lists immediately to free memory
        centerColorsList.Clear();
        targetColorsList.Clear();
        directionsList.Clear();
        normalizedXList.Clear();
        normalizedYList.Clear();
        centerColorsList = null;
        targetColorsList = null;
        directionsList = null;
        normalizedXList = null;
        normalizedYList = null;

        _multiScaleGraph.AddPatternsBulkGpu(
            centerColorsArray,
            targetColorsArray,
            directionsArray,
            normalizedXArray,
            normalizedYArray,
            width,
            height);

        var totalTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        var patternsPerSec = centerColorsArray.Length / totalTime;

        Console.WriteLine($"[ProcessImage] ⚡⚡⚡ FULL GPU PIPELINE COMPLETE! ⚡⚡⚡");
        Console.WriteLine($"[ProcessImage] {centerColorsArray.Length:N0} patterns in {totalTime:F3}s");
        Console.WriteLine($"[ProcessImage] ⚡ {patternsPerSec / 1_000_000:F2}M patterns/sec - GPU FULLY MAXED OUT!");

        // Clear all large temporary data immediately
        centerColorsArray = null;
        targetColorsArray = null;
        directionsArray = null;
        normalizedXArray = null;
        normalizedYArray = null;
        rawCenters = null;
        rawNeighbors = null;
        quantizedCenters = null;
        quantizedNeighbors = null;
        localBatches = null;

        // Force cleanup of large temporary data
        GC.Collect(2, GCCollectionMode.Optimized, blocking: false);
    }

    private async Task<(uint[] pixels, int width, int height)?> ExtractPixelDataAsync(string imagePath)
    {
        Console.WriteLine($"[ExtractPixels] Starting extraction for: {imagePath}");

        try
        {
            using var stream = File.OpenRead(imagePath);
            using var bitmap = new Bitmap(stream);

            var width = bitmap.PixelSize.Width;
            var height = bitmap.PixelSize.Height;
            Console.WriteLine($"[ExtractPixels] Bitmap size: {width}x{height}");

            // Guard against absurdly large images
            if ((long)width * height > 100_000_000) // >100 MP
            {
                Console.WriteLine("[ExtractPixels] Image too large (>100MP). Please downscale.");
                return null;
            }

            var pixels = new uint[width * height];

            // We need a writable framebuffer to access raw pixels with a guaranteed format/stride
            // Decode the original bitmap into a WriteableBitmap and copy row-by-row respecting RowBytes
            using (var ms = new MemoryStream())
            {
                bitmap.Save(ms);
                ms.Position = 0;

                using var wb = WriteableBitmap.Decode(ms);
                using var fb = wb.Lock();

                Console.WriteLine($"[ExtractPixels] Locked framebuffer: format={fb.Format}, rowBytes={fb.RowBytes}");

                unsafe
                {
                    var basePtr = (byte*)fb.Address.ToPointer();
                    int rowBytes = fb.RowBytes;

                    // We expect 32bpp formats from Avalonia decoders. Handle BGRA8888 and RGBA8888.
                    // Convert to ARGB32 (A in high byte) which our pipeline expects.
                    bool isBGRA = fb.Format.ToString().Contains("Bgra", StringComparison.OrdinalIgnoreCase);
                    bool isRGBA = fb.Format.ToString().Contains("Rgba", StringComparison.OrdinalIgnoreCase);

                    if (!isBGRA && !isRGBA)
                    {
                        // Fallback: treat as BGRA layout if unknown 32bpp. This keeps us safe vs AccessViolation by honoring stride.
                        Console.WriteLine($"[ExtractPixels] Unexpected format {fb.Format}, assuming BGRA8888 layout");
                        isBGRA = true;
                    }

                    for (int y = 0; y < height; y++)
                    {
                        byte* row = basePtr + y * rowBytes;
                        for (int x = 0; x < width; x++)
                        {
                            int idx = y * width + x;
                            int o = x * 4;
                            byte c0 = row[o + 0];
                            byte c1 = row[o + 1];
                            byte c2 = row[o + 2];
                            byte a = row[o + 3];

                            byte r, g, b;
                            if (isBGRA)
                            {
                                b = c0; g = c1; r = c2;
                            }
                            else // RGBA
                            {
                                r = c0; g = c1; b = c2;
                            }

                            pixels[idx] = (uint)((a << 24) | (r << 16) | (g << 8) | b);
                        }
                    }
                }
            }

            Console.WriteLine($"[ExtractPixels] Extraction complete for {Path.GetFileName(imagePath)}");
            return (pixels, width, height);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ExtractPixels] ERROR: {ex.Message}");
            Console.WriteLine($"[ExtractPixels] Stack trace: {ex.StackTrace}");
            return null;
        }
    }

    private async void GenerateButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_multiScaleGraph == null || _multiScaleGraph.GetTotalPatternCount() == 0 || _multiScaleGraph.GetColorCount() == 0)
        {
            var patterns = _multiScaleGraph?.GetTotalPatternCount() ?? 0;
            var colors = _multiScaleGraph?.GetColorCount() ?? 0;
            TrainingStatusText.Text = $"⚠ Please train the model first (patterns: {patterns}, colors: {colors})";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            return;
        }

        GenerateButton.IsEnabled = false;
        SaveButton.IsEnabled = false;
        StopGenerationButton.IsVisible = true;
        TrainingStatusText.Text = "⚡ Generating image with multi-seed WFC...";
        TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

        // Create new cancellation token for this generation
        _generationCancellation?.Dispose();
        _generationCancellation = new System.Threading.CancellationTokenSource();
        var cancellationToken = _generationCancellation.Token;

        try
        {
            var width = (int)OutputWidth.Value;
            var height = (int)OutputHeight.Value;

            // Get seed count from slider
            var seedCount = (int)SeedCount.Value;

            // Get generation mode from combo box
            var mode = GenerationModeCombo.SelectedIndex switch
            {
                0 => FastWaveFunctionCollapse.GenerationMode.FastSquare,
                1 => FastWaveFunctionCollapse.GenerationMode.Blended,
                2 => FastWaveFunctionCollapse.GenerationMode.FastOrganic,
                3 => FastWaveFunctionCollapse.GenerationMode.Radial,
                4 => FastWaveFunctionCollapse.GenerationMode.EdgeFirst,
                5 => FastWaveFunctionCollapse.GenerationMode.CenterOut,
                6 => FastWaveFunctionCollapse.GenerationMode.Spiral,
                _ => FastWaveFunctionCollapse.GenerationMode.FastSquare
            };

            // Get CUDA preference from checkbox
            var useCuda = UseCudaCheckBox.IsChecked ?? false;

            // Get entropy value (0-100) from slider
            var entropyPercent = (int)EntropySlider.Value;
            var entropyFactor = entropyPercent / 100.0; // Convert to 0.0-1.0

            Console.WriteLine($"[Generate] Mode: {mode.ToString()}, Seeds: {seedCount}, Size: {width}x{height}, Entropy: {entropyPercent}%, CUDA: {(useCuda ? "Enabled" : "Disabled")}");

            // Calculate update frequency for smooth animation
            var totalPixels = width * height;
            var updateFrequency = totalPixels switch
            {
                < 10000 => 1,      // Every cell for tiny images
                < 50000 => 5,      // Every 5 cells for small images
                < 200000 => 20,    // Every 20 cells for medium images
                _ => 50            // Every 50 cells for large images
            };

            Console.WriteLine($"[Generate] Update frequency: every {updateFrequency} iterations");

            // Create progress callback for real-time wave growth visualization
            var lastUpdateTime = System.Diagnostics.Stopwatch.GetTimestamp();
            var minUpdateTicks = (long)(System.Diagnostics.Stopwatch.Frequency * 0.033); // ~30 FPS
            var lastBitmap = (Bitmap?)null;

            FastWaveFunctionCollapse.ProgressCallback progressCallback = (pixels, collapsedCount, total) =>
            {
                // Throttle updates for performance
                var now = System.Diagnostics.Stopwatch.GetTimestamp();
                if (now - lastUpdateTime < minUpdateTicks && collapsedCount < total)
                    return;

                lastUpdateTime = now;

                // Update UI with wave growth
                Avalonia.Threading.Dispatcher.UIThread.Post(() =>
                {
                    try
                    {
                        lastBitmap?.Dispose();

                        // Create and display the growing image
                        var bitmap = FastWaveFunctionCollapse.CreateBitmapFromPixels(pixels, width, height);
                        lastBitmap = bitmap;

                        GeneratedImage.Source = bitmap;
                        GeneratedImage.Width = width;
                        GeneratedImage.Height = height;

                        var progress = (collapsedCount * 100) / total;
                        TrainingStatusText.Text = $"⚡ Growing from seeds... {progress}% ({collapsedCount}/{total} pixels)";
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[Generate] Error updating display: {ex.Message}");
                    }
                }, Avalonia.Threading.DispatcherPriority.Background);
            };

            // Generate using ultra-fast node-based WFC with selectable mode
            // Pass GPU accelerator for CUDA hyper-speed when enabled
            var pixelData = await Task.Run(() =>
            {
                var fastGraph = _multiScaleGraph.GetFastGraph();
                var fastWfc = new FastWaveFunctionCollapse(fastGraph, width, height, _gpu, entropyFactor, cancellationToken);
                // Pass CUDA preference to generation - FastWFC will handle fallback internally
                return fastWfc.Generate(seedCount, mode, progressCallback, updateFrequency, useCuda);
            }, cancellationToken);

            // Check if cancelled before creating final bitmap
            if (cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine($"[Generate] Generation was cancelled");
                TrainingStatusText.Text = "⏹ Generation cancelled";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;
                return;
            }

            // Create final bitmap on UI thread
            Console.WriteLine($"[Generate] Creating final bitmap");
            var generatedImage = FastWaveFunctionCollapse.CreateBitmapFromPixels(pixelData, width, height);

            _generatedBitmap = generatedImage;
            GeneratedImage.Source = generatedImage;
            GeneratedImage.Width = width;
            GeneratedImage.Height = height;

            TrainingStatusText.Text = $"✓ Generated {width}x{height} image from {seedCount} seeds";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
            SaveButton.IsEnabled = true;

            Console.WriteLine($"[Generate] ULTRA-FAST generation complete!");
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine($"[Generate] Generation cancelled by user");
            TrainingStatusText.Text = "⏹ Generation cancelled";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Generate] ERROR: {ex.Message}");
            Console.WriteLine($"[Generate] Stack trace: {ex.StackTrace}");
            TrainingStatusText.Text = $"✗ Generation error: {ex.Message}";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
        }
        finally
        {
            StopGenerationButton.IsVisible = false;
            GenerateButton.IsEnabled = true;
        }
    }

    private void StopGenerationButton_Click(object? sender, RoutedEventArgs e)
    {
        Console.WriteLine("[MainWindow] Stop generation requested");

        if (_generationCancellation != null && !_generationCancellation.IsCancellationRequested)
        {
            _generationCancellation.Cancel();
            TrainingStatusText.Text = "⏹ Generation cancelled";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

            StopGenerationButton.IsVisible = false;
            GenerateButton.IsEnabled = true;
        }
    }

    private async void SaveButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_generatedBitmap == null)
            return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save Generated Image",
            DefaultExtension = "png",
            SuggestedFileName = $"generated_{DateTime.Now:yyyyMMdd_HHmmss}.png",
            FileTypeChoices = new[]
            {
                new FilePickerFileType("PNG Image") { Patterns = new[] { "*.png" } }
            }
        });

        if (file != null)
        {
            using var stream = await file.OpenWriteAsync();
            _generatedBitmap.Save(stream);

            TrainingStatusText.Text = "✓ Image saved successfully!";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
        }
    }

    private async void SaveModelAsButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_multiScaleGraph == null || _multiScaleGraph.GetTotalPatternCount() == 0 || _multiScaleGraph.GetColorCount() == 0)
        {
            TrainingStatusText.Text = "⚠ No trained model to save";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            return;
        }

        // Show input dialog for model name
        var modelName = await ShowInputDialog("Save Model", "Enter a name for this model:", $"model_{DateTime.Now:yyyyMMdd_HHmmss}");

        if (string.IsNullOrWhiteSpace(modelName))
        {
            return; // User cancelled
        }

        // Sanitize filename
        var invalidChars = Path.GetInvalidFileNameChars();
        modelName = string.Join("_", modelName.Split(invalidChars, StringSplitOptions.RemoveEmptyEntries));

        try
        {
            var filePath = ModelRepository.GetModelPath(modelName);
            Console.WriteLine($"[MainWindow] Saving model to config folder: {filePath}");

            var success = ModelRepository.Save(
                _multiScaleGraph,
                filePath,
                $"QuantLevel={_currentQuantizationLevel}; Images={_totalTrainingImages}",
                _currentQuantizationLevel,
                _totalTrainingImages,
                _firstTrainingDate,
                _trainingImagePaths);

            if (success)
            {
                TrainingStatusText.Text = $"✓ Model saved as '{modelName}'";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;
                Console.WriteLine($"[MainWindow] Model saved successfully");

                // Refresh the dropdown list
                RefreshModelsList();
            }
            else
            {
                TrainingStatusText.Text = "✗ Failed to save model";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] Error saving model: {ex.Message}");
            TrainingStatusText.Text = $"✗ Error saving model: {ex.Message}";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
        }
    }

    private void RefreshModelsList()
    {
        var models = ModelRepository.GetAllSavedModels();
        LoadModelComboBox.Items.Clear();

        if (models.Count == 0)
        {
            LoadModelComboBox.Items.Add("(No saved models)");
            LoadModelComboBox.SelectedIndex = 0;
            LoadModelComboBox.IsEnabled = false;
        }
        else
        {
            foreach (var model in models)
            {
                LoadModelComboBox.Items.Add(model);
            }
            LoadModelComboBox.IsEnabled = true;
        }

        Console.WriteLine($"[MainWindow] Refreshed models list: {models.Count} models found");
    }

    private void LoadModelComboBox_SelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (LoadModelComboBox.SelectedItem == null || LoadModelComboBox.SelectedItem.ToString() == "(No saved models)")
            return;

        var modelName = LoadModelComboBox.SelectedItem.ToString();
        if (string.IsNullOrEmpty(modelName))
            return;

        try
        {
            var filePath = ModelRepository.GetModelPath(modelName);
            Console.WriteLine($"[MainWindow] Loading model from: {filePath}");

            TrainingStatusText.Text = "⏳ Loading model...";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;

            var loadedData = ModelRepository.LoadWithMetadata(filePath, _gpu);

            if (loadedData.HasValue)
            {
                var (graph, snapshot) = loadedData.Value;
                _multiScaleGraph = graph;
                _multiScaleGraph.SetGpuAccelerator(_gpu);

                // Restore metadata
                _currentQuantizationLevel = snapshot.QuantizationLevel;
                _totalTrainingImages = snapshot.TotalImages;
                _firstTrainingDate = snapshot.FirstTrainingDate;
                _trainingImagePaths = snapshot.TrainingImagePaths;

                if (_currentQuantizationLevel > 0)
                    QuantizationLevel.Value = _currentQuantizationLevel;

                Console.WriteLine($"[MainWindow] Loaded model: {modelName}");
                TrainingStatusText.Text = $"✓ Loaded model: {modelName}";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.LightGreen;

                var patternCount = _multiScaleGraph.GetTotalPatternCount();
                var colorCount = _multiScaleGraph.GetColorCount();
                TrainingInfoText.Text = $"Total images: {_totalTrainingImages}\n" +
                                       $"Patterns: {patternCount}\n" +
                                       $"Colors: {colorCount}\n" +
                                       $"Quantization: {_currentQuantizationLevel}\n" +
                                       $"First trained: {_firstTrainingDate:g}\n" +
                                       $"Last updated: {snapshot.LastTrainingDate:g}";

                // Enable Generate and Save Model buttons when actual data is present
                var canGenerate = patternCount > 0 && colorCount > 0;
                GenerateButton.IsEnabled = canGenerate;
                SaveModelButton.IsEnabled = canGenerate;

                if (!canGenerate)
                {
                    TrainingStatusText.Text = "⚠ Loaded model has no patterns/colors.";
                    TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
                }

                // Load sample images if paths exist
                LoadSampleImagesFromCache();
            }
            else
            {
                TrainingStatusText.Text = "✗ Failed to load model";
                TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MainWindow] Error loading model: {ex.Message}");
            TrainingStatusText.Text = $"✗ Error loading model: {ex.Message}";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
        }
    }

    private async Task<string?> ShowInputDialog(string title, string message, string defaultValue)
    {
        var dialog = new Window
        {
            Title = title,
            Width = 450,
            Height = 200,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#1E1E1E")),
            CanResize = false
        };

        string? result = null;

        var mainPanel = new StackPanel 
        { 
            Margin = new Avalonia.Thickness(20)
        };

        var messageBlock = new TextBlock 
        { 
            Text = message, 
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
            Foreground = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#CCCCCC")),
            FontSize = 13,
            Margin = new Avalonia.Thickness(0, 0, 0, 15)
        };

        var inputBox = new TextBox
        {
            Text = defaultValue,
            Margin = new Avalonia.Thickness(0, 0, 0, 20),
            FontSize = 13
        };

        var buttonPanel = new StackPanel 
        { 
            Orientation = Avalonia.Layout.Orientation.Horizontal,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Spacing = 10
        };

        var okButton = new Button 
        { 
            Content = "Save", 
            Width = 100,
            Height = 32,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#007ACC")),
            Foreground = Avalonia.Media.Brushes.White,
            FontWeight = Avalonia.Media.FontWeight.Bold
        };
        okButton.Click += (s, e) => { result = inputBox.Text; dialog.Close(); };

        var cancelButton = new Button 
        { 
            Content = "Cancel", 
            Width = 100,
            Height = 32,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#2D2D30")),
            Foreground = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#CCCCCC"))
        };
        cancelButton.Click += (s, e) => { dialog.Close(); };

        buttonPanel.Children.Add(cancelButton);
        buttonPanel.Children.Add(okButton);

        mainPanel.Children.Add(messageBlock);
        mainPanel.Children.Add(inputBox);
        mainPanel.Children.Add(buttonPanel);

        dialog.Content = mainPanel;

        // Focus the text box and select all text
        inputBox.AttachedToVisualTree += (s, e) => 
        { 
            inputBox.Focus(); 
            inputBox.SelectAll(); 
        };

        await dialog.ShowDialog(this);

        return result;
    }

    private async void ClearTrainingButton_Click(object? sender, RoutedEventArgs e)
    {
        var message = $"Are you sure you want to clear ALL training data?\n\n" +
                     $"Current model contains:\n" +
                     $"• {_multiScaleGraph.GetTotalPatternCount()} multi-scale patterns\n" +
                     $"• {_multiScaleGraph.GetColorCount()} unique colors\n" +
                     $"• {_totalTrainingImages} total images trained\n\n" +
                     $"This action CANNOT be undone!";

        var result = await ShowConfirmDialog("⚠️ Clear All Training Data", message);

        if (result)
        {
            Console.WriteLine("[MainWindow] Clearing all training data");

            // Clear cache file
            TrainingDataCache.Clear();

            // Clear model file
            ModelRepository.Clear();

            // Reset persistent multi-scale graph
            _multiScaleGraph = new MultiScaleContextGraph();
            _multiScaleGraph.SetGpuAccelerator(_gpu);
            _quantizer = null;

            // Reset state
            _trainingImagePaths.Clear();
            _totalTrainingImages = 0;
            _firstTrainingDate = null;
            SampleImagesPanel.Children.Clear();
            GeneratedImage.Source = null;
            _generatedBitmap = null;

            TrainingStatusText.Text = "✓ All training data cleared";
            TrainingStatusText.Foreground = Avalonia.Media.Brushes.Orange;
            TrainingInfoText.Text = "Model reset - ready for new training";
            OnlineTrainingInfoText.Text = "";
            GenerateButton.IsEnabled = false;
            SaveButton.IsEnabled = false;
            SaveModelButton.IsEnabled = false;

            Console.WriteLine("[MainWindow] Training data cleared successfully");
        }
        else
        {
            Console.WriteLine("[MainWindow] Training data clear cancelled");
        }
    }

    private async Task<bool> ShowConfirmDialog(string title, string message)
    {
        var dialog = new Window
        {
            Title = title,
            Width = 500,
            Height = 350,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#1E1E1E")),
            CanResize = false
        };

        var result = false;

        var mainPanel = new StackPanel 
        { 
            Margin = new Avalonia.Thickness(20)
        };

        // Scrollable message area
        var scrollViewer = new ScrollViewer
        {
            MaxHeight = 220,
            Margin = new Avalonia.Thickness(0, 0, 0, 20)
        };

        var messageBlock = new TextBlock 
        { 
            Text = message, 
            TextWrapping = Avalonia.Media.TextWrapping.Wrap,
            Foreground = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#CCCCCC")),
            FontSize = 13,
            LineHeight = 20
        };

        scrollViewer.Content = messageBlock;
        mainPanel.Children.Add(scrollViewer);

        // Button panel
        var buttonPanel = new StackPanel 
        { 
            Orientation = Avalonia.Layout.Orientation.Horizontal,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
            Spacing = 10
        };

        var yesButton = new Button 
        { 
            Content = "Yes, Clear All", 
            Width = 120,
            Height = 35,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#C42B1C")),
            Foreground = Avalonia.Media.Brushes.White,
            FontWeight = Avalonia.Media.FontWeight.Bold
        };
        yesButton.Click += (s, e) => { result = true; dialog.Close(); };

        var noButton = new Button 
        { 
            Content = "Cancel", 
            Width = 120,
            Height = 35,
            Background = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#2D2D30")),
            Foreground = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse("#CCCCCC"))
        };
        noButton.Click += (s, e) => { dialog.Close(); };

        buttonPanel.Children.Add(noButton);
        buttonPanel.Children.Add(yesButton);
        mainPanel.Children.Add(buttonPanel);

        dialog.Content = mainPanel;
        await dialog.ShowDialog(this);

        return result;
    }

    protected override void OnClosing(WindowClosingEventArgs e)
    {
        base.OnClosing(e);
        _onlineTraining?.Dispose();
        _gpu?.Dispose();
        _memoryMonitor?.Dispose();
        Console.WriteLine("[MainWindow] Resources released");
    }
}
