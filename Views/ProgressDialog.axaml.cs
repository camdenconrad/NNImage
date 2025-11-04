using Avalonia.Controls;
using Avalonia.Threading;
using Avalonia.Markup.Xaml;
using System;
using System.Text;

namespace NNImage.Views;

public partial class ProgressDialog : Window
{
    private readonly StringBuilder _logBuilder = new();
    private TextBlock _titleText;
    private ProgressBar _progressBar;
    private TextBlock _logText;
    private TextBlock _statusText;
    private TextBlock _etaText;

    // ETA calculation fields
    private DateTime _startTime;
    private int _lastProcessedCount = 0;
    private DateTime _lastUpdateTime;

    public ProgressDialog()
    {
        InitializeComponent();

        // Manually find controls
        _titleText = this.FindControl<TextBlock>("TitleText") ?? new TextBlock();
        _progressBar = this.FindControl<ProgressBar>("ProgressBar") ?? new ProgressBar();
        _logText = this.FindControl<TextBlock>("LogText") ?? new TextBlock();
        _statusText = this.FindControl<TextBlock>("StatusText") ?? new TextBlock();
        _etaText = this.FindControl<TextBlock>("EtaText") ?? new TextBlock();

        // Initialize timing
        _startTime = DateTime.Now;
        _lastUpdateTime = DateTime.Now;
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public void UpdateProgress(int current, int total, string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var percentage = total > 0 ? (current * 100.0 / total) : 0;
            _progressBar.Value = percentage;
            _statusText.Text = message;

            // Calculate and display ETA
            if (current > 0 && current < total)
            {
                var now = DateTime.Now;
                var elapsed = now - _startTime;
                var itemsProcessed = current - _lastProcessedCount;

                // Calculate rate based on recent progress (more accurate)
                if (itemsProcessed > 0)
                {
                    var timeSinceLastUpdate = now - _lastUpdateTime;
                    var itemsPerSecond = itemsProcessed / timeSinceLastUpdate.TotalSeconds;

                    // Smooth out rate with running average
                    var remainingItems = total - current;
                    var estimatedSecondsRemaining = itemsPerSecond > 0 ? remainingItems / itemsPerSecond : 0;

                    if (estimatedSecondsRemaining > 0 && estimatedSecondsRemaining < 86400) // Less than 24 hours
                    {
                        var eta = TimeSpan.FromSeconds(estimatedSecondsRemaining);
                        var etaString = eta.TotalHours >= 1 
                            ? $"{(int)eta.TotalHours}h {eta.Minutes}m {eta.Seconds}s"
                            : eta.TotalMinutes >= 1
                                ? $"{(int)eta.TotalMinutes}m {eta.Seconds}s"
                                : $"{(int)eta.TotalSeconds}s";

                        _etaText.Text = $"⏱ ETA: {etaString} remaining | Elapsed: {elapsed.Hours:D2}:{elapsed.Minutes:D2}:{elapsed.Seconds:D2} | Speed: {itemsPerSecond:F1} images/sec";
                    }

                    _lastProcessedCount = current;
                    _lastUpdateTime = now;
                }
            }
            else if (current >= total)
            {
                var totalElapsed = DateTime.Now - _startTime;
                _etaText.Text = $"✓ Completed in {totalElapsed.Hours:D2}:{totalElapsed.Minutes:D2}:{totalElapsed.Seconds:D2}";
            }

            var logMessage = $"[{DateTime.Now:HH:mm:ss}] {message}";
            _logBuilder.AppendLine(logMessage);
            _logText.Text = _logBuilder.ToString();

            Console.WriteLine(logMessage);
        });
    }

    public void SetTitle(string title)
    {
        Dispatcher.UIThread.Post(() =>
        {
            _titleText.Text = title;
        });
    }

    public void AddLog(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var logMessage = $"[{DateTime.Now:HH:mm:ss}] {message}";
            _logBuilder.AppendLine(logMessage);
            _logText.Text = _logBuilder.ToString();
            Console.WriteLine(logMessage);
        });
    }

    public void Complete(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            _progressBar.Value = 100;
            _statusText.Text = message;
            _statusText.Foreground = Avalonia.Media.Brushes.LightGreen;

            var totalElapsed = DateTime.Now - _startTime;
            _etaText.Text = $"✓ Completed in {totalElapsed.Hours:D2}:{totalElapsed.Minutes:D2}:{totalElapsed.Seconds:D2}";
            _etaText.Foreground = Avalonia.Media.Brushes.LightGreen;

            AddLog($"✓ {message}");
        });
    }

    public void Error(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            _statusText.Text = message;
            _statusText.Foreground = Avalonia.Media.Brushes.OrangeRed;
            AddLog($"✗ ERROR: {message}");
        });
    }
}
