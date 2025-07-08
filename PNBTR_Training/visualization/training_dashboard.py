#!/usr/bin/env python3
"""
PNBTR Training Dashboard - Phase 3
Real-time visualization and monitoring of training progress.
Provides interactive plots, spectral analysis, and performance metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class TrainingVisualizer:
    """
    Real-time training visualization and dashboard system.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (15, 10)):
        self.figure_size = figure_size
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'learning_rates': [],
            'timestamps': []
        }
        
        self.metrics_history = {
            'sdr': [],
            'stoi': [],
            'pesq_like': [],
            'spectral_centroid_error': [],
            'harmonic_ratio_error': []
        }
        
        # Real-time plotting state
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.is_live = False
        
    def initialize_dashboard(self) -> bool:
        """
        Initialize the interactive dashboard.
        
        Returns:
            True if successful, False if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available - visualization disabled")
            return False
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=self.figure_size)
        self.fig.suptitle('PNBTR Training Dashboard - Real-Time Monitoring', fontsize=16)
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Training progress plots
        self.axes['loss'] = self.fig.add_subplot(gs[0, 0])
        self.axes['accuracy'] = self.fig.add_subplot(gs[0, 1])
        self.axes['learning_rate'] = self.fig.add_subplot(gs[0, 2])
        
        # Metrics plots
        self.axes['sdr_stoi'] = self.fig.add_subplot(gs[1, 0])
        self.axes['pesq'] = self.fig.add_subplot(gs[1, 1])
        self.axes['spectral_errors'] = self.fig.add_subplot(gs[1, 2])
        
        # Signal analysis plots
        self.axes['waveform'] = self.fig.add_subplot(gs[2, 0])
        self.axes['spectrum'] = self.fig.add_subplot(gs[2, 1])
        self.axes['spectrogram'] = self.fig.add_subplot(gs[2, 2])
        
        # Initialize plot lines
        self._setup_plot_lines()
        
        # Configure plots
        self._configure_plots()
        
        plt.ion()  # Interactive mode
        self.is_live = True
        
        return True
    
    def _setup_plot_lines(self):
        """Setup plot lines for real-time updating"""
        # Training progress lines
        self.lines['loss'], = self.axes['loss'].plot([], [], 'b-', label='Training Loss')
        self.lines['accuracy'], = self.axes['accuracy'].plot([], [], 'g-', label='Accuracy')
        self.lines['learning_rate'], = self.axes['learning_rate'].plot([], [], 'r-', label='Learning Rate')
        
        # Metrics lines
        self.lines['sdr'], = self.axes['sdr_stoi'].plot([], [], 'b-', label='SDR')
        self.lines['stoi'], = self.axes['sdr_stoi'].plot([], [], 'c-', label='STOI')
        self.lines['pesq'], = self.axes['pesq'].plot([], [], 'm-', label='PESQ-like')
        
        self.lines['centroid_error'], = self.axes['spectral_errors'].plot([], [], 'r-', label='Centroid Error')
        self.lines['harmonic_error'], = self.axes['spectral_errors'].plot([], [], 'orange', label='Harmonic Error')
        
        # Signal display lines (will be updated with actual data)
        self.lines['target_waveform'], = self.axes['waveform'].plot([], [], 'g-', label='Target', alpha=0.7)
        self.lines['pred_waveform'], = self.axes['waveform'].plot([], [], 'r--', label='Prediction', alpha=0.8)
    
    def _configure_plots(self):
        """Configure plot appearance and labels"""
        # Training progress plots
        self.axes['loss'].set_title('Training Loss')
        self.axes['loss'].set_xlabel('Epoch')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].grid(True, alpha=0.3)
        self.axes['loss'].legend()
        
        self.axes['accuracy'].set_title('Accuracy Progress')
        self.axes['accuracy'].set_xlabel('Epoch')
        self.axes['accuracy'].set_ylabel('Accuracy')
        self.axes['accuracy'].set_ylim(0, 1)
        self.axes['accuracy'].grid(True, alpha=0.3)
        self.axes['accuracy'].legend()
        
        self.axes['learning_rate'].set_title('Learning Rate')
        self.axes['learning_rate'].set_xlabel('Epoch')
        self.axes['learning_rate'].set_ylabel('Learning Rate')
        self.axes['learning_rate'].set_yscale('log')
        self.axes['learning_rate'].grid(True, alpha=0.3)
        self.axes['learning_rate'].legend()
        
        # Metrics plots
        self.axes['sdr_stoi'].set_title('SDR & STOI Metrics')
        self.axes['sdr_stoi'].set_xlabel('Epoch')
        self.axes['sdr_stoi'].set_ylabel('Score')
        self.axes['sdr_stoi'].grid(True, alpha=0.3)
        self.axes['sdr_stoi'].legend()
        
        self.axes['pesq'].set_title('PESQ-like Score')
        self.axes['pesq'].set_xlabel('Epoch')
        self.axes['pesq'].set_ylabel('PESQ Score (1-5)')
        self.axes['pesq'].set_ylim(1, 5)
        self.axes['pesq'].grid(True, alpha=0.3)
        self.axes['pesq'].legend()
        
        self.axes['spectral_errors'].set_title('Spectral Errors')
        self.axes['spectral_errors'].set_xlabel('Epoch')
        self.axes['spectral_errors'].set_ylabel('Error')
        self.axes['spectral_errors'].grid(True, alpha=0.3)
        self.axes['spectral_errors'].legend()
        
        # Signal analysis plots
        self.axes['waveform'].set_title('Signal Comparison')
        self.axes['waveform'].set_xlabel('Time (samples)')
        self.axes['waveform'].set_ylabel('Amplitude')
        self.axes['waveform'].grid(True, alpha=0.3)
        self.axes['waveform'].legend()
        
        self.axes['spectrum'].set_title('Frequency Spectrum')
        self.axes['spectrum'].set_xlabel('Frequency (Hz)')
        self.axes['spectrum'].set_ylabel('Magnitude (dB)')
        self.axes['spectrum'].grid(True, alpha=0.3)
        
        self.axes['spectrogram'].set_title('Prediction Spectrogram')
        self.axes['spectrogram'].set_xlabel('Time')
        self.axes['spectrogram'].set_ylabel('Frequency (Hz)')
    
    def update_training_progress(self, epoch: int, loss: float, accuracy: float, 
                               learning_rate: float, metrics: Optional[Dict] = None):
        """
        Update training progress data and refresh plots.
        
        Args:
            epoch: Current training epoch
            loss: Current loss value
            accuracy: Current accuracy score
            learning_rate: Current learning rate
            metrics: Optional dictionary of additional metrics
        """
        # Update history
        self.training_history['epochs'].append(epoch)
        self.training_history['losses'].append(loss)
        self.training_history['accuracies'].append(accuracy)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['timestamps'].append(time.time())
        
        # Update metrics if provided
        if metrics:
            self.metrics_history['sdr'].append(metrics.get('SDR', 0))
            self.metrics_history['stoi'].append(metrics.get('STOI', 0))
            self.metrics_history['pesq_like'].append(metrics.get('PESQ_like', 1))
            self.metrics_history['spectral_centroid_error'].append(metrics.get('CentroidError', 0))
            self.metrics_history['harmonic_ratio_error'].append(metrics.get('HarmonicError', 0))
        
        # Update plots if dashboard is active
        if self.is_live and MATPLOTLIB_AVAILABLE:
            self._update_training_plots()
    
    def _update_training_plots(self):
        """Update the training progress plots"""
        epochs = self.training_history['epochs']
        
        if not epochs:
            return
        
        # Update training progress lines
        self.lines['loss'].set_data(epochs, self.training_history['losses'])
        self.lines['accuracy'].set_data(epochs, self.training_history['accuracies'])
        self.lines['learning_rate'].set_data(epochs, self.training_history['learning_rates'])
        
        # Update metrics lines (if data available)
        if self.metrics_history['sdr']:
            self.lines['sdr'].set_data(epochs[-len(self.metrics_history['sdr']):], self.metrics_history['sdr'])
            self.lines['stoi'].set_data(epochs[-len(self.metrics_history['stoi']):], self.metrics_history['stoi'])
            self.lines['pesq'].set_data(epochs[-len(self.metrics_history['pesq_like']):], self.metrics_history['pesq_like'])
            self.lines['centroid_error'].set_data(epochs[-len(self.metrics_history['spectral_centroid_error']):], 
                                                 self.metrics_history['spectral_centroid_error'])
            self.lines['harmonic_error'].set_data(epochs[-len(self.metrics_history['harmonic_ratio_error']):], 
                                                 self.metrics_history['harmonic_ratio_error'])
        
        # Auto-scale axes
        for ax_name, ax in self.axes.items():
            if ax_name in ['loss', 'accuracy', 'learning_rate', 'sdr_stoi', 'pesq', 'spectral_errors']:
                ax.relim()
                ax.autoscale_view()
        
        # Redraw
        plt.draw()
        plt.pause(0.01)
    
    def update_signal_display(self, target_signal: np.ndarray, 
                            predicted_signal: np.ndarray, 
                            sample_rate: int = 48000):
        """
        Update signal comparison and spectral analysis displays.
        
        Args:
            target_signal: Ground truth signal
            predicted_signal: Model prediction
            sample_rate: Audio sample rate
        """
        if not (self.is_live and MATPLOTLIB_AVAILABLE):
            return
        
        # Limit display length for performance
        max_samples = min(2048, len(target_signal), len(predicted_signal))
        target_display = target_signal[:max_samples]
        pred_display = predicted_signal[:max_samples]
        
        # Update waveform display
        time_axis = np.arange(max_samples)
        self.lines['target_waveform'].set_data(time_axis, target_display)
        self.lines['pred_waveform'].set_data(time_axis, pred_display)
        
        self.axes['waveform'].set_xlim(0, max_samples)
        self.axes['waveform'].set_ylim(
            min(np.min(target_display), np.min(pred_display)) * 1.1,
            max(np.max(target_display), np.max(pred_display)) * 1.1
        )
        
        # Update spectrum display
        self._update_spectrum_display(target_display, pred_display, sample_rate)
        
        # Update spectrogram
        self._update_spectrogram_display(pred_display, sample_rate)
        
        plt.draw()
        plt.pause(0.01)
    
    def _update_spectrum_display(self, target: np.ndarray, prediction: np.ndarray, sample_rate: int):
        """Update frequency spectrum comparison"""
        # Compute FFTs
        n_fft = min(1024, len(target))
        
        target_fft = np.abs(np.fft.fft(target[:n_fft]))[:n_fft//2]
        pred_fft = np.abs(np.fft.fft(prediction[:n_fft]))[:n_fft//2]
        
        # Convert to dB
        target_db = 20 * np.log10(target_fft + 1e-12)
        pred_db = 20 * np.log10(pred_fft + 1e-12)
        
        # Frequency axis
        freqs = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2]
        
        # Clear and replot
        self.axes['spectrum'].clear()
        self.axes['spectrum'].plot(freqs, target_db, 'g-', label='Target', alpha=0.7)
        self.axes['spectrum'].plot(freqs, pred_db, 'r--', label='Prediction', alpha=0.8)
        
        self.axes['spectrum'].set_title('Frequency Spectrum')
        self.axes['spectrum'].set_xlabel('Frequency (Hz)')
        self.axes['spectrum'].set_ylabel('Magnitude (dB)')
        self.axes['spectrum'].grid(True, alpha=0.3)
        self.axes['spectrum'].legend()
    
    def _update_spectrogram_display(self, signal: np.ndarray, sample_rate: int):
        """Update spectrogram display"""
        if len(signal) < 512:
            return
        
        # Compute spectrogram
        window_size = min(256, len(signal) // 4)
        hop_size = window_size // 4
        
        # Manual STFT for simplicity
        n_frames = (len(signal) - window_size) // hop_size + 1
        n_freqs = window_size // 2 + 1
        
        spectrogram = np.zeros((n_freqs, n_frames))
        
        for frame in range(n_frames):
            start = frame * hop_size
            end = start + window_size
            
            if end <= len(signal):
                windowed = signal[start:end] * np.hanning(window_size)
                fft_frame = np.abs(np.fft.fft(windowed))[:n_freqs]
                spectrogram[:, frame] = fft_frame
        
        # Convert to dB
        spectrogram_db = 20 * np.log10(spectrogram + 1e-12)
        
        # Time and frequency axes
        time_axis = np.arange(n_frames) * hop_size / sample_rate
        freq_axis = np.fft.fftfreq(window_size, 1/sample_rate)[:n_freqs]
        
        # Clear and replot
        self.axes['spectrogram'].clear()
        im = self.axes['spectrogram'].imshow(
            spectrogram_db, 
            aspect='auto', 
            origin='lower',
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        )
        
        self.axes['spectrogram'].set_title('Prediction Spectrogram')
        self.axes['spectrogram'].set_xlabel('Time (s)')
        self.axes['spectrogram'].set_ylabel('Frequency (Hz)')
    
    def save_training_summary(self, filepath: Path):
        """Save complete training history and metrics"""
        summary = {
            'training_history': self.training_history,
            'metrics_history': self.metrics_history,
            'session_info': {
                'total_epochs': len(self.training_history['epochs']),
                'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0,
                'final_accuracy': self.training_history['accuracies'][-1] if self.training_history['accuracies'] else 0,
                'training_duration': (
                    self.training_history['timestamps'][-1] - self.training_history['timestamps'][0]
                    if len(self.training_history['timestamps']) > 1 else 0
                )
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Training summary saved: {filepath}")
    
    def generate_static_report(self, save_path: Path):
        """Generate static HTML/PNG report of training session"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Cannot generate static report - matplotlib unavailable")
            return
        
        # Create a new figure for the report
        report_fig = plt.figure(figsize=(16, 12))
        report_fig.suptitle('PNBTR Training Session Report', fontsize=16)
        
        # Create comprehensive plots
        gs = GridSpec(3, 4, figure=report_fig, hspace=0.4, wspace=0.3)
        
        epochs = self.training_history['epochs']
        
        if not epochs:
            print("⚠️  No training data to report")
            return
        
        # Training curves
        ax1 = report_fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.training_history['losses'], 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2 = report_fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.training_history['accuracies'], 'g-', linewidth=2)
        ax2.axhline(y=0.9, color='r', linestyle='--', label='Mastery Threshold')
        ax2.set_title('Accuracy Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3 = report_fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.training_history['learning_rates'], 'r-', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Training statistics
        ax4 = report_fig.add_subplot(gs[0, 3])
        stats_text = [
            f"Total Epochs: {len(epochs)}",
            f"Final Loss: {self.training_history['losses'][-1]:.6f}",
            f"Final Accuracy: {self.training_history['accuracies'][-1]:.3f}",
            f"Peak Accuracy: {max(self.training_history['accuracies']):.3f}",
            f"Mastery Achieved: {'Yes' if max(self.training_history['accuracies']) >= 0.9 else 'No'}"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Training Statistics')
        ax4.axis('off')
        
        # Metrics plots (if available)
        if self.metrics_history['sdr']:
            ax5 = report_fig.add_subplot(gs[1, 0])
            metric_epochs = epochs[-len(self.metrics_history['sdr']):]
            ax5.plot(metric_epochs, self.metrics_history['sdr'], 'b-', label='SDR')
            ax5.plot(metric_epochs, self.metrics_history['stoi'], 'c-', label='STOI')
            ax5.set_title('Audio Quality Metrics')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Score')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            ax6 = report_fig.add_subplot(gs[1, 1])
            ax6.plot(metric_epochs, self.metrics_history['pesq_like'], 'm-', linewidth=2)
            ax6.set_title('PESQ-like Score')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('PESQ Score')
            ax6.set_ylim(1, 5)
            ax6.grid(True, alpha=0.3)
        
        # Save the report
        report_path = save_path.with_suffix('.png')
        report_fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(report_fig)
        
        print(f"📈 Static report saved: {report_path}")
    
    def close_dashboard(self):
        """Close the interactive dashboard"""
        if self.is_live and MATPLOTLIB_AVAILABLE:
            plt.ioff()
            plt.close(self.fig)
            self.is_live = False

def create_training_visualizer(figure_size: Tuple[int, int] = (15, 10)) -> TrainingVisualizer:
    """Factory function to create training visualizer"""
    return TrainingVisualizer(figure_size)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Training Dashboard Test")
    
    # Create visualizer
    visualizer = create_training_visualizer()
    
    if not visualizer.initialize_dashboard():
        print("❌ Could not initialize dashboard - matplotlib required")
        exit(1)
    
    print("📊 Dashboard initialized, simulating training...")
    
    # Simulate training progress
    np.random.seed(42)
    
    for epoch in range(50):
        # Simulate training metrics
        loss = 1.0 * np.exp(-epoch * 0.1) + 0.1 * np.random.random()
        accuracy = 1.0 - loss + 0.05 * np.random.random()
        lr = 0.001 * (0.95 ** (epoch // 10))
        
        # Simulate signal metrics
        metrics = {
            'SDR': min(1.0, 0.3 + accuracy * 0.7) + 0.05 * np.random.random(),
            'STOI': min(1.0, 0.5 + accuracy * 0.5) + 0.03 * np.random.random(),
            'PESQ_like': min(5.0, 1.5 + accuracy * 3.5) + 0.1 * np.random.random(),
            'CentroidError': (1.0 - accuracy) * 1000 + 50 * np.random.random(),
            'HarmonicError': (1.0 - accuracy) * 5 + np.random.random()
        }
        
        # Update dashboard
        visualizer.update_training_progress(epoch, loss, accuracy, lr, metrics)
        
        # Simulate signal update every 10 epochs
        if epoch % 10 == 0:
            # Generate test signals
            t = np.linspace(0, 0.1, 4800)  # 0.1 second at 48kHz
            target = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
            prediction = target + (1.0 - accuracy) * 0.1 * np.random.normal(0, 1, len(target))
            
            visualizer.update_signal_display(target, prediction, 48000)
        
        time.sleep(0.1)  # Simulate training time
        
        # Check for early mastery
        if accuracy >= 0.9:
            print(f"🏆 Mastery achieved at epoch {epoch}!")
            break
    
    print("\n📊 Training simulation complete")
    
    # Save results
    summary_path = Path("training_summary.json")
    visualizer.save_training_summary(summary_path)
    
    report_path = Path("training_report")
    visualizer.generate_static_report(report_path)
    
    # Keep dashboard open for inspection
    input("Press Enter to close dashboard...")
    visualizer.close_dashboard()
    
    print("✅ Dashboard test complete") 