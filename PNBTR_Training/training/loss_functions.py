#!/usr/bin/env python3
"""
PNBTR Loss Functions & Metric Evaluators
Implements the core metrics for evaluating reconstruction quality:
- SDR (Signal-to-Distortion Ratio)
- ΔFFT (Spectral deviation via FFT comparison)
- Envelope Deviation
- Phase Skew/Alignment
- Dynamic Range Preservation
- Frequency Response Retention
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def evaluate_metrics(predicted, target, sample_rate=192000):
    """
    Comprehensive evaluation of reconstruction quality.
    
    Args:
        predicted: Reconstructed signal array
        target: Ground truth signal array  
        sample_rate: Audio sample rate (default 192kHz for JELLIE)
        
    Returns:
        dict: All metric scores where 1.0 = perfect, 0.0 = worst
    """
    # Ensure signals are same length and aligned
    min_len = min(len(predicted), len(target))
    predicted = predicted[:min_len]
    target = target[:min_len]
    
    # Prevent division by zero
    if np.max(np.abs(target)) < 1e-10:
        return _zero_signal_metrics()
    
    metrics = {
        "SDR": calculate_sdr(predicted, target),
        "DeltaFFT": calculate_delta_fft(predicted, target, sample_rate),
        "EnvelopeDev": calculate_envelope_deviation(predicted, target),
        "PhaseSkew": calculate_phase_alignment(predicted, target),
        "DynamicRange": calculate_dynamic_range_preservation(predicted, target),
        "FrequencyResponse": calculate_frequency_response_retention(predicted, target, sample_rate)
    }
    
    return metrics

def calculate_sdr(predicted, target):
    """
    Signal-to-Distortion Ratio calculation.
    
    SDR = 20 * log10(||target||^2 / ||target - predicted||^2)
    
    Returns:
        float: SDR score normalized to [0,1] where 1.0 = perfect (>30dB)
    """
    # Calculate signal and error power
    signal_power = np.sum(target ** 2)
    error_power = np.sum((target - predicted) ** 2)
    
    # Avoid division by zero
    if error_power < 1e-15:
        return 1.0  # Perfect reconstruction
    
    # Calculate SDR in dB
    sdr_db = 10 * np.log10(signal_power / error_power)
    
    # Normalize: 0dB = 0.0, 30dB+ = 1.0 (excellent quality)
    sdr_normalized = np.clip(sdr_db / 30.0, 0.0, 1.0)
    
    return float(sdr_normalized)

def calculate_delta_fft(predicted, target, sample_rate):
    """
    Spectral deviation via FFT comparison.
    Measures how much the frequency content differs.
    
    Returns:
        float: Spectral similarity [0,1] where 1.0 = identical spectra
    """
    # Calculate FFTs
    predicted_fft = np.abs(fft(predicted))
    target_fft = np.abs(fft(target))
    
    # Focus on audible range (20Hz - 20kHz)
    freqs = fftfreq(len(target), 1/sample_rate)
    audible_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 20000)
    
    predicted_spectrum = predicted_fft[audible_mask]
    target_spectrum = target_fft[audible_mask]
    
    # Normalize spectra
    predicted_norm = predicted_spectrum / (np.max(predicted_spectrum) + 1e-15)
    target_norm = target_spectrum / (np.max(target_spectrum) + 1e-15)
    
    # Calculate spectral correlation coefficient
    correlation = np.corrcoef(predicted_norm, target_norm)[0, 1]
    
    # Handle NaN case
    if np.isnan(correlation):
        return 0.0
    
    # Return correlation (1.0 = perfect spectral match)
    return max(0.0, float(correlation))

def calculate_envelope_deviation(predicted, target):
    """
    Amplitude envelope shape preservation.
    Measures how well the signal's dynamics are preserved.
    
    Returns:
        float: Envelope similarity [0,1] where 1.0 = perfect envelope match
    """
    # Calculate amplitude envelopes using Hilbert transform
    predicted_env = np.abs(signal.hilbert(predicted))
    target_env = np.abs(signal.hilbert(target))
    
    # Smooth envelopes to focus on macro dynamics (not sample-level)
    window_size = max(1, len(target) // 1000)  # ~1ms windows at 192kHz
    predicted_smooth = signal.savgol_filter(predicted_env, window_size | 1, 3)
    target_smooth = signal.savgol_filter(target_env, window_size | 1, 3)
    
    # Normalize envelopes
    predicted_norm = predicted_smooth / (np.max(predicted_smooth) + 1e-15)
    target_norm = target_smooth / (np.max(target_smooth) + 1e-15)
    
    # Calculate envelope correlation
    correlation = np.corrcoef(predicted_norm, target_norm)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return max(0.0, float(correlation))

def calculate_phase_alignment(predicted, target):
    """
    Temporal/phase alignment between signals.
    Measures timing accuracy of reconstruction.
    
    Returns:
        float: Phase alignment [0,1] where 1.0 = perfect timing
    """
    # Cross-correlation to find optimal alignment
    correlation = signal.correlate(target, predicted, mode='full')
    
    # Find peak correlation and its offset
    peak_idx = np.argmax(np.abs(correlation))
    optimal_offset = peak_idx - (len(target) - 1)
    
    # Calculate phase alignment score based on offset
    # Perfect alignment = 0 offset, 1 sample off = slight penalty
    max_acceptable_offset = len(target) * 0.01  # 1% of signal length
    
    if abs(optimal_offset) <= max_acceptable_offset:
        # Penalize based on offset magnitude
        alignment_score = 1.0 - (abs(optimal_offset) / max_acceptable_offset) * 0.2
    else:
        # Severe misalignment
        alignment_score = 0.8 * np.exp(-abs(optimal_offset) / max_acceptable_offset)
    
    return max(0.0, float(alignment_score))

def calculate_dynamic_range_preservation(predicted, target):
    """
    Measure preservation of dynamic range (loud vs quiet passages).
    
    Returns:
        float: Dynamic range similarity [0,1] where 1.0 = perfect preservation
    """
    # Calculate RMS in overlapping windows
    window_samples = max(1, len(target) // 100)  # ~10ms windows at 192kHz
    hop_samples = window_samples // 2
    
    predicted_rms = []
    target_rms = []
    
    for start in range(0, len(target) - window_samples, hop_samples):
        end = start + window_samples
        
        pred_window = predicted[start:end]
        target_window = target[start:end]
        
        predicted_rms.append(np.sqrt(np.mean(pred_window ** 2)))
        target_rms.append(np.sqrt(np.mean(target_window ** 2)))
    
    predicted_rms = np.array(predicted_rms)
    target_rms = np.array(target_rms)
    
    # Convert to dB (avoiding log(0))
    predicted_db = 20 * np.log10(predicted_rms + 1e-15)
    target_db = 20 * np.log10(target_rms + 1e-15)
    
    # Calculate dynamic range correlation
    correlation = np.corrcoef(predicted_db, target_db)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return max(0.0, float(correlation))

def calculate_frequency_response_retention(predicted, target, sample_rate):
    """
    Measure how flat/similar the frequency response is between signals.
    Tests whether reconstruction preserves frequency balance.
    
    Returns:
        float: Frequency response similarity [0,1] where 1.0 = perfect retention
    """
    # Calculate power spectral densities
    freqs_pred, psd_pred = signal.welch(predicted, sample_rate, nperseg=min(8192, len(predicted)//4))
    freqs_target, psd_target = signal.welch(target, sample_rate, nperseg=min(8192, len(target)//4))
    
    # Focus on audible range
    audible_mask = (freqs_target >= 20) & (freqs_target <= 20000)
    psd_pred_audible = psd_pred[audible_mask]
    psd_target_audible = psd_target[audible_mask]
    
    # Convert to dB and normalize
    pred_db = 10 * np.log10(psd_pred_audible + 1e-15)
    target_db = 10 * np.log10(psd_target_audible + 1e-15)
    
    # Normalize relative to peak
    pred_norm = pred_db - np.max(pred_db)
    target_norm = target_db - np.max(target_db)
    
    # Calculate frequency response correlation
    correlation = np.corrcoef(pred_norm, target_norm)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return max(0.0, float(correlation))

def _zero_signal_metrics():
    """Return default metrics for zero/silent signals"""
    return {
        "SDR": 0.0,
        "DeltaFFT": 0.0, 
        "EnvelopeDev": 0.0,
        "PhaseSkew": 0.0,
        "DynamicRange": 0.0,
        "FrequencyResponse": 0.0
    }

# Additional utility functions for training

def calculate_perceptual_loss(predicted, target, sample_rate=192000):
    """
    Perceptually-weighted loss function that emphasizes audible differences.
    Uses A-weighting and bark scale for human hearing characteristics.
    """
    # A-weighting filter approximation
    freqs = fftfreq(len(target), 1/sample_rate)
    
    # A-weighting curve (simplified)
    a_weight = np.ones_like(freqs)
    for i, f in enumerate(np.abs(freqs)):
        if f > 0:
            # Simplified A-weighting formula
            a_weight[i] = (12194**2 * f**4) / ((f**2 + 20.6**2) * 
                         np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * 
                         (f**2 + 12194**2))
    
    # Apply weighting to FFT difference
    pred_fft = fft(predicted)
    target_fft = fft(target)
    
    weighted_error = (pred_fft - target_fft) * a_weight
    perceptual_loss = np.mean(np.abs(weighted_error)**2)
    
    return float(perceptual_loss)

def calculate_transient_preservation(predicted, target):
    """
    Measure how well transients (sharp attacks) are preserved.
    Important for percussion, speech consonants, etc.
    """
    # Detect transients using onset detection
    pred_novelty = np.diff(np.abs(predicted))
    target_novelty = np.diff(np.abs(target))
    
    # Find peaks (transients)
    pred_peaks, _ = signal.find_peaks(pred_novelty, height=np.std(pred_novelty))
    target_peaks, _ = signal.find_peaks(target_novelty, height=np.std(target_novelty))
    
    if len(target_peaks) == 0:
        return 1.0  # No transients to preserve
    
    # Measure timing accuracy of transient preservation
    matches = 0
    tolerance = len(target) * 0.001  # 0.1% timing tolerance
    
    for target_peak in target_peaks:
        # Find closest predicted peak
        distances = np.abs(pred_peaks - target_peak)
        if len(distances) > 0 and np.min(distances) <= tolerance:
            matches += 1
    
    transient_score = matches / len(target_peaks)
    return float(transient_score)

# Debug and visualization helpers

def plot_metrics_comparison(predicted, target, sample_rate=192000):
    """
    Generate comparison plots for debugging reconstruction quality.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time domain comparison
        time = np.arange(len(target)) / sample_rate
        axes[0,0].plot(time, target, label='Target', alpha=0.7)
        axes[0,0].plot(time, predicted, label='Predicted', alpha=0.7)
        axes[0,0].set_title('Waveform Comparison')
        axes[0,0].legend()
        
        # Frequency domain comparison
        freqs, target_psd = signal.welch(target, sample_rate)
        _, pred_psd = signal.welch(predicted, sample_rate)
        
        axes[0,1].semilogx(freqs, 10*np.log10(target_psd), label='Target')
        axes[0,1].semilogx(freqs, 10*np.log10(pred_psd), label='Predicted')
        axes[0,1].set_title('Power Spectral Density')
        axes[0,1].legend()
        
        # Envelope comparison
        target_env = np.abs(signal.hilbert(target))
        pred_env = np.abs(signal.hilbert(predicted))
        
        axes[1,0].plot(time, target_env, label='Target Envelope')
        axes[1,0].plot(time, pred_env, label='Predicted Envelope')
        axes[1,0].set_title('Amplitude Envelope')
        axes[1,0].legend()
        
        # Difference signal
        difference = predicted - target
        axes[1,1].plot(time, difference)
        axes[1,1].set_title('Reconstruction Error')
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        print("matplotlib not available for plotting")
        return None

def print_metric_summary(metrics):
    """Print formatted summary of all metrics"""
    print("\n📊 PNBTR Reconstruction Metrics:")
    print("=" * 40)
    
    for metric, value in metrics.items():
        percentage = value * 100
        status = "✅" if value >= 0.9 else "⚠️" if value >= 0.7 else "❌"
        print(f"{status} {metric:<20}: {percentage:6.1f}%")
    
    # Composite score
    composite = np.mean(list(metrics.values()))
    comp_status = "✅" if composite >= 0.9 else "⚠️" if composite >= 0.7 else "❌"
    print("-" * 40)
    print(f"{comp_status} {'Composite Score':<20}: {composite*100:6.1f}%")
    print("=" * 40) 