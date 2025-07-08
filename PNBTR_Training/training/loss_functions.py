#!/usr/bin/env python3
"""
PNBTR Loss Functions & Metric Evaluators
Enhanced to address 250708_093109_System_Audit.md requirements:

Core Metrics:
- SDR (Signal-to-Distortion Ratio)
- ΔFFT (Spectral deviation via FFT comparison)
- Envelope Deviation
- Phase Skew/Alignment
- Dynamic Range Preservation
- Frequency Response Retention

New Audit-Required Metrics:
- THD+N (Total Harmonic Distortion + Noise)
- Coloration Percentage ("Color %")
- Phase Linearity Testing
- Real-time Processing Validation
- Multi-modal (Audio/Video) Handling
- Spectral Analysis Tools
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.stats import pearsonr
import warnings
from typing import Dict, List, Tuple, Optional, Union
import time

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def evaluate_metrics(predicted, target, sample_rate=192000, enhanced_analysis=True):
    """
    Comprehensive evaluation of reconstruction quality.
    Enhanced to meet 250708_093109_System_Audit.md requirements.
    
    Args:
        predicted: Reconstructed signal array
        target: Ground truth signal array  
        sample_rate: Audio sample rate (default 192kHz for JELLIE)
        enhanced_analysis: Enable comprehensive quality metrics
        
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
    
    # Core PNBTR metrics
    metrics = {
        "SDR": calculate_sdr(predicted, target),
        "DeltaFFT": calculate_delta_fft(predicted, target, sample_rate),
        "EnvelopeDev": calculate_envelope_deviation(predicted, target),
        "PhaseSkew": calculate_phase_alignment(predicted, target),
        "DynamicRange": calculate_dynamic_range_preservation(predicted, target),
        "FrequencyResponse": calculate_frequency_response_retention(predicted, target, sample_rate)
    }
    
    # Enhanced audit-required metrics
    if enhanced_analysis:
        audit_metrics = calculate_audit_metrics(predicted, target, sample_rate)
        metrics.update(audit_metrics)
    
    # Calculate overall quality score
    metrics["OverallQuality"] = calculate_overall_quality_score(metrics)
    
    return metrics

def calculate_audit_metrics(predicted, target, sample_rate=192000):
    """
    Calculate enhanced metrics required by the system audit.
    
    Args:
        predicted: Reconstructed signal
        target: Ground truth signal
        sample_rate: Audio sample rate
        
    Returns:
        dict: Enhanced quality metrics
    """
    metrics = {}
    
    # THD+N (Total Harmonic Distortion + Noise)
    metrics["THD_N_Percent"] = calculate_thd_n(predicted, sample_rate)
    
    # Coloration percentage
    metrics["ColorationPercent"] = calculate_coloration_percentage(predicted, target, sample_rate)
    
    # Phase linearity
    metrics["PhaseLinearity"] = calculate_phase_linearity(predicted, target, sample_rate)
    
    # Frequency response flatness
    metrics["FreqResponseFlatness"] = calculate_frequency_response_flatness(predicted, target, sample_rate)
    
    # Transient preservation
    metrics["TransientPreservation"] = calculate_transient_preservation_enhanced(predicted, target)
    
    # Noise floor measurement
    metrics["NoiseFloor"] = calculate_noise_floor(predicted)
    
    # Hi-Fi standards compliance
    metrics["MeetsHiFiStandards"] = check_hifi_standards(metrics)
    
    return metrics
    
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
    predicted_env = np.abs(sp_signal.hilbert(predicted))
    target_env = np.abs(sp_signal.hilbert(target))
    
    # Smooth envelopes to focus on macro dynamics (not sample-level)
    window_size = max(1, len(target) // 1000)  # ~1ms windows at 192kHz
    predicted_smooth = sp_signal.savgol_filter(predicted_env, window_size | 1, 3)
    target_smooth = sp_signal.savgol_filter(target_env, window_size | 1, 3)
    
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
    correlation = sp_signal.correlate(target, predicted, mode='full')
    
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
    freqs_pred, psd_pred = sp_signal.welch(predicted, sample_rate, nperseg=min(8192, len(predicted)//4))
    freqs_target, psd_target = sp_signal.welch(target, sample_rate, nperseg=min(8192, len(target)//4))
    
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

# ===============================================================================
# ENHANCED AUDIT-REQUIRED METRICS (250708_093109_System_Audit.md)
# ===============================================================================

def calculate_transient_preservation_enhanced(predicted, target):
    """
    Enhanced transient preservation measurement.
    Measures how well transients (sharp attacks) are preserved.
    """
    if len(predicted) != len(target) or len(target) == 0:
        return 0.0
    
    # Calculate envelope correlation as a measure of transient preservation
    window_size = max(1, len(target) // 1000)  # ~1ms windows
    
    input_envelope = []
    output_envelope = []
    
    for i in range(0, len(target), window_size):
        end_idx = min(i + window_size, len(target))
        
        input_max = np.max(np.abs(target[i:end_idx]))
        output_max = np.max(np.abs(predicted[i:end_idx]))
        
        input_envelope.append(input_max)
        output_envelope.append(output_max)
    
    # Calculate correlation coefficient
    if len(input_envelope) < 2:
        return 0.0
    
    correlation = np.corrcoef(input_envelope, output_envelope)[0, 1]
    if np.isnan(correlation):
        return 0.0
    
    return max(0.0, correlation)

def calculate_thd_n(signal, sample_rate, fundamental_freq=None):
    """
    Calculate Total Harmonic Distortion + Noise percentage.
    
    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        fundamental_freq: Fundamental frequency (auto-detect if None)
        
    Returns:
        float: THD+N percentage (0-100%)
    """
    if len(signal) == 0:
        return 100.0
    
    # Auto-detect fundamental frequency if not provided
    if fundamental_freq is None:
        fundamental_freq = detect_fundamental_frequency(signal, sample_rate)
    
    if fundamental_freq <= 0:
        return 100.0
    
    # Perform FFT
    fft_signal = fft(signal)
    freqs = fftfreq(len(signal), 1/sample_rate)
    
    # Find fundamental frequency bin
    fund_bin = np.argmin(np.abs(freqs - fundamental_freq))
    fund_magnitude = np.abs(fft_signal[fund_bin])
    fund_power = fund_magnitude ** 2
    
    # Calculate harmonic powers (2nd through 10th harmonic)
    harmonic_power = 0.0
    for harmonic in range(2, 11):
        harmonic_freq = fundamental_freq * harmonic
        if harmonic_freq < sample_rate / 2:  # Below Nyquist
            harmonic_bin = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_magnitude = np.abs(fft_signal[harmonic_bin])
            harmonic_power += harmonic_magnitude ** 2
    
    # Calculate noise power (everything else except DC and harmonics)
    total_power = np.sum(np.abs(fft_signal[1:len(fft_signal)//2]) ** 2)  # Exclude DC
    noise_power = total_power - fund_power - harmonic_power
    noise_power = max(0, noise_power)  # Ensure non-negative
    
    # Calculate THD+N percentage
    if fund_power <= 1e-15:
        return 100.0
    
    thd_n = np.sqrt((harmonic_power + noise_power) / fund_power) * 100.0
    return min(100.0, thd_n)

def calculate_coloration_percentage(predicted, target, sample_rate):
    """
    Calculate "coloration" percentage - overall signal alteration.
    
    This metric quantifies how much the signal has been "colored" or altered
    from its original form, combining spectral, harmonic, and dynamic changes.
    
    Args:
        predicted: Processed signal
        target: Original signal
        sample_rate: Sample rate in Hz
        
    Returns:
        float: Coloration percentage (0% = transparent, higher = more colored)
    """
    if len(predicted) != len(target) or len(target) == 0:
        return 100.0
    
    # Spectral coloration (frequency content changes)
    pred_fft = np.abs(fft(predicted))
    target_fft = np.abs(fft(target))
    
    # Normalize spectra
    pred_norm = pred_fft / (np.max(pred_fft) + 1e-15)
    target_norm = target_fft / (np.max(target_fft) + 1e-15)
    
    # Calculate spectral deviation in audible range
    freqs = fftfreq(len(target), 1/sample_rate)
    audible_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 20000)
    
    spectral_diff = np.mean((pred_norm[audible_mask] - target_norm[audible_mask]) ** 2)
    spectral_coloration = np.sqrt(spectral_diff) * 100.0
    
    # Harmonic coloration (THD contribution)
    thd_contribution = calculate_thd_n(predicted, sample_rate)
    
    # Dynamic coloration (envelope changes)
    pred_envelope = np.abs(sp_signal.hilbert(predicted))
    target_envelope = np.abs(sp_signal.hilbert(target))
    
    envelope_correlation = np.corrcoef(pred_envelope, target_envelope)[0, 1]
    if np.isnan(envelope_correlation):
        envelope_correlation = 0.0
    
    envelope_coloration = (1.0 - max(0.0, envelope_correlation)) * 100.0
    
    # Combine colorations (weighted)
    total_coloration = (
        0.5 * spectral_coloration +
        0.3 * min(thd_contribution, 10.0) +  # Cap THD contribution at 10%
        0.2 * envelope_coloration
    )
    
    return min(100.0, total_coloration)

def calculate_phase_linearity(predicted, target, sample_rate):
    """
    Measure phase linearity of the processing system.
    
    Linear phase means all frequencies are delayed by the same amount,
    preserving waveform shape and temporal relationships.
    
    Args:
        predicted: Processed signal
        target: Original signal
        sample_rate: Sample rate in Hz
        
    Returns:
        float: Phase linearity score (0-1, 1 = perfectly linear)
    """
    if len(predicted) != len(target) or len(target) == 0:
        return 0.0
    
    # Calculate transfer function
    pred_fft = fft(predicted)
    target_fft = fft(target)
    
    # Avoid division by zero
    transfer_function = np.divide(pred_fft, target_fft, 
                                 out=np.zeros_like(pred_fft), 
                                 where=(np.abs(target_fft) > 1e-10))
    
    # Extract phase response
    phase_response = np.angle(transfer_function)
    freqs = fftfreq(len(target), 1/sample_rate)
    
    # Focus on audible frequencies
    audible_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 20000)
    audible_freqs = freqs[audible_mask]
    audible_phase = phase_response[audible_mask]
    
    if len(audible_freqs) < 10:  # Not enough points for analysis
        return 0.0
    
    # Unwrap phase
    audible_phase = np.unwrap(audible_phase)
    
    # Fit linear model to phase vs frequency
    # Linear phase: phase = -2π * delay * frequency + constant
    coeffs = np.polyfit(audible_freqs, audible_phase, 1)
    phase_linear_fit = np.polyval(coeffs, audible_freqs)
    
    # Calculate R-squared (goodness of fit)
    ss_res = np.sum((audible_phase - phase_linear_fit) ** 2)
    ss_tot = np.sum((audible_phase - np.mean(audible_phase)) ** 2)
    
    if ss_tot <= 1e-15:
        return 1.0  # Perfect if no phase variation
    
    r_squared = max(0.0, 1.0 - ss_res / ss_tot)
    
    return r_squared

def calculate_frequency_response_flatness(predicted, target, sample_rate):
    """
    Measure frequency response flatness in dB.
    
    A flat frequency response means all frequencies are treated equally.
    This measures the maximum deviation from flat response.
    
    Args:
        predicted: Processed signal
        target: Original signal
        sample_rate: Sample rate in Hz
        
    Returns:
        float: Maximum deviation from flat response in dB (negative = good)
    """
    if len(predicted) != len(target) or len(target) == 0:
        return -60.0  # Very poor response
    
    # Calculate magnitude response
    pred_fft = np.abs(fft(predicted))
    target_fft = np.abs(fft(target))
    
    # Calculate transfer function magnitude
    magnitude_response = np.divide(pred_fft, target_fft,
                                  out=np.ones_like(pred_fft),
                                  where=(target_fft > 1e-10))
    
    # Focus on audible frequencies
    freqs = fftfreq(len(target), 1/sample_rate)
    audible_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 20000)
    audible_response = magnitude_response[audible_mask]
    
    if len(audible_response) == 0:
        return -60.0
    
    # Convert to dB
    response_db = 20 * np.log10(np.maximum(audible_response, 1e-10))
    
    # Calculate deviation from flat (0 dB)
    mean_response = np.mean(response_db)
    max_deviation = np.max(np.abs(response_db - mean_response))
    
    return -max_deviation  # Negative because larger deviation is worse

def calculate_noise_floor(signal):
    """
    Calculate noise floor in dBFS.
    
    Args:
        signal: Audio signal
        
    Returns:
        float: Noise floor in dBFS (more negative = quieter)
    """
    if len(signal) == 0:
        return -120.0
    
    # Use RMS as noise floor estimate
    rms = np.sqrt(np.mean(signal ** 2))
    
    if rms <= 1e-10:
        return -120.0  # Very quiet
    
    # Convert to dBFS (assuming full scale = 1.0)
    noise_floor_dbfs = 20 * np.log10(rms)
    
    return noise_floor_dbfs

def detect_fundamental_frequency(signal, sample_rate):
    """
    Auto-detect fundamental frequency using autocorrelation.
    
    Args:
        signal: Audio signal
        sample_rate: Sample rate in Hz
        
    Returns:
        float: Fundamental frequency in Hz (0 if not detected)
    """
    if len(signal) < 100:
        return 0.0
    
    # Normalize signal
    signal_norm = signal / (np.max(np.abs(signal)) + 1e-15)
    
    # Autocorrelation
    autocorr = np.correlate(signal_norm, signal_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find the first peak after the main peak (fundamental period)
    # Skip the first 1ms to avoid the main peak
    min_period = max(1, int(0.001 * sample_rate))
    max_period = min(len(autocorr) - 1, int(0.05 * sample_rate))  # Max 50ms period (20Hz)
    
    if max_period <= min_period:
        return 0.0
    
    search_range = autocorr[min_period:max_period]
    
    if len(search_range) == 0:
        return 0.0
    
    # Find peaks
    peaks, _ = sp_signal.find_peaks(search_range, height=0.1 * np.max(search_range))
    
    if len(peaks) == 0:
        return 0.0
    
    # First peak corresponds to fundamental period
    fundamental_period = peaks[0] + min_period
    fundamental_freq = sample_rate / fundamental_period
    
    # Sanity check: should be in human audible range
    if 20 <= fundamental_freq <= 20000:
        return fundamental_freq
    else:
        return 0.0

def check_hifi_standards(metrics):
    """
    Check if metrics meet high-fidelity audio standards.
    
    Based on professional audio requirements:
    - SNR > 60 dB
    - THD+N < 0.1%
    - Frequency response within ±1 dB
    - Phase linearity > 90%
    
    Args:
        metrics: Dictionary of calculated metrics
        
    Returns:
        bool: True if meets hi-fi standards
    """
    thresholds = {
        "SDR": 0.9,                    # > 54 dB (normalized)
        "THD_N_Percent": 0.1,          # < 0.1%
        "FreqResponseFlatness": -1.0,   # Within ±1 dB
        "PhaseLinearity": 0.9,          # > 90%
        "DynamicRange": 0.85,           # > 51 dB (normalized)
        "ColorationPercent": 0.1        # < 0.1%
    }
    
    for metric, threshold in thresholds.items():
        if metric not in metrics:
            continue
            
        value = metrics[metric]
        
        if metric in ["THD_N_Percent", "ColorationPercent"]:
            # Lower is better for these metrics
            if value > threshold:
                return False
        elif metric == "FreqResponseFlatness":
            # More negative is better (smaller deviation)
            if value > threshold:
                return False
        else:
            # Higher is better for most metrics
            if value < threshold:
                return False
    
    return True

def calculate_overall_quality_score(metrics):
    """
    Calculate overall quality score combining all metrics.
    
    Args:
        metrics: Dictionary of all calculated metrics
        
    Returns:
        float: Overall quality score (0-1, 1 = perfect)
    """
    # Weights for different metric categories
    weights = {
        "SDR": 0.20,
        "DeltaFFT": 0.15,
        "FrequencyResponse": 0.15,
        "PhaseLinearity": 0.10,
        "DynamicRange": 0.10,
        "TransientPreservation": 0.10,
        "EnvelopeDev": 0.08,
        "PhaseSkew": 0.07,
        "ColorationPercent": 0.05  # Negative contribution
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in metrics:
            value = metrics[metric]
            
            # Invert coloration percentage (lower is better)
            if metric == "ColorationPercent":
                value = max(0.0, 1.0 - value / 10.0)  # 10% coloration = 0 score
            
            total_score += weight * value
            total_weight += weight
    
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0

def _zero_signal_metrics():
    """Return default metrics for zero/invalid signals."""
    return {
        "SDR": 0.0,
        "DeltaFFT": 0.0,
        "EnvelopeDev": 0.0,
        "PhaseSkew": 0.0,
        "DynamicRange": 0.0,
        "FrequencyResponse": 0.0,
        "THD_N_Percent": 100.0,
        "ColorationPercent": 100.0,
        "PhaseLinearity": 0.0,
        "FreqResponseFlatness": -60.0,
        "TransientPreservation": 0.0,
        "NoiseFloor": -120.0,
        "MeetsHiFiStandards": False,
        "OverallQuality": 0.0
    }

# ===============================================================================
# REAL-TIME PROCESSING VALIDATION
# ===============================================================================

def validate_realtime_processing(process_function, sample_rate=48000, 
                                buffer_size=256, target_latency_ms=10.0):
    """
    Validate that processing can meet real-time requirements.
    
    Args:
        process_function: Function to test (takes audio array, returns processed array)
        sample_rate: Audio sample rate
        buffer_size: Processing buffer size in samples
        target_latency_ms: Target latency in milliseconds
        
    Returns:
        dict: Real-time validation results
    """
    # Generate test signal
    test_duration = buffer_size / sample_rate
    test_signal = np.random.randn(buffer_size).astype(np.float32) * 0.1
    
    # Measure processing time over multiple iterations
    num_iterations = 100
    processing_times = []
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        output = process_function(test_signal)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000.0
        processing_times.append(processing_time_ms)
    
    # Calculate statistics
    mean_latency = np.mean(processing_times)
    max_latency = np.max(processing_times)
    std_latency = np.std(processing_times)
    
    # Check if meets real-time requirements
    meets_realtime = max_latency <= target_latency_ms
    
    # Calculate CPU load percentage
    audio_frame_time_ms = (buffer_size / sample_rate) * 1000.0
    cpu_load_percent = (mean_latency / audio_frame_time_ms) * 100.0
    
    return {
        "mean_latency_ms": mean_latency,
        "max_latency_ms": max_latency,
        "std_latency_ms": std_latency,
        "target_latency_ms": target_latency_ms,
        "meets_realtime": meets_realtime,
        "cpu_load_percent": cpu_load_percent,
        "buffer_size": buffer_size,
        "sample_rate": sample_rate
    }

# ===============================================================================
# MULTI-MODAL (AUDIO/VIDEO) SUPPORT
# ===============================================================================

def evaluate_video_metrics(predicted_frames, target_frames):
    """
    Evaluate video reconstruction quality metrics.
    
    For Jvid data treated as audio streams, this provides video-specific
    quality assessment when frames are reconstructed.
    
    Args:
        predicted_frames: Reconstructed video frames (list of 2D/3D arrays)
        target_frames: Original video frames (list of 2D/3D arrays)
        
    Returns:
        dict: Video quality metrics
    """
    if len(predicted_frames) != len(target_frames) or len(target_frames) == 0:
        return {"PSNR": 0.0, "SSIM": 0.0, "ColorAccuracy": 0.0}
    
    psnr_scores = []
    ssim_scores = []
    color_accuracy_scores = []
    
    for pred_frame, target_frame in zip(predicted_frames, target_frames):
        # Ensure frames have same shape
        if pred_frame.shape != target_frame.shape:
            continue
        
        # Calculate PSNR
        mse = np.mean((pred_frame - target_frame) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100.0  # Perfect reconstruction
        psnr_scores.append(psnr)
        
        # Calculate simplified SSIM (structural similarity)
        pred_flat = pred_frame.flatten()
        target_flat = target_frame.flatten()
        
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        ssim_scores.append(max(0.0, correlation))
        
        # Calculate color accuracy (for color images)
        if len(pred_frame.shape) == 3 and pred_frame.shape[2] >= 3:
            color_diff = np.mean(np.abs(pred_frame - target_frame), axis=(0, 1))
            color_accuracy = 1.0 - np.mean(color_diff) / 255.0
            color_accuracy_scores.append(max(0.0, color_accuracy))
    
    return {
        "PSNR": np.mean(psnr_scores) if psnr_scores else 0.0,
        "SSIM": np.mean(ssim_scores) if ssim_scores else 0.0,
        "ColorAccuracy": np.mean(color_accuracy_scores) if color_accuracy_scores else 1.0
    }