#!/usr/bin/env python3
"""
PNBTR Advanced Signal Analysis - Phase 3
Multi-resolution analysis, wavelet transforms, phase coherence, and transient detection.
Provides deep signal insights for training optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

# Optional advanced dependencies
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq, stft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class AdvancedSignalAnalyzer:
    """
    Advanced signal analysis suite for detailed audio signal characterization.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate // 2
        
        # Multi-resolution analysis parameters
        self.fft_sizes = [256, 512, 1024, 2048, 4096]
        self.hop_ratios = [0.25, 0.5, 0.75]  # Hop size as fraction of window
        
        # Wavelet-like analysis parameters
        self.octave_bands = self._create_octave_bands()
        
    def _create_octave_bands(self) -> List[Tuple[float, float]]:
        """Create octave band frequency ranges"""
        bands = []
        f_low = 62.5  # Start at 62.5 Hz
        
        while f_low < self.nyquist:
            f_high = f_low * 2
            if f_high > self.nyquist:
                f_high = self.nyquist
            bands.append((f_low, f_high))
            f_low = f_high
        
        return bands
    
    def multi_resolution_fft(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform multi-resolution FFT analysis with different window sizes.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary containing FFT results for each window size
        """
        results = {}
        
        for fft_size in self.fft_sizes:
            if len(signal_data) >= fft_size:
                # Use windowing for better spectral estimation
                window = np.hanning(fft_size)
                
                # Compute FFT on windowed signal
                windowed_signal = signal_data[:fft_size] * window
                fft_result = np.abs(fft(windowed_signal))[:fft_size//2]
                
                # Normalize by window energy
                window_energy = np.sum(window ** 2)
                if window_energy > 0:
                    fft_result = fft_result / np.sqrt(window_energy)
                
                # Create frequency axis
                freqs = fftfreq(fft_size, 1/self.sample_rate)[:fft_size//2]
                
                results[f'fft_{fft_size}'] = {
                    'magnitude': fft_result,
                    'frequencies': freqs,
                    'resolution_hz': self.sample_rate / fft_size
                }
            
        return results
    
    def short_time_fourier_transform(self, signal_data: np.ndarray, 
                                   window_size: int = 1024,
                                   hop_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute Short-Time Fourier Transform for time-frequency analysis.
        
        Args:
            signal_data: Input signal
            window_size: STFT window size
            hop_size: Hop size (defaults to window_size/4)
            
        Returns:
            Dictionary with STFT magnitude, phase, frequencies, and times
        """
        if hop_size is None:
            hop_size = window_size // 4
        
        if SCIPY_AVAILABLE:
            # Use scipy's implementation
            frequencies, times, stft_matrix = stft(
                signal_data, 
                fs=self.sample_rate,
                window='hann',
                nperseg=window_size,
                noverlap=window_size - hop_size
            )
            
            magnitude = np.abs(stft_matrix)
            phase = np.angle(stft_matrix)
            
        else:
            # Manual STFT implementation
            window = np.hanning(window_size)
            n_frames = (len(signal_data) - window_size) // hop_size + 1
            n_freqs = window_size // 2 + 1
            
            magnitude = np.zeros((n_freqs, n_frames))
            phase = np.zeros((n_freqs, n_frames))
            
            for frame in range(n_frames):
                start = frame * hop_size
                end = start + window_size
                
                if end <= len(signal_data):
                    windowed_frame = signal_data[start:end] * window
                    fft_frame = fft(windowed_frame)[:n_freqs]
                    
                    magnitude[:, frame] = np.abs(fft_frame)
                    phase[:, frame] = np.angle(fft_frame)
            
            frequencies = fftfreq(window_size, 1/self.sample_rate)[:n_freqs]
            times = np.arange(n_frames) * hop_size / self.sample_rate
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'frequencies': frequencies,
            'times': times,
            'window_size': window_size,
            'hop_size': hop_size
        }
    
    def phase_coherence_analysis(self, prediction: np.ndarray, 
                               target: np.ndarray) -> Dict[str, float]:
        """
        Analyze phase coherence between prediction and target signals.
        
        Args:
            prediction: Predicted signal
            target: Reference signal
            
        Returns:
            Dictionary of phase coherence metrics
        """
        # Ensure same length
        min_len = min(len(prediction), len(target))
        pred = prediction[:min_len]
        ref = target[:min_len]
        
        results = {}
        
        # Global phase coherence using cross-spectrum
        if SCIPY_AVAILABLE:
            # Use welch method for robust spectral estimation
            f, pred_psd = signal.welch(pred, fs=self.sample_rate, nperseg=1024)
            _, ref_psd = signal.welch(ref, fs=self.sample_rate, nperseg=1024)
            _, cross_psd = signal.csd(pred, ref, fs=self.sample_rate, nperseg=1024)
            
            # Coherence function
            coherence = np.abs(cross_psd) ** 2 / (pred_psd * ref_psd + 1e-12)
            
            # Average coherence across frequency bands
            results['global_coherence'] = float(np.mean(coherence))
            
            # Coherence in different frequency bands
            for i, (f_low, f_high) in enumerate(self.octave_bands):
                band_mask = (f >= f_low) & (f < f_high)
                if np.any(band_mask):
                    band_coherence = np.mean(coherence[band_mask])
                    results[f'coherence_band_{i}'] = float(band_coherence)
        
        # Time-domain phase analysis using analytic signal
        try:
            # Create analytic signals
            if SCIPY_AVAILABLE:
                pred_analytic = signal.hilbert(pred)
                ref_analytic = signal.hilbert(ref)
            else:
                # Simplified analytic signal approximation
                pred_fft = fft(pred)
                ref_fft = fft(ref)
                
                # Create analytic signals by zeroing negative frequencies
                n = len(pred_fft)
                pred_fft[n//2:] = 0
                ref_fft[n//2:] = 0
                pred_fft[1:n//2] *= 2
                ref_fft[1:n//2] *= 2
                
                pred_analytic = np.fft.ifft(pred_fft)
                ref_analytic = np.fft.ifft(ref_fft)
            
            # Instantaneous phase difference
            pred_phase = np.angle(pred_analytic)
            ref_phase = np.angle(ref_analytic)
            
            phase_diff = np.angle(np.exp(1j * (pred_phase - ref_phase)))
            
            # Phase coherence metrics
            results['mean_phase_error'] = float(np.mean(np.abs(phase_diff)))
            results['phase_stability'] = float(1.0 - np.std(phase_diff) / np.pi)
            results['phase_coherence_time'] = float(np.mean(np.cos(phase_diff)))
            
        except Exception as e:
            warnings.warn(f"Phase analysis failed: {e}")
            results.update({
                'mean_phase_error': 0.0,
                'phase_stability': 0.0,
                'phase_coherence_time': 0.0
            })
        
        return results
    
    def transient_detection(self, signal_data: np.ndarray, 
                           threshold_factor: float = 3.0) -> Dict[str, np.ndarray]:
        """
        Detect transients and attacks in the signal.
        
        Args:
            signal_data: Input signal
            threshold_factor: Threshold multiplier for transient detection
            
        Returns:
            Dictionary with transient locations and characteristics
        """
        # Compute energy envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms window
        
        if window_size >= len(signal_data):
            return {
                'transient_indices': np.array([]),
                'energy_envelope': np.abs(signal_data),
                'energy_derivative': np.zeros_like(signal_data),
                'n_transients': 0
            }
        
        # Energy computation using moving window
        energy_envelope = np.zeros(len(signal_data))
        
        for i in range(len(signal_data)):
            start = max(0, i - window_size//2)
            end = min(len(signal_data), i + window_size//2)
            energy_envelope[i] = np.mean(signal_data[start:end] ** 2)
        
        # Compute energy derivative (rate of change)
        energy_derivative = np.gradient(energy_envelope)
        
        # Detect transients as rapid energy increases
        energy_std = np.std(energy_derivative)
        threshold = threshold_factor * energy_std
        
        # Find peaks in energy derivative
        transient_candidates = np.where(energy_derivative > threshold)[0]
        
        # Group nearby transients and keep only the strongest
        transient_indices = []
        min_separation = int(0.05 * self.sample_rate)  # 50ms minimum separation
        
        if len(transient_candidates) > 0:
            current_group = [transient_candidates[0]]
            
            for idx in transient_candidates[1:]:
                if idx - current_group[-1] < min_separation:
                    current_group.append(idx)
                else:
                    # Find strongest transient in current group
                    group_energies = energy_derivative[current_group]
                    strongest_idx = current_group[np.argmax(group_energies)]
                    transient_indices.append(strongest_idx)
                    
                    current_group = [idx]
            
            # Handle last group
            if current_group:
                group_energies = energy_derivative[current_group]
                strongest_idx = current_group[np.argmax(group_energies)]
                transient_indices.append(strongest_idx)
        
        return {
            'transient_indices': np.array(transient_indices),
            'energy_envelope': energy_envelope,
            'energy_derivative': energy_derivative,
            'n_transients': len(transient_indices),
            'threshold_used': threshold
        }
    
    def wavelet_like_analysis(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform wavelet-like multi-resolution analysis using bandpass filters.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary with analysis results for each frequency band
        """
        results = {}
        
        if SCIPY_AVAILABLE:
            # Create bandpass filters for octave bands
            for i, (f_low, f_high) in enumerate(self.octave_bands):
                try:
                    # Design bandpass filter
                    nyquist = self.sample_rate / 2
                    low_norm = f_low / nyquist
                    high_norm = f_high / nyquist
                    
                    # Ensure normalized frequencies are valid
                    low_norm = max(0.01, min(0.99, low_norm))
                    high_norm = max(low_norm + 0.01, min(0.99, high_norm))
                    
                    # Butterworth bandpass filter
                    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    
                    # Apply filter
                    filtered_signal = signal.filtfilt(b, a, signal_data)
                    
                    # Compute envelope and energy
                    envelope = np.abs(signal.hilbert(filtered_signal))
                    total_energy = np.sum(filtered_signal ** 2)
                    
                    results[f'band_{i}'] = {
                        'frequency_range': (f_low, f_high),
                        'filtered_signal': filtered_signal,
                        'envelope': envelope,
                        'total_energy': total_energy,
                        'peak_envelope': np.max(envelope),
                        'mean_envelope': np.mean(envelope)
                    }
                    
                except Exception as e:
                    warnings.warn(f"Filter design failed for band {i}: {e}")
                    continue
        else:
            # Simplified implementation using FFT-based filtering
            fft_signal = fft(signal_data)
            freqs = fftfreq(len(signal_data), 1/self.sample_rate)
            
            for i, (f_low, f_high) in enumerate(self.octave_bands):
                # Create frequency mask
                mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
                
                # Apply mask and inverse FFT
                filtered_fft = fft_signal.copy()
                filtered_fft[~mask] = 0
                filtered_signal = np.real(np.fft.ifft(filtered_fft))
                
                # Simple envelope estimation
                envelope = np.abs(filtered_signal)
                total_energy = np.sum(filtered_signal ** 2)
                
                results[f'band_{i}'] = {
                    'frequency_range': (f_low, f_high),
                    'filtered_signal': filtered_signal,
                    'envelope': envelope,
                    'total_energy': total_energy,
                    'peak_envelope': np.max(envelope),
                    'mean_envelope': np.mean(envelope)
                }
        
        return results
    
    def temporal_structure_analysis(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze temporal structure and rhythmic content.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of temporal structure metrics
        """
        results = {}
        
        # Onset detection using spectral flux
        if SCIPY_AVAILABLE and len(signal_data) > 2048:
            # STFT for onset detection
            _, _, stft_matrix = stft(signal_data, fs=self.sample_rate, nperseg=1024, noverlap=512)
            magnitude = np.abs(stft_matrix)
            
            # Spectral flux (frame-to-frame spectral change)
            spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
            
            # Peak picking for onset detection
            onset_threshold = np.mean(spectral_flux) + 2 * np.std(spectral_flux)
            onset_frames = np.where(spectral_flux > onset_threshold)[0]
            
            # Convert to time
            frame_times = np.arange(len(spectral_flux)) * 512 / self.sample_rate
            onset_times = frame_times[onset_frames]
            
            # Temporal metrics
            if len(onset_times) > 1:
                inter_onset_intervals = np.diff(onset_times)
                results['onset_density'] = len(onset_times) / (len(signal_data) / self.sample_rate)
                results['mean_ioi'] = float(np.mean(inter_onset_intervals))
                results['ioi_variability'] = float(np.std(inter_onset_intervals))
                results['n_onsets'] = len(onset_times)
            else:
                results.update({
                    'onset_density': 0.0,
                    'mean_ioi': 0.0,
                    'ioi_variability': 0.0,
                    'n_onsets': len(onset_times)
                })
        else:
            # Simplified onset detection using energy
            window_size = int(0.02 * self.sample_rate)  # 20ms
            energy = np.array([
                np.sum(signal_data[i:i+window_size] ** 2)
                for i in range(0, len(signal_data) - window_size, window_size//2)
            ])
            
            energy_diff = np.diff(energy)
            threshold = np.mean(energy_diff) + 2 * np.std(energy_diff)
            onsets = np.where(energy_diff > threshold)[0]
            
            results['onset_density'] = len(onsets) / (len(signal_data) / self.sample_rate)
            results['n_onsets'] = len(onsets)
            results['mean_ioi'] = 0.0
            results['ioi_variability'] = 0.0
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        results['zero_crossing_rate'] = len(zero_crossings) / (len(signal_data) / self.sample_rate)
        
        # Dynamic range
        results['dynamic_range_db'] = 20 * np.log10(np.max(np.abs(signal_data)) / (np.mean(np.abs(signal_data)) + 1e-12))
        
        return results
    
    def comprehensive_analysis(self, signal_data: np.ndarray) -> Dict[str, any]:
        """
        Perform comprehensive signal analysis combining all methods.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        try:
            # Multi-resolution FFT
            results['multi_resolution_fft'] = self.multi_resolution_fft(signal_data)
            
            # STFT analysis
            results['stft'] = self.short_time_fourier_transform(signal_data)
            
            # Transient detection
            results['transients'] = self.transient_detection(signal_data)
            
            # Wavelet-like analysis
            results['wavelet_bands'] = self.wavelet_like_analysis(signal_data)
            
            # Temporal structure
            results['temporal_structure'] = self.temporal_structure_analysis(signal_data)
            
            # Summary statistics
            results['summary'] = {
                'signal_length_samples': len(signal_data),
                'signal_length_seconds': len(signal_data) / self.sample_rate,
                'rms_level': float(np.sqrt(np.mean(signal_data ** 2))),
                'peak_level': float(np.max(np.abs(signal_data))),
                'crest_factor': float(np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data ** 2)) + 1e-12)),
                'sample_rate': self.sample_rate
            }
            
        except Exception as e:
            warnings.warn(f"Comprehensive analysis failed: {e}")
            results['error'] = str(e)
        
        return results

def create_signal_analyzer(sample_rate: int = 48000) -> AdvancedSignalAnalyzer:
    """Factory function to create signal analyzer"""
    return AdvancedSignalAnalyzer(sample_rate)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Advanced Signal Analysis Test")
    
    # Create test signal
    duration = 2.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Complex test signal with multiple components
    signal_data = (
        0.5 * np.sin(2 * np.pi * 440 * t) +                    # Fundamental
        0.2 * np.sin(2 * np.pi * 880 * t) +                    # Harmonic
        0.1 * np.sin(2 * np.pi * 1200 * t) +                   # Higher harmonic
        0.05 * np.random.normal(0, 1, len(t))                   # Noise
    )
    
    # Add some transients
    transient_times = [0.5, 1.0, 1.5]
    for t_transient in transient_times:
        idx = int(t_transient * sample_rate)
        if idx < len(signal_data):
            signal_data[idx:idx+100] += 0.3 * np.exp(-np.arange(100) / 20)
    
    # Create analyzer
    analyzer = create_signal_analyzer(sample_rate)
    
    # Perform comprehensive analysis
    analysis = analyzer.comprehensive_analysis(signal_data)
    
    print("\n📊 Signal Analysis Results:")
    print("=" * 40)
    
    # Summary
    summary = analysis['summary']
    print(f"Signal length: {summary['signal_length_seconds']:.2f}s ({summary['signal_length_samples']} samples)")
    print(f"RMS level: {summary['rms_level']:.4f}")
    print(f"Peak level: {summary['peak_level']:.4f}")
    print(f"Crest factor: {summary['crest_factor']:.2f}")
    
    # Transients
    transients = analysis['transients']
    print(f"\nTransients detected: {transients['n_transients']}")
    if transients['n_transients'] > 0:
        transient_times_detected = transients['transient_indices'] / sample_rate
        print(f"Transient times: {transient_times_detected}")
    
    # Temporal structure
    temporal = analysis['temporal_structure']
    print(f"\nTemporal structure:")
    print(f"  Onset density: {temporal['onset_density']:.2f} onsets/sec")
    print(f"  Zero crossing rate: {temporal['zero_crossing_rate']:.1f} Hz")
    print(f"  Dynamic range: {temporal['dynamic_range_db']:.1f} dB")
    
    # Multi-resolution FFT
    print(f"\nMulti-resolution FFT:")
    for fft_key, fft_data in analysis['multi_resolution_fft'].items():
        resolution = fft_data['resolution_hz']
        peak_freq_idx = np.argmax(fft_data['magnitude'])
        peak_freq = fft_data['frequencies'][peak_freq_idx]
        print(f"  {fft_key}: {resolution:.1f} Hz resolution, peak at {peak_freq:.1f} Hz")
    
    print(f"\nSciPy available: {SCIPY_AVAILABLE}")
    print("✅ Advanced signal analysis test complete") 