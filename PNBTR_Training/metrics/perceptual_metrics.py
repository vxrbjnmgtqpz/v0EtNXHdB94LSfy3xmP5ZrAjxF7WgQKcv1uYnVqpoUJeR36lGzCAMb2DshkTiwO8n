#!/usr/bin/env python3
"""
PNBTR Perceptual Metrics - Phase 3
Advanced perceptual quality metrics for audio signal reconstruction.
Implements industry-standard and research-grade perceptual evaluations.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Optional dependencies for advanced metrics
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class PerceptualMetrics:
    """
    Comprehensive perceptual metrics suite for audio quality assessment.
    Includes STOI, PESQ-like scoring, spectral analysis, and harmonic content metrics.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate // 2
        
        # Perceptual frequency bands (Bark scale approximation)
        self.bark_bands = self._create_bark_bands()
        
        # Psychoacoustic masking thresholds
        self.masking_thresholds = self._create_masking_thresholds()
        
    def _create_bark_bands(self) -> np.ndarray:
        """Create Bark scale frequency bands for perceptual analysis"""
        # Bark scale: f_bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f/7500)^2)
        freqs = np.linspace(0, self.nyquist, 1000)
        bark_freqs = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Create 24 Bark bands
        bark_edges = np.linspace(0, 24, 25)
        freq_edges = []
        
        for bark_edge in bark_edges:
            # Find frequency closest to this Bark value
            idx = np.argmin(np.abs(bark_freqs - bark_edge))
            freq_edges.append(freqs[idx])
        
        return np.array(freq_edges)
    
    def _create_masking_thresholds(self) -> np.ndarray:
        """Create psychoacoustic masking thresholds"""
        # Simplified absolute threshold of hearing (ATH)
        freqs = np.logspace(1, np.log10(self.nyquist), 1000)
        
        # ATH formula (simplified version)
        ath_db = (
            3.64 * (freqs / 1000) ** -0.8 
            - 6.5 * np.exp(-0.6 * (freqs / 1000 - 3.3) ** 2)
            + 1e-3 * (freqs / 1000) ** 4
        )
        
        return 10 ** (ath_db / 20)  # Convert to linear scale
    
    def compute_stoi(self, prediction: np.ndarray, target: np.ndarray, 
                     frame_length: int = 256) -> float:
        """
        Compute Short-Time Objective Intelligibility (STOI) score.
        
        Args:
            prediction: Predicted signal
            target: Clean reference signal
            frame_length: Frame length for short-time analysis
            
        Returns:
            STOI score (0-1, higher is better)
        """
        # Ensure signals have same length
        min_len = min(len(prediction), len(target))
        pred = prediction[:min_len]
        ref = target[:min_len]
        
        # Frame-based analysis
        hop_length = frame_length // 2
        n_frames = (len(pred) - frame_length) // hop_length + 1
        
        correlations = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            
            pred_frame = pred[start:end]
            ref_frame = ref[start:end]
            
            # Remove DC component
            pred_frame = pred_frame - np.mean(pred_frame)
            ref_frame = ref_frame - np.mean(ref_frame)
            
            # Compute correlation in frequency domain per band
            if SCIPY_AVAILABLE:
                # Use proper STFT for better frequency resolution
                _, _, pred_stft = signal.stft(pred_frame, fs=self.sample_rate, 
                                            nperseg=64, noverlap=32)
                _, _, ref_stft = signal.stft(ref_frame, fs=self.sample_rate,
                                           nperseg=64, noverlap=32)
                
                # Compute correlation per frequency band
                for band_idx in range(len(self.bark_bands) - 1):
                    band_start = int(self.bark_bands[band_idx] / self.nyquist * len(pred_stft))
                    band_end = int(self.bark_bands[band_idx + 1] / self.nyquist * len(pred_stft))
                    
                    if band_end > band_start:
                        pred_band = np.abs(pred_stft[band_start:band_end]).flatten()
                        ref_band = np.abs(ref_stft[band_start:band_end]).flatten()
                        
                        if len(pred_band) > 1 and np.std(ref_band) > 1e-8:
                            corr = np.corrcoef(pred_band, ref_band)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(max(0, corr))
            else:
                # Fallback to simple time-domain correlation
                if np.std(ref_frame) > 1e-8:
                    corr = np.corrcoef(pred_frame, ref_frame)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(max(0, corr))
        
        if not correlations:
            return 0.0
        
        # STOI is the mean correlation across all bands and frames
        stoi_score = np.mean(correlations)
        return float(np.clip(stoi_score, 0, 1))
    
    def compute_pesq_like(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute PESQ-like perceptual score.
        Simplified version of PESQ focusing on perceptual distortion.
        
        Args:
            prediction: Predicted signal
            target: Clean reference signal
            
        Returns:
            PESQ-like score (1-5, higher is better)
        """
        # Ensure signals have same length
        min_len = min(len(prediction), len(target))
        pred = prediction[:min_len]
        ref = target[:min_len]
        
        # Compute power spectral densities
        if SCIPY_AVAILABLE:
            freqs, pred_psd = signal.welch(pred, fs=self.sample_rate, nperseg=1024)
            _, ref_psd = signal.welch(ref, fs=self.sample_rate, nperseg=1024)
        else:
            # Fallback FFT-based PSD
            n_fft = 1024
            pred_fft = np.abs(fft(pred[:n_fft])) ** 2
            ref_fft = np.abs(fft(ref[:n_fft])) ** 2
            freqs = fftfreq(n_fft, 1/self.sample_rate)[:n_fft//2]
            pred_psd = pred_fft[:n_fft//2]
            ref_psd = ref_fft[:n_fft//2]
        
        # Perceptual weighting using Bark bands
        perceptual_error = 0.0
        total_weight = 0.0
        
        for i in range(len(self.bark_bands) - 1):
            # Find frequency indices for this Bark band
            band_start = self.bark_bands[i]
            band_end = self.bark_bands[i + 1]
            
            band_mask = (freqs >= band_start) & (freqs < band_end)
            
            if np.any(band_mask):
                # Average power in this band
                pred_power = np.mean(pred_psd[band_mask])
                ref_power = np.mean(ref_psd[band_mask])
                
                if ref_power > 1e-12:  # Avoid division by zero
                    # Log spectral distance with perceptual weighting
                    log_ratio = np.log10(max(pred_power, 1e-12) / ref_power)
                    
                    # Weight based on reference power and frequency importance
                    freq_weight = 1.0 / (1.0 + (band_start / 4000) ** 2)  # Emphasize mid frequencies
                    power_weight = np.log10(ref_power + 1e-12) + 120  # dB-like scaling
                    weight = max(0.1, freq_weight * power_weight)
                    
                    perceptual_error += weight * log_ratio ** 2
                    total_weight += weight
        
        if total_weight > 0:
            mse_log = perceptual_error / total_weight
            # Convert to PESQ-like scale (1-5)
            pesq_score = 5.0 - 4.0 * np.tanh(mse_log / 2.0)
        else:
            pesq_score = 1.0
        
        return float(np.clip(pesq_score, 1.0, 5.0))
    
    def compute_spectral_centroid(self, signal_data: np.ndarray) -> float:
        """Compute spectral centroid (brightness measure)"""
        if SCIPY_AVAILABLE:
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate, nperseg=1024)
        else:
            n_fft = min(1024, len(signal_data))
            fft_data = np.abs(fft(signal_data[:n_fft])) ** 2
            freqs = fftfreq(n_fft, 1/self.sample_rate)[:n_fft//2]
            psd = fft_data[:n_fft//2]
        
        # Spectral centroid = weighted average frequency
        if np.sum(psd) > 1e-12:
            centroid = np.sum(freqs * psd) / np.sum(psd)
        else:
            centroid = 0.0
        
        return float(centroid)
    
    def compute_spectral_rolloff(self, signal_data: np.ndarray, 
                                rolloff_threshold: float = 0.85) -> float:
        """Compute spectral rolloff frequency"""
        if SCIPY_AVAILABLE:
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate, nperseg=1024)
        else:
            n_fft = min(1024, len(signal_data))
            fft_data = np.abs(fft(signal_data[:n_fft])) ** 2
            freqs = fftfreq(n_fft, 1/self.sample_rate)[:n_fft//2]
            psd = fft_data[:n_fft//2]
        
        # Find frequency where cumulative energy reaches threshold
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        
        if total_energy > 1e-12:
            threshold_energy = rolloff_threshold * total_energy
            rolloff_idx = np.where(cumulative_energy >= threshold_energy)[0]
            
            if len(rolloff_idx) > 0:
                rolloff_freq = freqs[rolloff_idx[0]]
            else:
                rolloff_freq = freqs[-1]
        else:
            rolloff_freq = 0.0
        
        return float(rolloff_freq)
    
    def compute_harmonic_ratio(self, signal_data: np.ndarray) -> float:
        """
        Compute harmonic-to-noise ratio (HNR) for tonal content analysis.
        
        Returns:
            HNR in dB (higher values indicate more harmonic content)
        """
        if len(signal_data) < 1024:
            return 0.0
        
        # Use autocorrelation to find periodicity
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        if autocorr[0] > 1e-12:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0
        
        # Find the fundamental period (excluding the zero-lag peak)
        min_period = int(self.sample_rate / 800)  # 800 Hz max fundamental
        max_period = int(self.sample_rate / 50)   # 50 Hz min fundamental
        
        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1
        
        if min_period >= max_period:
            return 0.0
        
        # Find peak in autocorrelation (fundamental period)
        search_range = autocorr[min_period:max_period]
        if len(search_range) == 0:
            return 0.0
        
        peak_idx = np.argmax(search_range) + min_period
        peak_value = autocorr[peak_idx]
        
        # Estimate noise floor from autocorr values away from harmonics
        noise_indices = []
        for i in range(min_period, len(autocorr)):
            # Skip harmonic peaks
            is_harmonic = False
            for harmonic in range(1, 6):  # Check first 5 harmonics
                harmonic_period = peak_idx / harmonic
                if abs(i - harmonic_period) < 3:  # 3-sample tolerance
                    is_harmonic = True
                    break
            
            if not is_harmonic:
                noise_indices.append(i)
        
        if noise_indices:
            noise_floor = np.mean(autocorr[noise_indices[:100]])  # Sample noise floor
        else:
            noise_floor = 0.01  # Default noise floor
        
        # HNR calculation
        if noise_floor > 1e-12 and peak_value > noise_floor:
            hnr_db = 20 * np.log10(peak_value / noise_floor)
        else:
            hnr_db = 0.0
        
        return float(np.clip(hnr_db, 0, 40))  # Clip to reasonable range
    
    def compute_spectral_flatness(self, signal_data: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy) - measure of spectral shape.
        Values near 1 indicate noise-like signals, near 0 indicate tonal signals.
        """
        if SCIPY_AVAILABLE:
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate, nperseg=1024)
        else:
            n_fft = min(1024, len(signal_data))
            fft_data = np.abs(fft(signal_data[:n_fft])) ** 2
            psd = fft_data[:n_fft//2]
        
        # Remove DC and very low frequencies
        psd = psd[1:]
        
        if len(psd) == 0 or np.any(psd <= 0):
            return 0.0
        
        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
        arithmetic_mean = np.mean(psd)
        
        if arithmetic_mean > 1e-12:
            flatness = geometric_mean / arithmetic_mean
        else:
            flatness = 0.0
        
        return float(np.clip(flatness, 0, 1))
    
    def evaluate_all_metrics(self, prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute all perceptual metrics for a prediction-target pair.
        
        Args:
            prediction: Predicted signal
            target: Clean reference signal
            
        Returns:
            Dictionary of all perceptual metric scores
        """
        results = {}
        
        try:
            # Intelligibility and quality metrics
            results['STOI'] = self.compute_stoi(prediction, target)
            results['PESQ_like'] = self.compute_pesq_like(prediction, target)
            
            # Spectral characteristics of prediction
            results['SpectralCentroid_pred'] = self.compute_spectral_centroid(prediction)
            results['SpectralRolloff_pred'] = self.compute_spectral_rolloff(prediction)
            results['HarmonicRatio_pred'] = self.compute_harmonic_ratio(prediction)
            results['SpectralFlatness_pred'] = self.compute_spectral_flatness(prediction)
            
            # Spectral characteristics of target (for comparison)
            results['SpectralCentroid_target'] = self.compute_spectral_centroid(target)
            results['SpectralRolloff_target'] = self.compute_spectral_rolloff(target)
            results['HarmonicRatio_target'] = self.compute_harmonic_ratio(target)
            results['SpectralFlatness_target'] = self.compute_spectral_flatness(target)
            
            # Spectral similarity metrics
            results['CentroidError'] = abs(results['SpectralCentroid_pred'] - results['SpectralCentroid_target'])
            results['RolloffError'] = abs(results['SpectralRolloff_pred'] - results['SpectralRolloff_target'])
            results['HarmonicError'] = abs(results['HarmonicRatio_pred'] - results['HarmonicRatio_target'])
            results['FlatnessError'] = abs(results['SpectralFlatness_pred'] - results['SpectralFlatness_target'])
            
        except Exception as e:
            warnings.warn(f"Error computing perceptual metrics: {e}")
            # Return default values
            for key in ['STOI', 'PESQ_like', 'SpectralCentroid_pred', 'SpectralRolloff_pred', 
                       'HarmonicRatio_pred', 'SpectralFlatness_pred', 'SpectralCentroid_target',
                       'SpectralRolloff_target', 'HarmonicRatio_target', 'SpectralFlatness_target',
                       'CentroidError', 'RolloffError', 'HarmonicError', 'FlatnessError']:
                if key not in results:
                    results[key] = 0.0
        
        return results

def create_perceptual_evaluator(sample_rate: int = 48000) -> PerceptualMetrics:
    """Factory function to create perceptual metrics evaluator"""
    return PerceptualMetrics(sample_rate)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Perceptual Metrics Test")
    
    # Create test signals
    duration = 1.0  # 1 second
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Target: clean harmonic signal
    target = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    
    # Prediction: slightly distorted version
    prediction = target + 0.05 * np.random.normal(0, 1, len(target))
    
    # Create evaluator
    evaluator = create_perceptual_evaluator(sample_rate)
    
    # Compute all metrics
    metrics = evaluator.evaluate_all_metrics(prediction, target)
    
    print("\n📊 Perceptual Metrics Results:")
    print("=" * 40)
    for metric, value in metrics.items():
        if 'Error' in metric:
            print(f"{metric:20}: {value:8.3f}")
        elif metric in ['STOI', 'SpectralFlatness_pred', 'SpectralFlatness_target']:
            print(f"{metric:20}: {value:8.3f}")
        elif metric == 'PESQ_like':
            print(f"{metric:20}: {value:8.2f}")
        elif 'Freq' in metric or 'Centroid' in metric or 'Rolloff' in metric:
            print(f"{metric:20}: {value:8.0f} Hz")
        elif 'Harmonic' in metric:
            print(f"{metric:20}: {value:8.2f} dB")
        else:
            print(f"{metric:20}: {value:8.3f}")
    
    print(f"\nSciPy available: {SCIPY_AVAILABLE}")
    print("✅ Perceptual metrics test complete") 