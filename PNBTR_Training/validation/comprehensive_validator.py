#!/usr/bin/env python3
"""
PNBTR Comprehensive Validation Framework
Addresses 250708_093109_System_Audit.md requirements

This module provides comprehensive validation tools for the PNBTR system:
- Real-time processing validation
- Audio quality standards compliance
- Frequency response testing
- Multi-modal data handling
- Automated test suite generation
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.loss_functions import (
    evaluate_metrics, validate_realtime_processing, 
    evaluate_video_metrics, check_hifi_standards
)

class PNBTRValidator:
    """
    Comprehensive validation framework for PNBTR system.
    """
    
    def __init__(self, output_dir: str = "validation_reports"):
        """
        Initialize the validator.
        
        Args:
            output_dir: Directory to save validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PNBTRValidator')
        
        # Validation configuration
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict:
        """Load default validation configuration."""
        return {
            "audio_tests": {
                "sample_rates": [44100, 48000, 96000, 192000],
                "bit_depths": [16, 24, 32],
                "test_signals": ["sine", "white_noise", "pink_noise", "chirp", "music"],
                "frequency_range": {"min": 20, "max": 20000},
                "thd_threshold": 0.1,  # 0.1% THD+N for hi-fi
                "snr_threshold": 60.0,  # 60dB SNR minimum
                "freq_response_tolerance": 1.0  # ±1dB frequency response
            },
            "realtime_tests": {
                "sample_rates": [44100, 48000],
                "buffer_sizes": [64, 128, 256, 512, 1024],
                "target_latency_ms": 10.0,
                "cpu_load_threshold": 50.0  # 50% max CPU usage
            },
            "multimodal_tests": {
                "video_formats": ["480p", "720p", "1080p"],
                "frame_rates": [24, 30, 60],
                "color_spaces": ["RGB", "YUV"],
                "psnr_threshold": 30.0,  # Minimum PSNR for acceptable quality
                "ssim_threshold": 0.9   # Minimum SSIM for acceptable quality
            }
        }
    
    def run_comprehensive_validation(self, model_function, test_data: Dict) -> Dict:
        """
        Run comprehensive validation suite.
        
        Args:
            model_function: PNBTR model function to test
            test_data: Dictionary containing test datasets
            
        Returns:
            Dict: Comprehensive validation results
        """
        self.logger.info("Starting comprehensive PNBTR validation")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "audio_quality": self._validate_audio_quality(model_function, test_data.get("audio", {})),
            "realtime_performance": self._validate_realtime_performance(model_function),
            "frequency_response": self._validate_frequency_response(model_function),
            "multimodal_handling": self._validate_multimodal_handling(model_function, test_data.get("video", {})),
            "hifi_compliance": None,  # Will be set after other tests
            "overall_score": 0.0
        }
        
        # Calculate hi-fi compliance
        validation_results["hifi_compliance"] = self._assess_hifi_compliance(validation_results)
        
        # Calculate overall score
        validation_results["overall_score"] = self._calculate_overall_score(validation_results)
        
        # Generate report
        report_path = self._generate_validation_report(validation_results)
        validation_results["report_path"] = str(report_path)
        
        self.logger.info(f"Validation complete. Overall score: {validation_results['overall_score']:.2f}")
        self.logger.info(f"Report saved to: {report_path}")
        
        return validation_results
    
    def _validate_audio_quality(self, model_function, audio_test_data: Dict) -> Dict:
        """Validate audio quality metrics."""
        self.logger.info("Validating audio quality...")
        
        results = {
            "test_results": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "average_snr": 0.0,
                "average_thd_n": 0.0,
                "average_coloration": 0.0,
                "meets_hifi_standards": False
            }
        }
        
        # Test different sample rates and signals
        for sample_rate in self.config["audio_tests"]["sample_rates"]:
            for signal_type in self.config["audio_tests"]["test_signals"]:
                
                # Generate test signal
                test_signal = self._generate_test_signal(signal_type, sample_rate)
                
                # Process with model
                try:
                    processed_signal = model_function(test_signal)
                    
                    # Evaluate metrics
                    metrics = evaluate_metrics(
                        processed_signal, test_signal, 
                        sample_rate=sample_rate, 
                        enhanced_analysis=True
                    )
                    
                    # Check if passes thresholds
                    passes_thresholds = (
                        metrics.get("SDR", 0) * 60 >= self.config["audio_tests"]["snr_threshold"] and
                        metrics.get("THD_N_Percent", 100) <= self.config["audio_tests"]["thd_threshold"] and
                        abs(metrics.get("FreqResponseFlatness", -60)) <= self.config["audio_tests"]["freq_response_tolerance"]
                    )
                    
                    test_result = {
                        "sample_rate": sample_rate,
                        "signal_type": signal_type,
                        "metrics": metrics,
                        "passes_thresholds": passes_thresholds,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results["test_results"].append(test_result)
                    results["summary"]["total_tests"] += 1
                    
                    if passes_thresholds:
                        results["summary"]["passed_tests"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {signal_type} at {sample_rate}Hz: {e}")
                    
        # Calculate summary statistics
        if results["test_results"]:
            all_metrics = [test["metrics"] for test in results["test_results"]]
            
            # Average metrics
            results["summary"]["average_snr"] = np.mean([
                m.get("SDR", 0) * 60 for m in all_metrics
            ])
            results["summary"]["average_thd_n"] = np.mean([
                m.get("THD_N_Percent", 100) for m in all_metrics
            ])
            results["summary"]["average_coloration"] = np.mean([
                m.get("ColorationPercent", 100) for m in all_metrics
            ])
            
            # Check if meets hi-fi standards
            hifi_compliant_tests = [
                m.get("MeetsHiFiStandards", False) for m in all_metrics
            ]
            results["summary"]["meets_hifi_standards"] = np.mean(hifi_compliant_tests) >= 0.8
        
        return results
    
    def _validate_realtime_performance(self, model_function) -> Dict:
        """Validate real-time processing performance."""
        self.logger.info("Validating real-time performance...")
        
        results = {
            "test_results": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "average_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "average_cpu_load": 0.0,
                "meets_realtime_requirements": False
            }
        }
        
        # Test different buffer sizes and sample rates
        for sample_rate in self.config["realtime_tests"]["sample_rates"]:
            for buffer_size in self.config["realtime_tests"]["buffer_sizes"]:
                
                try:
                    # Validate real-time processing
                    rt_results = validate_realtime_processing(
                        model_function,
                        sample_rate=sample_rate,
                        buffer_size=buffer_size,
                        target_latency_ms=self.config["realtime_tests"]["target_latency_ms"]
                    )
                    
                    meets_requirements = (
                        rt_results["meets_realtime"] and
                        rt_results["cpu_load_percent"] <= self.config["realtime_tests"]["cpu_load_threshold"]
                    )
                    
                    test_result = {
                        "sample_rate": sample_rate,
                        "buffer_size": buffer_size,
                        "realtime_metrics": rt_results,
                        "meets_requirements": meets_requirements,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results["test_results"].append(test_result)
                    results["summary"]["total_tests"] += 1
                    
                    if meets_requirements:
                        results["summary"]["passed_tests"] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in real-time test {sample_rate}Hz/{buffer_size}: {e}")
        
        # Calculate summary statistics
        if results["test_results"]:
            all_rt_metrics = [test["realtime_metrics"] for test in results["test_results"]]
            
            results["summary"]["average_latency_ms"] = np.mean([
                m["mean_latency_ms"] for m in all_rt_metrics
            ])
            results["summary"]["max_latency_ms"] = np.max([
                m["max_latency_ms"] for m in all_rt_metrics
            ])
            results["summary"]["average_cpu_load"] = np.mean([
                m["cpu_load_percent"] for m in all_rt_metrics
            ])
            
            results["summary"]["meets_realtime_requirements"] = (
                results["summary"]["passed_tests"] / results["summary"]["total_tests"] >= 0.8
            )
        
        return results
    
    def _validate_frequency_response(self, model_function) -> Dict:
        """Validate frequency response characteristics."""
        self.logger.info("Validating frequency response...")
        
        results = {
            "frequency_sweeps": [],
            "phase_linearity": [],
            "summary": {
                "max_deviation_db": 0.0,
                "avg_phase_linearity": 0.0,
                "meets_flatness_requirement": False
            }
        }
        
        # Test frequency sweeps at different sample rates
        for sample_rate in self.config["audio_tests"]["sample_rates"]:
            
            # Generate frequency sweep
            sweep_signal = self._generate_frequency_sweep(sample_rate)
            
            try:
                # Process with model
                processed_sweep = model_function(sweep_signal)
                
                # Analyze frequency response
                freq_response = self._analyze_frequency_response(
                    sweep_signal, processed_sweep, sample_rate
                )
                
                # Analyze phase linearity
                phase_metrics = self._analyze_phase_linearity(
                    sweep_signal, processed_sweep, sample_rate
                )
                
                results["frequency_sweeps"].append({
                    "sample_rate": sample_rate,
                    "frequency_response": freq_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                results["phase_linearity"].append({
                    "sample_rate": sample_rate,
                    "phase_metrics": phase_metrics,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in frequency response test at {sample_rate}Hz: {e}")
        
        # Calculate summary
        if results["frequency_sweeps"]:
            all_deviations = []
            all_phase_linearity = []
            
            for sweep in results["frequency_sweeps"]:
                deviations = [abs(point[1]) for point in sweep["frequency_response"]]
                all_deviations.extend(deviations)
            
            for phase_test in results["phase_linearity"]:
                all_phase_linearity.append(phase_test["phase_metrics"]["linearity_score"])
            
            results["summary"]["max_deviation_db"] = max(all_deviations) if all_deviations else 0.0
            results["summary"]["avg_phase_linearity"] = np.mean(all_phase_linearity) if all_phase_linearity else 0.0
            
            results["summary"]["meets_flatness_requirement"] = (
                results["summary"]["max_deviation_db"] <= self.config["audio_tests"]["freq_response_tolerance"]
            )
        
        return results
    
    def _validate_multimodal_handling(self, model_function, video_test_data: Dict) -> Dict:
        """Validate multi-modal (audio/video) data handling."""
        self.logger.info("Validating multi-modal handling...")
        
        results = {
            "video_tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "average_psnr": 0.0,
                "average_ssim": 0.0,
                "meets_quality_thresholds": False
            }
        }
        
        # If no video test data provided, generate synthetic data
        if not video_test_data:
            self.logger.info("No video test data provided, generating synthetic test frames")
            video_test_data = self._generate_synthetic_video_data()
        
        # Test video processing
        for format_name, test_frames in video_test_data.items():
            try:
                # Convert video frames to audio-like stream (simulate Jvid processing)
                video_stream = self._frames_to_stream(test_frames)
                
                # Process with model
                processed_stream = model_function(video_stream)
                
                # Convert back to frames
                processed_frames = self._stream_to_frames(processed_stream, test_frames[0].shape)
                
                # Evaluate video metrics
                video_metrics = evaluate_video_metrics(processed_frames, test_frames)
                
                meets_thresholds = (
                    video_metrics["PSNR"] >= self.config["multimodal_tests"]["psnr_threshold"] and
                    video_metrics["SSIM"] >= self.config["multimodal_tests"]["ssim_threshold"]
                )
                
                test_result = {
                    "format": format_name,
                    "metrics": video_metrics,
                    "meets_thresholds": meets_thresholds,
                    "timestamp": datetime.now().isoformat()
                }
                
                results["video_tests"].append(test_result)
                results["summary"]["total_tests"] += 1
                
                if meets_thresholds:
                    results["summary"]["passed_tests"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error in video test {format_name}: {e}")
        
        # Calculate summary
        if results["video_tests"]:
            all_metrics = [test["metrics"] for test in results["video_tests"]]
            
            results["summary"]["average_psnr"] = np.mean([m["PSNR"] for m in all_metrics])
            results["summary"]["average_ssim"] = np.mean([m["SSIM"] for m in all_metrics])
            
            results["summary"]["meets_quality_thresholds"] = (
                results["summary"]["passed_tests"] / results["summary"]["total_tests"] >= 0.8
            )
        
        return results
    
    def _assess_hifi_compliance(self, validation_results: Dict) -> Dict:
        """Assess overall hi-fi compliance."""
        audio_results = validation_results.get("audio_quality", {})
        rt_results = validation_results.get("realtime_performance", {})
        freq_results = validation_results.get("frequency_response", {})
        
        compliance = {
            "audio_quality_compliant": audio_results.get("summary", {}).get("meets_hifi_standards", False),
            "realtime_compliant": rt_results.get("summary", {}).get("meets_realtime_requirements", False),
            "frequency_response_compliant": freq_results.get("summary", {}).get("meets_flatness_requirement", False),
            "overall_compliant": False
        }
        
        # Overall compliance requires all individual criteria to be met
        compliance["overall_compliant"] = (
            compliance["audio_quality_compliant"] and
            compliance["realtime_compliant"] and
            compliance["frequency_response_compliant"]
        )
        
        return compliance
    
    def _calculate_overall_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score."""
        scores = []
        
        # Audio quality score (40% weight)
        audio_summary = validation_results.get("audio_quality", {}).get("summary", {})
        if audio_summary.get("total_tests", 0) > 0:
            audio_score = audio_summary.get("passed_tests", 0) / audio_summary.get("total_tests", 1)
            scores.append(("audio", audio_score, 0.4))
        
        # Real-time performance score (30% weight)
        rt_summary = validation_results.get("realtime_performance", {}).get("summary", {})
        if rt_summary.get("total_tests", 0) > 0:
            rt_score = rt_summary.get("passed_tests", 0) / rt_summary.get("total_tests", 1)
            scores.append(("realtime", rt_score, 0.3))
        
        # Frequency response score (20% weight)
        freq_summary = validation_results.get("frequency_response", {}).get("summary", {})
        freq_score = 1.0 if freq_summary.get("meets_flatness_requirement", False) else 0.5
        scores.append(("frequency", freq_score, 0.2))
        
        # Multi-modal score (10% weight)
        mm_summary = validation_results.get("multimodal_handling", {}).get("summary", {})
        if mm_summary.get("total_tests", 0) > 0:
            mm_score = mm_summary.get("passed_tests", 0) / mm_summary.get("total_tests", 1)
            scores.append(("multimodal", mm_score, 0.1))
        
        # Calculate weighted average
        if scores:
            total_weighted_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            return total_weighted_score / total_weight if total_weight > 0 else 0.0
        else:
            return 0.0
    
    def _generate_validation_report(self, validation_results: Dict) -> Path:
        """Generate comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"pnbtr_validation_report_{timestamp}.html"
        
        html_content = self._create_html_report(validation_results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save JSON version
        json_path = self.output_dir / f"pnbtr_validation_data_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        return report_path
    
    # Helper methods for test signal generation and analysis
    
    def _generate_test_signal(self, signal_type: str, sample_rate: int, duration: float = 5.0) -> np.ndarray:
        """Generate test signals for validation."""
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        if signal_type == "sine":
            return 0.5 * np.sin(2 * np.pi * 1000 * t)
        elif signal_type == "white_noise":
            return 0.1 * np.random.randn(num_samples)
        elif signal_type == "pink_noise":
            # Simple pink noise approximation
            white = np.random.randn(num_samples)
            # Apply simple filtering for pink noise characteristic
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
            return 0.1 * signal.lfilter(b, a, white)
        elif signal_type == "chirp":
            return 0.5 * signal.chirp(t, 20, duration, 20000, method='logarithmic')
        elif signal_type == "music":
            # Generate synthetic "music-like" signal with multiple harmonics
            fundamental = 440  # A4
            harmonics = [1, 2, 3, 4, 5]
            amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]
            music_signal = np.zeros(num_samples)
            for harmonic, amplitude in zip(harmonics, amplitudes):
                music_signal += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
            return 0.3 * music_signal
        else:
            # Default to sine wave
            return 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    def _generate_frequency_sweep(self, sample_rate: int, duration: float = 10.0) -> np.ndarray:
        """Generate logarithmic frequency sweep."""
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        f_start = 20
        f_end = min(20000, sample_rate // 2 - 1000)  # Stay below Nyquist
        
        return 0.5 * signal.chirp(t, f_start, duration, f_end, method='logarithmic')
    
    def _analyze_frequency_response(self, input_signal: np.ndarray, 
                                   output_signal: np.ndarray, 
                                   sample_rate: int) -> List[Tuple[float, float]]:
        """Analyze frequency response from input/output signals."""
        # Simple FFT-based frequency response analysis
        input_fft = np.fft.fft(input_signal)
        output_fft = np.fft.fft(output_signal)
        
        freqs = np.fft.fftfreq(len(input_signal), 1/sample_rate)
        
        # Calculate magnitude response
        magnitude_response = np.abs(output_fft) / (np.abs(input_fft) + 1e-10)
        
        # Convert to dB and extract positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        response_db = 20 * np.log10(magnitude_response[:len(magnitude_response)//2] + 1e-10)
        
        # Focus on audible range
        audible_mask = (positive_freqs >= 20) & (positive_freqs <= 20000)
        audible_freqs = positive_freqs[audible_mask]
        audible_response = response_db[audible_mask]
        
        return list(zip(audible_freqs, audible_response))
    
    def _analyze_phase_linearity(self, input_signal: np.ndarray,
                                output_signal: np.ndarray,
                                sample_rate: int) -> Dict:
        """Analyze phase linearity."""
        input_fft = np.fft.fft(input_signal)
        output_fft = np.fft.fft(output_signal)
        
        # Calculate phase response
        phase_response = np.angle(output_fft) - np.angle(input_fft)
        
        # Unwrap phase
        phase_response = np.unwrap(phase_response)
        
        freqs = np.fft.fftfreq(len(input_signal), 1/sample_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_phase = phase_response[:len(phase_response)//2]
        
        # Focus on audible range
        audible_mask = (positive_freqs >= 20) & (positive_freqs <= 20000)
        audible_freqs = positive_freqs[audible_mask]
        audible_phase = positive_phase[audible_mask]
        
        # Calculate linearity (R-squared of linear fit)
        if len(audible_freqs) > 2:
            coeffs = np.polyfit(audible_freqs, audible_phase, 1)
            fitted_phase = np.polyval(coeffs, audible_freqs)
            
            ss_res = np.sum((audible_phase - fitted_phase) ** 2)
            ss_tot = np.sum((audible_phase - np.mean(audible_phase)) ** 2)
            
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            linearity_score = max(0.0, r_squared)
        else:
            linearity_score = 0.0
        
        return {
            "linearity_score": linearity_score,
            "phase_response": list(zip(audible_freqs, audible_phase))
        }
    
    def _generate_synthetic_video_data(self) -> Dict:
        """Generate synthetic video test data."""
        test_data = {}
        
        # Generate simple test patterns
        formats = {
            "480p": (480, 640, 3),
            "720p": (720, 1280, 3),
            "test_pattern": (240, 320, 3)
        }
        
        for format_name, (height, width, channels) in formats.items():
            # Generate a few test frames
            frames = []
            for i in range(10):  # 10 test frames
                # Create gradient pattern
                frame = np.zeros((height, width, channels), dtype=np.uint8)
                
                # Add gradient patterns
                y_grad = np.linspace(0, 255, height).reshape(-1, 1, 1)
                x_grad = np.linspace(0, 255, width).reshape(1, -1, 1)
                
                frame[:, :, 0] = y_grad.astype(np.uint8)  # Red gradient
                frame[:, :, 1] = x_grad.astype(np.uint8)  # Green gradient
                frame[:, :, 2] = ((y_grad + x_grad) / 2).astype(np.uint8)  # Blue diagonal
                
                # Add some temporal variation
                frame = np.roll(frame, i * 10, axis=1)
                
                frames.append(frame)
            
            test_data[format_name] = frames
        
        return test_data
    
    def _frames_to_stream(self, frames: List[np.ndarray]) -> np.ndarray:
        """Convert video frames to audio-like stream (simulate Jvid)."""
        # Flatten all frames and treat as 1D signal
        stream_data = []
        for frame in frames:
            stream_data.extend(frame.flatten())
        
        # Normalize to audio range [-1, 1]
        stream_array = np.array(stream_data, dtype=np.float32)
        stream_array = (stream_array / 255.0) * 2.0 - 1.0
        
        return stream_array
    
    def _stream_to_frames(self, stream: np.ndarray, frame_shape: Tuple) -> List[np.ndarray]:
        """Convert stream back to video frames."""
        # Denormalize from audio range
        stream_denorm = ((stream + 1.0) / 2.0) * 255.0
        stream_denorm = np.clip(stream_denorm, 0, 255).astype(np.uint8)
        
        # Reshape back to frames
        frame_size = np.prod(frame_shape)
        num_frames = len(stream_denorm) // frame_size
        
        frames = []
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            
            if end_idx <= len(stream_denorm):
                frame_data = stream_denorm[start_idx:end_idx]
                frame = frame_data.reshape(frame_shape)
                frames.append(frame)
        
        return frames
    
    def _create_html_report(self, validation_results: Dict) -> str:
        """Create HTML validation report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PNBTR Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
        .metric-good {{ color: green; font-weight: bold; }}
        .metric-warning {{ color: orange; font-weight: bold; }}
        .metric-bad {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PNBTR Comprehensive Validation Report</h1>
        <p><strong>Generated:</strong> {validation_results.get('timestamp', 'N/A')}</p>
        <p><strong>Overall Score:</strong> 
           <span class="{'metric-good' if validation_results.get('overall_score', 0) >= 0.8 else 'metric-warning' if validation_results.get('overall_score', 0) >= 0.6 else 'metric-bad'}">
           {validation_results.get('overall_score', 0):.1%}
           </span>
        </p>
    </div>
    
    <div class="section">
        <h2>Hi-Fi Compliance Summary</h2>
        <table>
            <tr><th>Criteria</th><th>Status</th></tr>
            <tr><td>Audio Quality</td><td class="{'metric-good' if validation_results.get('hifi_compliance', {}).get('audio_quality_compliant', False) else 'metric-bad'}">
                {'✓ PASS' if validation_results.get('hifi_compliance', {}).get('audio_quality_compliant', False) else '✗ FAIL'}
            </td></tr>
            <tr><td>Real-time Performance</td><td class="{'metric-good' if validation_results.get('hifi_compliance', {}).get('realtime_compliant', False) else 'metric-bad'}">
                {'✓ PASS' if validation_results.get('hifi_compliance', {}).get('realtime_compliant', False) else '✗ FAIL'}
            </td></tr>
            <tr><td>Frequency Response</td><td class="{'metric-good' if validation_results.get('hifi_compliance', {}).get('frequency_response_compliant', False) else 'metric-bad'}">
                {'✓ PASS' if validation_results.get('hifi_compliance', {}).get('frequency_response_compliant', False) else '✗ FAIL'}
            </td></tr>
            <tr><td><strong>Overall Compliance</strong></td><td class="{'metric-good' if validation_results.get('hifi_compliance', {}).get('overall_compliant', False) else 'metric-bad'}">
                {'✓ MEETS HI-FI STANDARDS' if validation_results.get('hifi_compliance', {}).get('overall_compliant', False) else '✗ DOES NOT MEET HI-FI STANDARDS'}
            </td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Audio Quality Results</h2>
        {self._format_audio_results_html(validation_results.get('audio_quality', {}))}
    </div>
    
    <div class="section">
        <h2>Real-time Performance Results</h2>
        {self._format_realtime_results_html(validation_results.get('realtime_performance', {}))}
    </div>
    
    <div class="section">
        <h2>Frequency Response Results</h2>
        {self._format_frequency_results_html(validation_results.get('frequency_response', {}))}
    </div>
    
    <div class="section">
        <h2>Multi-modal Handling Results</h2>
        {self._format_multimodal_results_html(validation_results.get('multimodal_handling', {}))}
    </div>
    
</body>
</html>
        """
        return html
    
    def _format_audio_results_html(self, audio_results: Dict) -> str:
        """Format audio results for HTML report."""
        if not audio_results:
            return "<p>No audio test results available.</p>"
        
        summary = audio_results.get('summary', {})
        
        html = f"""
        <h3>Summary</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Tests Passed</td><td>{summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}</td></tr>
            <tr><td>Average SNR</td><td>{summary.get('average_snr', 0):.1f} dB</td></tr>
            <tr><td>Average THD+N</td><td>{summary.get('average_thd_n', 0):.3f}%</td></tr>
            <tr><td>Average Coloration</td><td>{summary.get('average_coloration', 0):.3f}%</td></tr>
            <tr><td>Meets Hi-Fi Standards</td><td>{'Yes' if summary.get('meets_hifi_standards', False) else 'No'}</td></tr>
        </table>
        """
        
        return html
    
    def _format_realtime_results_html(self, rt_results: Dict) -> str:
        """Format real-time results for HTML report."""
        if not rt_results:
            return "<p>No real-time test results available.</p>"
        
        summary = rt_results.get('summary', {})
        
        html = f"""
        <h3>Summary</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Tests Passed</td><td>{summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}</td></tr>
            <tr><td>Average Latency</td><td>{summary.get('average_latency_ms', 0):.2f} ms</td></tr>
            <tr><td>Maximum Latency</td><td>{summary.get('max_latency_ms', 0):.2f} ms</td></tr>
            <tr><td>Average CPU Load</td><td>{summary.get('average_cpu_load', 0):.1f}%</td></tr>
            <tr><td>Meets Real-time Requirements</td><td>{'Yes' if summary.get('meets_realtime_requirements', False) else 'No'}</td></tr>
        </table>
        """
        
        return html
    
    def _format_frequency_results_html(self, freq_results: Dict) -> str:
        """Format frequency response results for HTML report."""
        if not freq_results:
            return "<p>No frequency response test results available.</p>"
        
        summary = freq_results.get('summary', {})
        
        html = f"""
        <h3>Summary</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Maximum Deviation</td><td>{summary.get('max_deviation_db', 0):.2f} dB</td></tr>
            <tr><td>Average Phase Linearity</td><td>{summary.get('avg_phase_linearity', 0):.1%}</td></tr>
            <tr><td>Meets Flatness Requirement</td><td>{'Yes' if summary.get('meets_flatness_requirement', False) else 'No'}</td></tr>
        </table>
        """
        
        return html
    
    def _format_multimodal_results_html(self, mm_results: Dict) -> str:
        """Format multi-modal results for HTML report."""
        if not mm_results:
            return "<p>No multi-modal test results available.</p>"
        
        summary = mm_results.get('summary', {})
        
        html = f"""
        <h3>Summary</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Tests Passed</td><td>{summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}</td></tr>
            <tr><td>Average PSNR</td><td>{summary.get('average_psnr', 0):.1f} dB</td></tr>
            <tr><td>Average SSIM</td><td>{summary.get('average_ssim', 0):.3f}</td></tr>
            <tr><td>Meets Quality Thresholds</td><td>{'Yes' if summary.get('meets_quality_thresholds', False) else 'No'}</td></tr>
        </table>
        """
        
        return html
