#!/usr/bin/env python3
"""
PNBTR Phase 3 Comprehensive Demo
Showcases advanced metrics, signal analysis, visualization, and performance profiling.
A complete demonstration of Phase 3 capabilities for production-ready training.
"""

import numpy as np
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def create_rich_test_signal(sample_rate: int = 48000, duration: float = 3.0) -> np.ndarray:
    """Create a complex, realistic test signal for demonstration"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Multi-tonal harmonic content (simulating instruments)
    fundamental = 440  # A4
    signal = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +           # Fundamental
        0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +       # 2nd harmonic
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +       # 3rd harmonic
        0.05 * np.sin(2 * np.pi * fundamental * 4 * t) +      # 4th harmonic
        0.15 * np.sin(2 * np.pi * 220 * t) +                  # Sub-harmonic (A3)
        0.1 * np.sin(2 * np.pi * 880 * t)                     # Octave (A5)
    )
    
    # Add amplitude modulation (tremolo effect)
    modulation_freq = 5  # 5 Hz tremolo
    tremolo = 1 + 0.3 * np.sin(2 * np.pi * modulation_freq * t)
    signal *= tremolo
    
    # Add frequency modulation (vibrato effect)
    vibrato_freq = 6  # 6 Hz vibrato
    vibrato_depth = 10  # Hz depth
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    signal += 0.2 * np.sin(2 * np.pi * (fundamental + vibrato) * t)
    
    # Add percussive transients
    transient_times = [0.5, 1.2, 2.1, 2.8]
    for t_trans in transient_times:
        start_idx = int(t_trans * sample_rate)
        if start_idx < len(signal) - 200:
            # Create realistic attack/decay envelope
            attack_samples = 50
            decay_samples = 150
            
            # Attack phase
            attack_env = np.linspace(0, 1, attack_samples)
            # Decay phase  
            decay_env = np.exp(-np.arange(decay_samples) / 30)
            
            # Combine
            transient_env = np.concatenate([attack_env, decay_env])
            transient_signal = 0.4 * np.sin(2 * np.pi * 1760 * np.linspace(0, len(transient_env)/sample_rate, len(transient_env)))
            
            # Apply to signal
            end_idx = start_idx + len(transient_env)
            if end_idx <= len(signal):
                signal[start_idx:end_idx] += transient_signal * transient_env
    
    # Add realistic noise floor
    noise_floor = 0.02 * np.random.normal(0, 1, len(signal))
    signal += noise_floor
    
    # Apply realistic dynamics (soft limiting)
    signal = np.tanh(signal * 0.8) * 0.9
    
    return signal

def demonstrate_perceptual_metrics():
    """Demonstrate advanced perceptual metrics capabilities"""
    print("🎵 PERCEPTUAL METRICS DEMONSTRATION")
    print("=" * 50)
    
    from metrics.perceptual_metrics import create_perceptual_evaluator
    
    # Create high-quality test signals
    sample_rate = 48000
    target_signal = create_rich_test_signal(sample_rate, 3.0)
    
    # Create different quality predictions to show metric sensitivity
    test_scenarios = {
        "Excellent": target_signal + 0.005 * np.random.normal(0, 1, len(target_signal)),
        "Good": target_signal + 0.02 * np.random.normal(0, 1, len(target_signal)),
        "Fair": target_signal + 0.05 * np.random.normal(0, 1, len(target_signal)),
        "Poor": target_signal + 0.1 * np.random.normal(0, 1, len(target_signal)),
        "Low-pass": target_signal * 0.8 + 0.02 * np.random.normal(0, 1, len(target_signal))
    }
    
    # Apply low-pass filtering to the last scenario
    from scipy import signal as sp_signal
    try:
        # Simple low-pass filter (removes high frequencies)
        b, a = sp_signal.butter(4, 0.3, btype='low')
        test_scenarios["Low-pass"] = sp_signal.filtfilt(b, a, test_scenarios["Low-pass"])
    except:
        pass  # Skip filtering if scipy unavailable
    
    evaluator = create_perceptual_evaluator(sample_rate)
    
    print(f"📊 Evaluating {len(test_scenarios)} quality scenarios...")
    print()
    
    results_summary = []
    for scenario_name, prediction in test_scenarios.items():
        print(f"🔍 Analyzing: {scenario_name}")
        
        # Compute all metrics
        metrics = evaluator.evaluate_all_metrics(prediction, target_signal)
        
        print(f"   STOI Score:      {metrics['STOI']:.3f}")
        print(f"   PESQ-like:       {metrics['PESQ_like']:.2f}")
        print(f"   Harmonic Error:  {metrics['HarmonicError']:.2f} dB")
        print(f"   Centroid Error:  {metrics['CentroidError']:.0f} Hz")
        print(f"   Flatness Error:  {metrics['FlatnessError']:.3f}")
        print()
        
        results_summary.append({
            'scenario': scenario_name,
            'stoi': metrics['STOI'],
            'pesq': metrics['PESQ_like'],
            'overall_quality': (metrics['STOI'] + (metrics['PESQ_like'] - 1) / 4) / 2
        })
    
    # Show quality ranking
    results_summary.sort(key=lambda x: x['overall_quality'], reverse=True)
    print("🏆 Quality Ranking (Best to Worst):")
    for i, result in enumerate(results_summary, 1):
        print(f"   {i}. {result['scenario']}: {result['overall_quality']:.3f} overall")
    
    return results_summary

def demonstrate_signal_analysis():
    """Demonstrate advanced signal analysis capabilities"""
    print("\n🔬 ADVANCED SIGNAL ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    from metrics.signal_analysis import create_signal_analyzer
    
    # Create test signal with known characteristics
    sample_rate = 48000
    test_signal = create_rich_test_signal(sample_rate, 3.0)
    
    analyzer = create_signal_analyzer(sample_rate)
    
    print("📈 Performing comprehensive signal analysis...")
    
    # Time the analysis
    start_time = time.time()
    analysis = analyzer.comprehensive_analysis(test_signal)
    analysis_time = time.time() - start_time
    
    print(f"⚡ Analysis completed in {analysis_time:.2f} seconds")
    print()
    
    # Display key findings
    summary = analysis['summary']
    temporal = analysis['temporal_structure']
    transients = analysis['transients']
    
    print("📊 Signal Characteristics:")
    print(f"   Duration:        {summary['signal_length_seconds']:.2f} seconds")
    print(f"   RMS Level:       {summary['rms_level']:.4f}")
    print(f"   Peak Level:      {summary['peak_level']:.4f}")
    print(f"   Crest Factor:    {summary['crest_factor']:.2f}")
    print(f"   Dynamic Range:   {temporal['dynamic_range_db']:.1f} dB")
    print()
    
    print("🎵 Temporal Structure:")
    print(f"   Transients:      {transients['n_transients']} detected")
    print(f"   Onset Density:   {temporal['onset_density']:.2f} onsets/sec")
    print(f"   Zero Crossings:  {temporal['zero_crossing_rate']:.1f} Hz")
    print()
    
    # Multi-resolution FFT analysis
    fft_results = analysis['multi_resolution_fft']
    print("🔍 Multi-Resolution Spectral Analysis:")
    for fft_key, fft_data in fft_results.items():
        resolution = fft_data['resolution_hz']
        peak_freq_idx = np.argmax(fft_data['magnitude'])
        peak_freq = fft_data['frequencies'][peak_freq_idx]
        print(f"   {fft_key}: {resolution:.1f} Hz resolution, peak at {peak_freq:.1f} Hz")
    print()
    
    # Wavelet-like analysis
    wavelet_bands = analysis['wavelet_bands']
    print("🌊 Frequency Band Energy Distribution:")
    total_energy = sum(band['total_energy'] for band in wavelet_bands.values())
    
    for band_key, band_data in list(wavelet_bands.items())[:8]:  # Show first 8 bands
        freq_range = band_data['frequency_range']
        energy_percent = (band_data['total_energy'] / total_energy) * 100
        print(f"   {freq_range[0]:4.0f}-{freq_range[1]:4.0f} Hz: {energy_percent:5.1f}% energy")
    
    if transients['n_transients'] > 0:
        print(f"\n⚡ Transient Analysis:")
        transient_times = transients['transient_indices'] / sample_rate
        print(f"   Detected at: {', '.join(f'{t:.2f}s' for t in transient_times)}")
    
    return analysis

def demonstrate_performance_profiling():
    """Demonstrate performance profiling and optimization analysis"""
    print("\n⚡ PERFORMANCE PROFILING DEMONSTRATION")
    print("=" * 50)
    
    from metrics.performance_profiler import create_performance_profiler
    from training.model_factory import create_pnbtr_model
    
    profiler = create_performance_profiler()
    
    # Test different model architectures
    architectures_to_test = ["dummy", "mlp"]
    input_size = 2048
    
    print(f"🏗️  Benchmarking {len(architectures_to_test)} model architectures...")
    print(f"📊 Input size: {input_size} samples")
    print()
    
    benchmark_results = profiler.benchmark_model_architectures(
        architectures_to_test, 
        input_size=input_size,
        config={'hidden_dims': [128, 64, 32]}
    )
    
    # Display results in a formatted table
    print("📈 PERFORMANCE BENCHMARK RESULTS")
    print("-" * 70)
    print(f"{'Architecture':<12} {'Size(MB)':<10} {'Params':<10} {'Train(ms)':<12} {'Infer(ms)':<12} {'RT Score':<10}")
    print("-" * 70)
    
    for arch, results in benchmark_results.items():
        if 'error' not in results:
            size_mb = results['model_size']['model_size_mb']
            params = results['model_size']['parameters_count']
            train_time = results['training_performance']['time_per_step_ms']
            infer_time = results['inference_performance']['mean_inference_time_ms']
            rt_score = results['realtime_score']
            
            print(f"{arch:<12} {size_mb:<10.3f} {params:<10,} {train_time:<12.2f} {infer_time:<12.2f} {rt_score:<10.3f}")
        else:
            print(f"{arch:<12} {'ERROR':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("-" * 70)
    print()
    
    # Real-time compatibility analysis
    print("🎯 Real-Time Compatibility Analysis:")
    realtime_threshold = profiler.realtime_threshold_ms
    memory_limit = profiler.memory_limit_mb
    
    print(f"   Latency threshold: ≤ {realtime_threshold} ms")
    print(f"   Memory limit:      ≤ {memory_limit} MB")
    print()
    
    compatible_archs = []
    for arch, results in benchmark_results.items():
        if 'error' not in results:
            infer_time = results['inference_performance']['mean_inference_time_ms']
            memory_usage = results['model_size']['memory_footprint_mb']
            is_compatible = infer_time <= realtime_threshold and memory_usage <= memory_limit
            
            status = "✅ COMPATIBLE" if is_compatible else "❌ NOT COMPATIBLE"
            print(f"   {arch}: {status}")
            
            if is_compatible:
                compatible_archs.append(arch)
    
    if compatible_archs:
        print(f"\n🏆 {len(compatible_archs)} architecture(s) meet real-time requirements!")
    else:
        print(f"\n⚠️  No architectures meet strict real-time requirements")
        print("    Consider model optimization or relaxing constraints")
    
    # Memory usage analysis
    print(f"\n💾 Memory Usage Analysis:")
    report = profiler.generate_performance_report()
    session_info = report['session_info']
    
    print(f"   Baseline memory:   {session_info['baseline_memory_mb']:.1f} MB")
    print(f"   Current memory:    {session_info['current_memory_mb']:.1f} MB")
    print(f"   Memory increase:   {session_info['memory_increase_mb']:.1f} MB")
    print(f"   Available tools:   {'psutil' if session_info['psutil_available'] else 'basic'}")
    
    return benchmark_results

def demonstrate_integrated_workflow():
    """Demonstrate complete Phase 3 integrated workflow"""
    print("\n🔄 INTEGRATED WORKFLOW DEMONSTRATION")
    print("=" * 50)
    
    from metrics.perceptual_metrics import create_perceptual_evaluator
    from metrics.signal_analysis import create_signal_analyzer
    from metrics.performance_profiler import create_performance_profiler
    from training.model_factory import create_pnbtr_model
    
    print("🚀 Initializing complete Phase 3 analysis pipeline...")
    
    # Initialize all components
    perceptual_evaluator = create_perceptual_evaluator(48000)
    signal_analyzer = create_signal_analyzer(48000)
    profiler = create_performance_profiler()
    
    # Create realistic scenario
    print("🎵 Creating realistic audio processing scenario...")
    target_signal = create_rich_test_signal(48000, 2.0)
    
    # Profile the complete workflow
    workflow_timings = {}
    
    # 1. Model creation and setup
    with profiler.profile_execution("model_setup") as setup_metrics:
        model = create_pnbtr_model("mlp", input_size=len(target_signal), 
                                 config={'hidden_dims': [256, 128, 64]})
    workflow_timings['model_setup'] = setup_metrics.execution_time_ms
    
    # 2. Signal processing (prediction)
    with profiler.profile_execution("signal_processing") as process_metrics:
        prediction = model.predict(target_signal)
    workflow_timings['signal_processing'] = process_metrics.execution_time_ms
    
    # 3. Comprehensive signal analysis
    with profiler.profile_execution("signal_analysis") as analysis_metrics:
        target_analysis = signal_analyzer.comprehensive_analysis(target_signal)
        pred_analysis = signal_analyzer.comprehensive_analysis(prediction)
    workflow_timings['signal_analysis'] = analysis_metrics.execution_time_ms
    
    # 4. Perceptual quality evaluation
    with profiler.profile_execution("perceptual_evaluation") as eval_metrics:
        quality_metrics = perceptual_evaluator.evaluate_all_metrics(prediction, target_signal)
    workflow_timings['perceptual_evaluation'] = eval_metrics.execution_time_ms
    
    # 5. Performance analysis
    with profiler.profile_execution("performance_analysis") as perf_metrics:
        model_performance = profiler.profile_model_size(model)
        inference_performance = profiler.profile_inference(model, target_signal[:1024], num_runs=20)
    workflow_timings['performance_analysis'] = perf_metrics.execution_time_ms
    
    # Calculate total workflow time
    total_workflow_time = sum(workflow_timings.values())
    
    print(f"⚡ Complete workflow executed in {total_workflow_time:.1f} ms")
    print()
    
    # Present comprehensive results
    print("📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("-" * 50)
    
    # Workflow performance breakdown
    print("⏱️  Workflow Timing Breakdown:")
    for stage, timing in workflow_timings.items():
        percentage = (timing / total_workflow_time) * 100
        print(f"   {stage.replace('_', ' ').title():<20}: {timing:6.1f} ms ({percentage:4.1f}%)")
    print(f"   {'Total Workflow':<20}: {total_workflow_time:6.1f} ms")
    print()
    
    # Signal quality assessment
    print("🎵 Signal Quality Assessment:")
    print(f"   STOI Score:        {quality_metrics['STOI']:.3f}")
    print(f"   PESQ-like Score:   {quality_metrics['PESQ_like']:.2f}")
    print(f"   Spectral Accuracy: {1 - quality_metrics['CentroidError']/1000:.3f}")
    print(f"   Harmonic Fidelity: {max(0, 1 - quality_metrics['HarmonicError']/10):.3f}")
    print()
    
    # Model efficiency metrics
    print("🏗️  Model Efficiency:")
    print(f"   Model Size:        {model_performance['model_size_mb']:.3f} MB")
    print(f"   Parameters:        {model_performance['parameters_count']:,}")
    print(f"   Inference Speed:   {inference_performance['mean_inference_time_ms']:.2f} ms")
    print(f"   Throughput:        {inference_performance['samples_per_second']:.0f} samples/sec")
    print(f"   Real-time Ready:   {'✅ Yes' if inference_performance['realtime_compatible'] else '❌ No'}")
    print()
    
    # Signal characteristics comparison
    print("🔬 Signal Analysis Comparison:")
    target_transients = target_analysis['transients']['n_transients']
    pred_transients = pred_analysis['transients']['n_transients']
    target_dynamic = target_analysis['temporal_structure']['dynamic_range_db']
    pred_dynamic = pred_analysis['temporal_structure']['dynamic_range_db']
    
    print(f"   Transients:        Target={target_transients}, Prediction={pred_transients}")
    print(f"   Dynamic Range:     Target={target_dynamic:.1f} dB, Prediction={pred_dynamic:.1f} dB")
    print(f"   Preservation:      {min(1.0, pred_dynamic/target_dynamic if target_dynamic > 0 else 1.0):.2f}")
    
    # Final assessment
    overall_score = (
        quality_metrics['STOI'] * 0.3 +
        (quality_metrics['PESQ_like'] - 1) / 4 * 0.3 +
        (1 if inference_performance['realtime_compatible'] else 0) * 0.2 +
        min(1.0, 10 / workflow_timings['signal_processing']) * 0.2
    )
    
    print(f"\n🏆 Overall System Score: {overall_score:.3f} / 1.000")
    
    if overall_score >= 0.8:
        print("✅ EXCELLENT - Production ready!")
    elif overall_score >= 0.6:
        print("✅ GOOD - Minor optimizations recommended")
    elif overall_score >= 0.4:
        print("⚠️  FAIR - Significant improvements needed")
    else:
        print("❌ POOR - Major rework required")
    
    return {
        'workflow_timings': workflow_timings,
        'quality_metrics': quality_metrics,
        'model_performance': model_performance,
        'overall_score': overall_score
    }

def main():
    """Run the complete Phase 3 demonstration"""
    print("🎉 PNBTR PHASE 3 COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("Advanced Metrics & Analysis System")
    print("Perceptual Metrics • Signal Analysis • Performance Profiling")
    print("=" * 60)
    
    demo_results = {}
    
    try:
        # 1. Perceptual Metrics Demo
        demo_results['perceptual'] = demonstrate_perceptual_metrics()
        
        # 2. Signal Analysis Demo
        demo_results['signal_analysis'] = demonstrate_signal_analysis()
        
        # 3. Performance Profiling Demo
        demo_results['performance'] = demonstrate_performance_profiling()
        
        # 4. Integrated Workflow Demo
        demo_results['integrated'] = demonstrate_integrated_workflow()
        
        print("\n" + "=" * 60)
        print("🎊 PHASE 3 DEMONSTRATION COMPLETE")
        print("=" * 60)
        
        print("✅ All Phase 3 systems operational:")
        print("   • Perceptual Metrics: STOI, PESQ-like, spectral analysis")
        print("   • Signal Analysis: Multi-resolution FFT, transients, wavelets")
        print("   • Performance Profiling: Memory, speed, real-time validation")
        print("   • Integrated Workflow: Complete end-to-end analysis")
        
        # Final summary
        integrated_results = demo_results['integrated']
        total_time = integrated_results['workflow_timings']['signal_processing']
        overall_score = integrated_results['overall_score']
        
        print(f"\n🏆 Final Assessment:")
        print(f"   Processing Speed:  {total_time:.1f} ms")
        print(f"   Quality Score:     {overall_score:.3f} / 1.000")
        print(f"   System Status:     {'🚀 PRODUCTION READY' if overall_score >= 0.7 else '🔧 OPTIMIZATION NEEDED'}")
        
        print(f"\n🚀 Phase 3 Advanced Metrics & Analysis System Ready!")
        print("Ready to integrate with production training pipelines.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Demo completed successfully!")
    else:
        print("\n💥 Demo encountered errors.")
    
    sys.exit(0 if success else 1) 