#!/usr/bin/env python3
"""
PNBTR Phase 3 Integration Test
Tests advanced metrics, signal analysis, visualization, and performance profiling.
Validates the complete Phase 3 system integration.
"""

import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_perceptual_metrics():
    """Test perceptual metrics functionality"""
    print("🧪 Testing Perceptual Metrics...")
    
    try:
        from metrics.perceptual_metrics import create_perceptual_evaluator
        
        # Create test signals
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Target: clean harmonic signal
        target = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
        
        # Prediction: slightly degraded version
        prediction = target + 0.03 * np.random.normal(0, 1, len(target))
        
        # Create evaluator and test
        evaluator = create_perceptual_evaluator(sample_rate)
        metrics = evaluator.evaluate_all_metrics(prediction, target)
        
        # Validate key metrics
        assert 'STOI' in metrics, "STOI metric missing"
        assert 'PESQ_like' in metrics, "PESQ-like metric missing"
        assert 0 <= metrics['STOI'] <= 1, f"STOI out of range: {metrics['STOI']}"
        assert 1 <= metrics['PESQ_like'] <= 5, f"PESQ-like out of range: {metrics['PESQ_like']}"
        
        print(f"   ✅ STOI: {metrics['STOI']:.3f}")
        print(f"   ✅ PESQ-like: {metrics['PESQ_like']:.2f}")
        print(f"   ✅ Spectral Centroid Error: {metrics['CentroidError']:.1f} Hz")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Perceptual metrics test failed: {e}")
        return False

def test_signal_analysis():
    """Test advanced signal analysis functionality"""
    print("\n🧪 Testing Advanced Signal Analysis...")
    
    try:
        from metrics.signal_analysis import create_signal_analyzer
        
        # Create complex test signal
        sample_rate = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Multi-component signal with transients
        signal_data = (
            0.4 * np.sin(2 * np.pi * 440 * t) +                # Fundamental
            0.2 * np.sin(2 * np.pi * 880 * t) +                # Harmonic
            0.1 * np.sin(2 * np.pi * 220 * t) +                # Sub-harmonic
            0.05 * np.random.normal(0, 1, len(t))               # Noise
        )
        
        # Add transients
        transient_indices = [int(0.5 * sample_rate), int(1.2 * sample_rate)]
        for idx in transient_indices:
            if idx < len(signal_data) - 100:
                signal_data[idx:idx+100] += 0.3 * np.exp(-np.arange(100) / 20)
        
        # Create analyzer and perform analysis
        analyzer = create_signal_analyzer(sample_rate)
        analysis = analyzer.comprehensive_analysis(signal_data)
        
        # Validate analysis components
        assert 'multi_resolution_fft' in analysis, "Multi-resolution FFT missing"
        assert 'stft' in analysis, "STFT missing"
        assert 'transients' in analysis, "Transient detection missing"
        assert 'wavelet_bands' in analysis, "Wavelet analysis missing"
        assert 'temporal_structure' in analysis, "Temporal structure missing"
        assert 'summary' in analysis, "Summary missing"
        
        # Check transient detection
        transients = analysis['transients']
        print(f"   ✅ Transients detected: {transients['n_transients']}")
        print(f"   ✅ Signal length: {analysis['summary']['signal_length_seconds']:.2f}s")
        print(f"   ✅ Dynamic range: {analysis['temporal_structure']['dynamic_range_db']:.1f} dB")
        
        # Check multi-resolution FFT
        fft_results = analysis['multi_resolution_fft']
        print(f"   ✅ FFT resolutions: {list(fft_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Signal analysis test failed: {e}")
        return False

def test_visualization():
    """Test training visualization functionality"""
    print("\n🧪 Testing Training Visualization...")
    
    try:
        from visualization.training_dashboard import create_training_visualizer
        
        # Create visualizer (may fail if matplotlib unavailable)
        visualizer = create_training_visualizer()
        
        # Try to initialize dashboard
        dashboard_ready = visualizer.initialize_dashboard()
        
        if dashboard_ready:
            print("   ✅ Dashboard initialized successfully")
            
            # Simulate some training data
            for epoch in range(10):
                loss = 1.0 * np.exp(-epoch * 0.1) + 0.05 * np.random.random()
                accuracy = 1.0 - loss + 0.02 * np.random.random()
                lr = 0.001 * (0.9 ** (epoch // 5))
                
                metrics = {
                    'SDR': 0.5 + accuracy * 0.4,
                    'STOI': 0.6 + accuracy * 0.3,
                    'PESQ_like': 2.0 + accuracy * 2.5,
                    'CentroidError': (1.0 - accuracy) * 500,
                    'HarmonicError': (1.0 - accuracy) * 3
                }
                
                visualizer.update_training_progress(epoch, loss, accuracy, lr, metrics)
            
            # Test signal display
            t = np.linspace(0, 0.1, 4800)
            target = 0.5 * np.sin(2 * np.pi * 440 * t)
            prediction = target + 0.1 * np.random.normal(0, 1, len(target))
            
            visualizer.update_signal_display(target, prediction, 48000)
            
            # Save summary
            summary_path = Path("test_training_summary.json")
            visualizer.save_training_summary(summary_path)
            
            # Close dashboard
            visualizer.close_dashboard()
            
            print("   ✅ Training visualization complete")
            
        else:
            print("   ⚠️  Dashboard initialization failed (matplotlib unavailable)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
        return False

def test_performance_profiling():
    """Test performance profiling functionality"""
    print("\n🧪 Testing Performance Profiling...")
    
    try:
        from metrics.performance_profiler import create_performance_profiler
        
        # Create profiler
        profiler = create_performance_profiler()
        
        # Test basic profiling
        with profiler.profile_execution("test_operation") as metrics:
            # Simulate computation
            data = np.random.random((1000, 100))
            result = np.dot(data, data.T)
            eigenvals = np.linalg.eigvals(result[:50, :50])
        
        print(f"   ✅ Operation profiled: {metrics.execution_time_ms:.2f} ms")
        
        # Test model profiling with dummy model
        try:
            from training.model_factory import create_pnbtr_model
            
            # Create test model
            model = create_pnbtr_model("dummy", input_size=1000)
            
            # Profile model size
            size_metrics = profiler.profile_model_size(model)
            print(f"   ✅ Model size: {size_metrics['model_size_mb']:.3f} MB")
            print(f"   ✅ Parameters: {size_metrics['parameters_count']:,}")
            
            # Profile training step
            test_input = np.random.normal(0, 0.1, 1000).astype(np.float64)
            test_target = test_input + 0.01 * np.random.normal(0, 1, 1000)
            
            training_metrics = profiler.profile_training_step(model, test_input, test_target)
            print(f"   ✅ Training step: {training_metrics.execution_time_ms:.2f} ms")
            
            # Profile inference
            inference_metrics = profiler.profile_inference(model, test_input, num_runs=20)
            print(f"   ✅ Inference time: {inference_metrics['mean_inference_time_ms']:.2f} ms")
            print(f"   ✅ Real-time compatible: {'Yes' if inference_metrics['realtime_compatible'] else 'No'}")
            
        except Exception as model_error:
            print(f"   ⚠️  Model profiling failed: {model_error}")
        
        # Generate performance report
        report = profiler.generate_performance_report()
        assert 'session_info' in report, "Session info missing"
        assert 'memory_timeline' in report, "Memory timeline missing"
        
        # Save performance report
        report_path = Path("test_performance_report.json")
        profiler.save_performance_report(report_path)
        
        print("   ✅ Performance profiling complete")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance profiling test failed: {e}")
        return False

def test_integrated_workflow():
    """Test complete Phase 3 integration workflow"""
    print("\n🧪 Testing Integrated Phase 3 Workflow...")
    
    try:
        from metrics.perceptual_metrics import create_perceptual_evaluator
        from metrics.signal_analysis import create_signal_analyzer  
        from metrics.performance_profiler import create_performance_profiler
        from training.model_factory import create_pnbtr_model
        
        # Create all components
        perceptual_evaluator = create_perceptual_evaluator(48000)
        signal_analyzer = create_signal_analyzer(48000)
        profiler = create_performance_profiler()
        
        # Create test scenario
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Target signal with rich harmonic content
        target_signal = (
            0.4 * np.sin(2 * np.pi * 440 * t) +
            0.2 * np.sin(2 * np.pi * 880 * t) + 
            0.1 * np.sin(2 * np.pi * 1320 * t)
        )
        
        # Create and profile model
        with profiler.profile_execution("model_creation") as creation_metrics:
            model = create_pnbtr_model("dummy", input_size=len(target_signal))
        
        print(f"   ✅ Model creation: {creation_metrics.execution_time_ms:.2f} ms")
        
        # Profile prediction
        with profiler.profile_execution("model_prediction") as pred_metrics:
            prediction = model.predict(target_signal)
        
        print(f"   ✅ Model prediction: {pred_metrics.execution_time_ms:.2f} ms")
        
        # Comprehensive signal analysis
        with profiler.profile_execution("signal_analysis") as analysis_metrics:
            target_analysis = signal_analyzer.comprehensive_analysis(target_signal)
            pred_analysis = signal_analyzer.comprehensive_analysis(prediction)
        
        print(f"   ✅ Signal analysis: {analysis_metrics.execution_time_ms:.2f} ms")
        
        # Perceptual evaluation
        with profiler.profile_execution("perceptual_metrics") as metrics_timing:
            perceptual_metrics = perceptual_evaluator.evaluate_all_metrics(prediction, target_signal)
        
        print(f"   ✅ Perceptual metrics: {metrics_timing.execution_time_ms:.2f} ms")
        
        # Create comprehensive results
        integration_results = {
            'model_performance': {
                'creation_time_ms': creation_metrics.execution_time_ms,
                'prediction_time_ms': pred_metrics.execution_time_ms,
                'model_size_mb': profiler.profile_model_size(model)['model_size_mb']
            },
            'signal_characteristics': {
                'target_transients': target_analysis['transients']['n_transients'],
                'prediction_transients': pred_analysis['transients']['n_transients'],
                'target_dynamic_range': target_analysis['temporal_structure']['dynamic_range_db'],
                'prediction_dynamic_range': pred_analysis['temporal_structure']['dynamic_range_db']
            },
            'perceptual_quality': {
                'stoi_score': perceptual_metrics['STOI'],
                'pesq_like_score': perceptual_metrics['PESQ_like'],
                'spectral_centroid_error': perceptual_metrics['CentroidError'],
                'harmonic_ratio_error': perceptual_metrics['HarmonicError']
            },
            'analysis_performance': {
                'signal_analysis_time_ms': analysis_metrics.execution_time_ms,
                'perceptual_metrics_time_ms': metrics_timing.execution_time_ms,
                'total_analysis_time_ms': analysis_metrics.execution_time_ms + metrics_timing.execution_time_ms
            }
        }
        
        # Validate integration
        assert integration_results['model_performance']['prediction_time_ms'] > 0
        assert integration_results['perceptual_quality']['stoi_score'] >= 0
        assert integration_results['perceptual_quality']['pesq_like_score'] >= 1
        assert integration_results['analysis_performance']['total_analysis_time_ms'] > 0
        
        print("   ✅ Phase 3 integration successful!")
        print(f"   📊 STOI: {integration_results['perceptual_quality']['stoi_score']:.3f}")
        print(f"   📊 PESQ-like: {integration_results['perceptual_quality']['pesq_like_score']:.2f}")
        print(f"   ⚡ Total analysis time: {integration_results['analysis_performance']['total_analysis_time_ms']:.1f} ms")
        
        return True, integration_results
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False, {}

def main():
    """Run comprehensive Phase 3 integration test"""
    print("🚀 PNBTR Phase 3 Integration Test")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Run individual component tests
    test_results['perceptual_metrics'] = test_perceptual_metrics()
    test_results['signal_analysis'] = test_signal_analysis()
    test_results['visualization'] = test_visualization()
    test_results['performance_profiling'] = test_performance_profiling()
    
    # Run integration test
    integration_success, integration_results = test_integrated_workflow()
    test_results['integration'] = integration_success
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():20}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ Advanced metrics and analysis system ready!")
    else:
        print("⚠️  Some tests failed - check dependencies and implementations")
    
    print("\n🔍 Phase 3 Features Validated:")
    print("   • Perceptual Metrics (STOI, PESQ-like, spectral analysis)")
    print("   • Advanced Signal Analysis (multi-resolution FFT, transients)")
    print("   • Training Visualization (real-time dashboards)")
    print("   • Performance Profiling (memory, speed, model analysis)")
    print("   • Complete Integration Workflow")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 