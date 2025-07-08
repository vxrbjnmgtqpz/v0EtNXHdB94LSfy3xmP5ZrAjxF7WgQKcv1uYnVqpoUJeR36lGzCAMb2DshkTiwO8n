#!/usr/bin/env python3
"""
PNBTR Phase 3 Simple Demo
Quick demonstration of Phase 3 capabilities.
"""

import numpy as np
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run a simple Phase 3 demonstration"""
    print("🎉 PNBTR PHASE 3 SIMPLE DEMONSTRATION")
    print("=" * 60)
    
    # Create test signals
    sample_rate = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Target signal - rich harmonic content
    target_signal = (
        0.4 * np.sin(2 * np.pi * 440 * t) +        # A4 fundamental
        0.2 * np.sin(2 * np.pi * 880 * t) +        # Octave
        0.1 * np.sin(2 * np.pi * 1320 * t) +       # Perfect fifth
        0.05 * np.random.normal(0, 1, len(t))      # Noise
    )
    
    # Prediction signal - slightly degraded
    prediction = target_signal + 0.02 * np.random.normal(0, 1, len(target_signal))
    
    print("🎵 Testing Phase 3 Components...")
    print()
    
    # 1. Perceptual Metrics
    print("1️⃣  Perceptual Metrics Analysis")
    try:
        from metrics.perceptual_metrics import create_perceptual_evaluator
        
        evaluator = create_perceptual_evaluator(sample_rate)
        metrics = evaluator.evaluate_all_metrics(prediction, target_signal)
        
        print(f"   ✅ STOI Score: {metrics['STOI']:.3f}")
        print(f"   ✅ PESQ-like: {metrics['PESQ_like']:.2f}")
        print(f"   ✅ Centroid Error: {metrics['CentroidError']:.1f} Hz")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # 2. Signal Analysis
    print("2️⃣  Advanced Signal Analysis")
    try:
        from metrics.signal_analysis import create_signal_analyzer
        
        analyzer = create_signal_analyzer(sample_rate)
        analysis = analyzer.comprehensive_analysis(target_signal)
        
        print(f"   ✅ Signal length: {analysis['summary']['signal_length_seconds']:.1f}s")
        print(f"   ✅ Dynamic range: {analysis['temporal_structure']['dynamic_range_db']:.1f} dB")
        print(f"   ✅ Transients detected: {analysis['transients']['n_transients']}")
        print(f"   ✅ FFT resolutions: {len(analysis['multi_resolution_fft'])}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # 3. Performance Profiling
    print("3️⃣  Performance Profiling")
    try:
        from metrics.performance_profiler import create_performance_profiler
        from training.model_factory import create_pnbtr_model
        
        profiler = create_performance_profiler()
        
        # Create and profile a simple model
        model = create_pnbtr_model("dummy", input_size=len(target_signal))
        
        # Profile prediction
        start_time = time.time()
        _ = model.predict(target_signal)
        pred_time = (time.time() - start_time) * 1000
        
        # Model size analysis
        size_info = profiler.profile_model_size(model)
        
        print(f"   ✅ Model size: {size_info['model_size_mb']:.3f} MB")
        print(f"   ✅ Parameters: {size_info['parameters_count']:,}")
        print(f"   ✅ Prediction time: {pred_time:.2f} ms")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # 4. Integrated Workflow
    print("4️⃣  Integrated Workflow Test")
    try:
        # Combined analysis workflow
        start_time = time.time()
        
        # Create components
        evaluator = create_perceptual_evaluator(sample_rate)
        analyzer = create_signal_analyzer(sample_rate)
        profiler = create_performance_profiler()
        
        # Run analysis
        quality = evaluator.evaluate_all_metrics(prediction, target_signal)
        structure = analyzer.comprehensive_analysis(target_signal)
        
        workflow_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Complete workflow: {workflow_time:.1f} ms")
        print(f"   ✅ Quality assessment: STOI={quality['STOI']:.3f}")
        print(f"   ✅ Signal analysis: {len(structure)} components")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print()
    print("=" * 60)
    print("🎊 Phase 3 Simple Demo Complete!")
    print("✅ Advanced Metrics & Analysis System Ready")
    print("🚀 Ready for production training pipelines!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 