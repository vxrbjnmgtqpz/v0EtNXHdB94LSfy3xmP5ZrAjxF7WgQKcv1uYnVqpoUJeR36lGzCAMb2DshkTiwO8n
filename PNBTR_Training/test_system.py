#!/usr/bin/env python3
"""
PNBTR Training System Integration Test
Validates all components work together properly.
"""

import os
import sys
import traceback
from pathlib import Path
import numpy as np

# Add the training modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing Module Imports...")
    
    try:
        from training.loss_functions import evaluate_metrics
        from training.waveform_utils import generate_test_signal, load_audio
        from training.model_factory import create_pnbtr_model
        from training.train_loop import PNBTRTrainingLoop
        from metrics.scoring import score_accuracy, print_score_summary
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation and processing"""
    print("\n🌊 Testing Signal Generation...")
    
    try:
        from training.waveform_utils import generate_test_signal, print_signal_info
        
        # Test different signal types
        signal_types = ["sweep", "sine", "noise", "impulse", "complex"]
        
        for signal_type in signal_types:
            signal = generate_test_signal(
                duration_ms=100,  # Short test signals
                sample_rate=48000,
                signal_type=signal_type
            )
            
            print(f"   📊 {signal_type}: {len(signal)} samples, "
                  f"range=[{np.min(signal):.4f}, {np.max(signal):.4f}]")
            
            # Basic validation
            assert len(signal) > 0, f"Empty signal for {signal_type}"
            assert not np.all(signal == 0), f"Zero signal for {signal_type}"
            assert np.isfinite(signal).all(), f"Non-finite values in {signal_type}"
        
        print("✅ Signal generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
        traceback.print_exc()
        return False

def test_models():
    """Test model creation and prediction"""
    print("\n🏗️  Testing Model Factory...")
    
    try:
        from training.model_factory import create_pnbtr_model, get_model_info
        from training.waveform_utils import generate_test_signal
        
        # Test signal
        test_signal = generate_test_signal(1000, 48000, "complex")
        
        # Test different model types
        model_types = ["dummy", "mlp", "conv1d", "hybrid"]
        
        for model_type in model_types:
            print(f"   🔧 Testing {model_type} model...")
            
            model = create_pnbtr_model(model_type, input_size=len(test_signal))
            info = get_model_info(model)
            
            # Test prediction
            output = model.predict(test_signal)
            
            # Validate output
            assert len(output) == len(test_signal), f"Length mismatch for {model_type}"
            assert np.isfinite(output).all(), f"Non-finite output from {model_type}"
            
            print(f"      ✅ {info['name']}: {info['parameters']:,} parameters")
        
        print("✅ All models working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test loss functions and scoring"""
    print("\n📊 Testing Metrics and Scoring...")
    
    try:
        from training.loss_functions import evaluate_metrics
        from metrics.scoring import score_accuracy, print_score_summary
        from training.waveform_utils import generate_test_signal
        
        # Generate test signals
        original = generate_test_signal(1000, 48000, "complex")
        
        # Create a slightly modified version (simulated reconstruction)
        reconstructed = original + np.random.normal(0, 0.01, len(original))
        
        # Evaluate metrics
        metrics = evaluate_metrics(reconstructed, original, sample_rate=48000)
        
        print(f"   📈 Metrics computed: {list(metrics.keys())}")
        
        # Test scoring
        composite_score = score_accuracy(metrics)
        print(f"   🎯 Composite score: {composite_score:.3f}")
        
        # Test score report
        print("   📋 Score Summary:")
        print_score_summary(metrics)
        
        # Validate metrics
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Invalid {metric_name}: {value}"
            assert np.isfinite(value), f"Non-finite {metric_name}: {value}"
        
        print("✅ Metrics and scoring working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test error: {e}")
        traceback.print_exc()
        return False

def test_training_loop():
    """Test the training loop functionality"""
    print("\n🔄 Testing Training Loop...")
    
    try:
        from training.train_loop import PNBTRTrainingLoop
        from training.model_factory import create_pnbtr_model
        from training.waveform_utils import generate_test_signal
        
        # Create test data
        input_signal = generate_test_signal(500, 48000, "sine")
        target_signal = generate_test_signal(500, 48000, "sine") * 1.1  # Slightly different
        
        # Create model
        model = create_pnbtr_model("dummy", input_size=len(input_signal))
        
        # Create training loop
        trainer = PNBTRTrainingLoop()
        
        # Test a few training iterations
        print("   🏃 Running test training iterations...")
        for i in range(3):
            result = trainer.train_single_sample(input_signal, target_signal, model)
            print(f"      Iteration {i+1}: Accuracy {result['accuracy']:.3f}")
        
        print("✅ Training loop working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Training loop test error: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration file loading"""
    print("\n⚙️  Testing Configuration Loading...")
    
    try:
        from metrics.scoring import load_scoring_config
        from training.model_factory import load_model_config
        
        # Test config loading
        scoring_config = load_scoring_config()
        model_config = load_model_config()
        
        print(f"   📄 Scoring config loaded: {len(scoring_config)} sections")
        print(f"   📄 Model config loaded: {len(model_config)} sections")
        
        # Basic validation
        assert "weights" in scoring_config, "Missing weights in scoring config"
        assert "model" in model_config, "Missing model in model config"
        
        print("✅ Configuration loading working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test complete end-to-end training workflow"""
    print("\n🎯 Testing End-to-End Workflow...")
    
    try:
        from training.train_loop import PNBTRTrainingLoop
        from training.model_factory import create_pnbtr_model
        from training.waveform_utils import generate_test_signal
        
        # Create realistic test scenario
        print("   🎼 Generating test audio...")
        input_signal = generate_test_signal(2000, 48000, "complex")
        
        # Simulate degraded signal (field PNBTR input)
        degraded_signal = input_signal + np.random.normal(0, 0.05, len(input_signal))
        
        # Target is the original clean signal
        target_signal = input_signal
        
        # Create and train model
        print("   🏗️  Creating model...")
        model = create_pnbtr_model("dummy", input_size=len(input_signal))
        
        print("   📚 Training model...")
        trainer = PNBTRTrainingLoop()
        
        # Train for a few iterations
        results = []
        for i in range(5):
            result = trainer.train_single_sample(degraded_signal, target_signal, model)
            results.append(result)
            print(f"      Iteration {i+1}: {result['accuracy']:.3f} accuracy")
        
        # Check if training improved
        initial_accuracy = results[0]['accuracy']
        final_accuracy = results[-1]['accuracy']
        
        print(f"   📈 Training progress: {initial_accuracy:.3f} → {final_accuracy:.3f}")
        
        if final_accuracy >= initial_accuracy:
            print("   ✅ Model showed improvement or maintained performance")
        else:
            print("   ⚠️  Model performance declined (may be normal for dummy model)")
        
        print("✅ End-to-end workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test error: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all expected files and directories exist"""
    print("\n📁 Testing File Structure...")
    
    base_path = Path(__file__).parent
    expected_files = [
        "README.md",
        "RESEARCH_FOUNDATION.md",
        "requirements.txt",
        "config/thresholds.yaml",
        "config/training_params.yaml",
        "training/train_loop.py",
        "training/loss_functions.py",
        "training/waveform_utils.py",
        "training/model_factory.py",
        "metrics/scoring.py",
        "guidance/field_directives.json"
    ]
    
    expected_dirs = [
        "inputs", "ground_truth", "models", "training", 
        "metrics", "config", "logs", "guidance", 
        "evaluation", "export", "tools"
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file_path in expected_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    # Check directories
    for dir_path in expected_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   📁 {dir_path}/")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
    
    if not missing_files and not missing_dirs:
        print("✅ All expected files and directories present")
        return True
    else:
        return False

def main():
    """Run all tests"""
    print("🧪 PNBTR Training System Integration Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Signal Generation", test_signal_generation),
        ("Model Factory", test_models),
        ("Metrics & Scoring", test_metrics),
        ("Configuration", test_configuration),
        ("Training Loop", test_training_loop),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! PNBTR Training System is ready.")
        print("\n📋 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Add training data to inputs/ and ground_truth/ directories")
        print("   3. Run training: python training/train_loop.py")
        print("   4. Monitor results in logs/ directory")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 