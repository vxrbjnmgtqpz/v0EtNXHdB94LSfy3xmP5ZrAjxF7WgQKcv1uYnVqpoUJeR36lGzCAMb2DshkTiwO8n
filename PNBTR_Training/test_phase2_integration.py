#!/usr/bin/env python3
"""
PNBTR Phase 2 Integration Test
Comprehensive validation of all Phase 2 components:
- Real PyTorch models vs dummy models
- Advanced loss functions  
- Enhanced training loop
- Complete system integration
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add the parent directory to sys.path to import modules
sys.path.insert(0, str(Path(__file__).parent))

def test_pytorch_models():
    """Test PyTorch model creation and functionality"""
    print("🏗️  Testing PyTorch Models...")
    
    try:
        from training.pytorch_models import create_pytorch_model
        
        # Test different model types
        model_types = ["mlp", "conv1d", "hybrid", "transformer"]
        test_signal = np.random.normal(0, 0.1, 1024).astype(np.float64)
        
        for model_type in model_types:
            print(f"   Testing {model_type}...")
            
            # Create model
            model = create_pytorch_model(model_type, 1024)
            
            # Test prediction
            output = model.predict(test_signal)
            
            # Validate output
            assert output.shape == test_signal.shape, f"Shape mismatch: {output.shape} vs {test_signal.shape}"
            assert np.isfinite(output).all(), "Output contains non-finite values"
            
            print(f"      ✅ {model_type}: {model.name}")
            print(f"         Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"         Output range: [{np.min(output):.4f}, {np.max(output):.4f}]")
        
        return True
        
    except Exception as e:
        print(f"      ❌ PyTorch models test failed: {e}")
        return False

def test_advanced_loss():
    """Test advanced loss function components"""
    print("📊 Testing Advanced Loss Functions...")
    
    try:
        from training.pytorch_trainer import PNBTRAdvancedLoss
        import torch
        
        # Create test signals
        target = torch.randn(1, 1000)
        
        # Perfect prediction (should give very low loss)
        perfect_pred = target.clone()
        
        # Poor prediction
        poor_pred = torch.randn(1, 1000) * 2
        
        # Initialize loss function
        loss_fn = PNBTRAdvancedLoss()
        
        # Test losses
        perfect_loss = loss_fn(perfect_pred, target)
        poor_loss = loss_fn(poor_pred, target)
        
        print(f"   Perfect prediction loss: {perfect_loss.item():.6f}")
        print(f"   Poor prediction loss: {poor_loss.item():.6f}")
        
        # Validate loss behavior
        assert perfect_loss < poor_loss, "Loss function not working correctly"
        assert perfect_loss < 0.1, "Perfect prediction should have very low loss"
        
        print("      ✅ Advanced loss function working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ Advanced loss test failed: {e}")
        return False

def test_pytorch_trainer():
    """Test PyTorch trainer functionality"""
    print("🚀 Testing PyTorch Trainer...")
    
    try:
        from training.pytorch_models import create_pytorch_model
        from training.pytorch_trainer import create_pytorch_trainer
        
        # Create simple model and signals
        model = create_pytorch_model("mlp", 512)
        
        # Create training signals (target is input + small known transformation)
        input_signal = np.random.normal(0, 0.1, 512).astype(np.float64)
        target_signal = input_signal * 0.9 + 0.01  # Slight scaling and offset
        
        # Create trainer
        trainer = create_pytorch_trainer(model)
        
        # Quick training test (just a few iterations)
        print("   Running quick training test...")
        
        # Override max epochs for fast test
        trainer.max_epochs = 20
        
        result = trainer.train_single_sample(input_signal, target_signal)
        
        print(f"   Training result:")
        print(f"      Epochs: {result['epochs_trained']}")
        print(f"      Final loss: {result['final_loss']:.6f}")
        print(f"      Accuracy: {result['accuracy']:.3f}")
        print(f"      Mastery: {result['mastery_achieved']}")
        
        # Validate training occurred
        assert result['epochs_trained'] > 0, "No training iterations occurred"
        assert 'accuracy' in result, "Accuracy not calculated"
        
        print("      ✅ PyTorch trainer working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ PyTorch trainer test failed: {e}")
        return False

def test_model_factory():
    """Test enhanced model factory with PyTorch integration"""
    print("🏭 Testing Enhanced Model Factory...")
    
    try:
        from training.model_factory import create_pnbtr_model, get_model_info
        
        # Test PyTorch model creation
        pytorch_model = create_pnbtr_model("mlp", use_pytorch=True, input_size=1024)
        pytorch_info = get_model_info(pytorch_model)
        
        print(f"   PyTorch model: {pytorch_info['name']}")
        print(f"      Parameters: {pytorch_info['parameters']:,}")
        print(f"      Is PyTorch: {pytorch_info['is_pytorch']}")
        
        # Test dummy model creation
        dummy_model = create_pnbtr_model("mlp", use_pytorch=False, input_size=1024)
        dummy_info = get_model_info(dummy_model)
        
        print(f"   Dummy model: {dummy_info['name']}")
        print(f"      Parameters: {dummy_info['parameters']:,}")
        print(f"      Is PyTorch: {dummy_info['is_pytorch']}")
        
        # Test predictions
        test_signal = np.random.normal(0, 0.1, 1024).astype(np.float64)
        
        pytorch_output = pytorch_model.predict(test_signal)
        dummy_output = dummy_model.predict(test_signal)
        
        assert pytorch_output.shape == test_signal.shape, "PyTorch model shape mismatch"
        assert dummy_output.shape == test_signal.shape, "Dummy model shape mismatch"
        
        print("      ✅ Both PyTorch and dummy models working")
        return True
        
    except Exception as e:
        print(f"      ❌ Model factory test failed: {e}")
        return False

def test_enhanced_training_loop():
    """Test enhanced training loop with PyTorch integration"""
    print("🔄 Testing Enhanced Training Loop...")
    
    try:
        from training.train_loop import PNBTRTrainingLoop
        from training.waveform_utils import generate_test_signal
        
        # Create training loop
        trainer = PNBTRTrainingLoop()
        
        # Generate test signals
        input_signal = generate_test_signal(800, 48000, "harmonic")
        # Create target with known relationship
        target_signal = input_signal + np.random.normal(0, 0.01, len(input_signal))
        
        print(f"   Training on {len(input_signal)} sample signal...")
        
        # Train single sample
        result = trainer.train_single_sample(input_signal, target_signal)
        
        print(f"   Training result:")
        print(f"      Model type: {'PyTorch' if trainer.stats['pytorch_models'] > 0 else 'Dummy'}")
        print(f"      Accuracy: {result['accuracy']:.3f}")
        print(f"      Training time: {result.get('training_time_seconds', 0):.2f}s")
        print(f"      Mastery achieved: {result.get('mastery_achieved', False)}")
        
        # Validate statistics
        assert trainer.stats['samples_trained'] == 1, "Sample count incorrect"
        assert result['accuracy'] >= 0.0, "Invalid accuracy value"
        
        print("      ✅ Enhanced training loop working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ Enhanced training loop test failed: {e}")
        return False

def test_waveform_utilities():
    """Test waveform generation and processing utilities"""
    print("🌊 Testing Waveform Utilities...")
    
    try:
        from training.waveform_utils import generate_test_signal, align_signals
        
        # Test different signal types
        signal_types = ["sine", "complex", "harmonic", "noise"]
        
        for signal_type in signal_types:
            signal = generate_test_signal(1000, 48000, signal_type)
            
            assert len(signal) == 1000, f"Incorrect signal length for {signal_type}"
            assert np.isfinite(signal).all(), f"Non-finite values in {signal_type}"
            
            print(f"   ✅ {signal_type}: range [{np.min(signal):.4f}, {np.max(signal):.4f}]")
        
        # Test signal alignment
        signal1 = generate_test_signal(500, 48000, "sine")
        signal2 = generate_test_signal(600, 48000, "sine")  # Different length
        
        aligned1, aligned2 = align_signals(signal1, signal2)
        
        assert len(aligned1) == len(aligned2), "Signals not properly aligned"
        
        print("      ✅ Signal alignment working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ Waveform utilities test failed: {e}")
        return False

def test_metrics_and_scoring():
    """Test metrics evaluation and scoring system"""
    print("📈 Testing Metrics and Scoring...")
    
    try:
        from training.loss_functions import evaluate_metrics
        from metrics.scoring import score_accuracy, meets_mastery_threshold
        
        # Create test signals
        target = np.random.normal(0, 0.1, 1000)
        
        # Perfect prediction
        perfect_pred = target.copy()
        
        # Poor prediction  
        poor_pred = np.random.normal(0, 0.5, 1000)
        
        # Evaluate metrics
        perfect_metrics = evaluate_metrics(perfect_pred, target, sample_rate=48000)
        poor_metrics = evaluate_metrics(poor_pred, target, sample_rate=48000)
        
        # Score accuracy
        perfect_accuracy = score_accuracy(perfect_metrics)
        poor_accuracy = score_accuracy(poor_metrics)
        
        print(f"   Perfect prediction accuracy: {perfect_accuracy:.3f}")
        print(f"   Poor prediction accuracy: {poor_accuracy:.3f}")
        
        # Test mastery threshold
        perfect_mastery = meets_mastery_threshold(perfect_accuracy)
        poor_mastery = meets_mastery_threshold(poor_accuracy)
        
        print(f"   Perfect meets mastery: {perfect_mastery}")
        print(f"   Poor meets mastery: {poor_mastery}")
        
        # Validate scoring behavior
        assert perfect_accuracy > poor_accuracy, "Scoring not working correctly"
        assert perfect_accuracy > 0.8, "Perfect prediction should score highly"
        
        print("      ✅ Metrics and scoring working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ Metrics and scoring test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration loading and parameter management"""
    print("⚙️  Testing Configuration System...")
    
    try:
        from config.thresholds import load_thresholds
        import yaml
        
        # Test threshold loading
        thresholds = load_thresholds()
        
        assert "quality_tiers" in thresholds, "Quality tiers not found in thresholds"
        assert "mastery_threshold" in thresholds, "Mastery threshold not found"
        
        print(f"   Mastery threshold: {thresholds['mastery_threshold']}")
        print(f"   Quality tiers: {list(thresholds['quality_tiers'].keys())}")
        
        # Test training params loading
        config_path = Path(__file__).parent / "config" / "training_params.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert "model" in config, "Model config not found"
            assert "optimization" in config, "Optimization config not found"
            
            print(f"   Model types available: {list(config['model'].keys())}")
        
        print("      ✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"      ❌ Configuration system test failed: {e}")
        return False

def run_comprehensive_integration_test():
    """Run complete integration test of all Phase 2 components"""
    print("=" * 60)
    print("🧪 PNBTR PHASE 2 COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all test components
    tests = [
        ("Waveform Utilities", test_waveform_utilities),
        ("Configuration System", test_configuration_system),
        ("PyTorch Models", test_pytorch_models),
        ("Advanced Loss Functions", test_advanced_loss),
        ("PyTorch Trainer", test_pytorch_trainer),
        ("Enhanced Model Factory", test_model_factory),
        ("Metrics and Scoring", test_metrics_and_scoring),
        ("Enhanced Training Loop", test_enhanced_training_loop),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🏆 ALL TESTS PASSED - Phase 2 is ready!")
        print("✅ PyTorch integration complete")
        print("✅ Advanced training systems operational")
        print("✅ Enhanced models functioning correctly")
    else:
        print(f"\n⚠️  {total - passed} tests failed - Phase 2 needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1) 