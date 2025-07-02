"""
Test Script for VirtualAssistance Audit Fixes
Verifies that all the critical issues identified in the audit have been resolved
"""

import os
import sys
import torch
import json
from typing import Dict, List

def test_model_loading_fix():
    """Test Fix #1: Model loading bug (torch.path.exists → os.path.exists)"""
    print("🔧 Testing Fix #1: Model Loading Bug")
    
    try:
        from demo import ChordProgressionDemo
        
        # Test with non-existent model path
        demo = ChordProgressionDemo("non_existent_model.pth")
        print("   ✅ Model loading works with non-existent path")
        
        # Test with existing file (if available)
        if os.path.exists("chord_progression_model.pth"):
            demo_trained = ChordProgressionDemo("chord_progression_model.pth")
            print("   ✅ Model loading works with existing trained model")
        else:
            print("   ⚠️ No trained model found (chord_progression_model.pth)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        return False


def test_genre_mapping_fix():
    """Test Fix #2: Genre mapping implementation"""
    print("\n🔧 Testing Fix #2: Genre Mapping")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test genre catalog loading
        assert hasattr(model, 'genre_catalog'), "Genre catalog not loaded"
        assert len(model.genre_catalog) > 10, "Genre catalog too small"
        print(f"   ✅ Genre catalog loaded: {len(model.genre_catalog)} genres")
        
        # Test genre to index mapping
        assert hasattr(model, 'genre_to_idx'), "Genre to index mapping missing"
        
        # Test specific genre mappings
        test_genres = ["Pop", "Jazz", "Classical", "Rock"]
        for genre in test_genres:
            if genre in model.genre_to_idx:
                idx = model.get_genre_index(genre)
                print(f"   ✅ {genre} → index {idx}")
            else:
                print(f"   ⚠️ {genre} not in catalog")
        
        # Test fallback behavior
        unknown_idx = model.get_genre_index("UnknownGenre")
        assert unknown_idx == 0, "Unknown genre should fallback to index 0"
        print("   ✅ Unknown genre fallback works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Genre mapping test failed: {e}")
        return False


def test_mode_mapping_fix():
    """Test Fix #3: Mode mapping implementation"""
    print("\n🔧 Testing Fix #3: Mode Mapping")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test mode catalog
        assert hasattr(model, 'mode_catalog'), "Mode catalog not loaded"
        assert hasattr(model, 'mode_to_idx'), "Mode to index mapping missing"
        print(f"   ✅ Mode catalog loaded: {len(model.mode_catalog)} modes")
        
        # Test specific mode mappings
        test_modes = ["Ionian", "Dorian", "Aeolian", "Lydian"]
        for mode in test_modes:
            if mode in model.mode_to_idx:
                idx = model.get_mode_index(mode)
                print(f"   ✅ {mode} → index {idx}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mode mapping test failed: {e}")
        return False


def test_neural_generation_integration():
    """Test Fix #4: Neural generation pipeline integration"""
    print("\n🔧 Testing Fix #4: Neural Generation Integration")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test neural generation flags
        assert hasattr(model, 'use_neural_generation'), "Neural generation flag missing"
        assert hasattr(model, 'is_trained'), "Training flag missing"
        print("   ✅ Neural generation flags present")
        
        # Test enable/disable methods
        model.enable_neural_generation()
        assert model.use_neural_generation == True, "Enable neural generation failed"
        print("   ✅ Enable neural generation works")
        
        model.disable_neural_generation()
        assert model.use_neural_generation == False, "Disable neural generation failed"
        print("   ✅ Disable neural generation works")
        
        # Test generation method selection
        result = model.generate_from_prompt("happy", "Pop", 1)[0]
        assert 'metadata' in result, "Metadata missing from result"
        assert 'generation_method' in result['metadata'], "Generation method not recorded"
        print(f"   ✅ Generation method recorded: {result['metadata']['generation_method']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Neural generation integration test failed: {e}")
        return False


def test_enhanced_emotion_parsing():
    """Test Fix #5: Enhanced emotion parsing with BERT integration"""
    print("\n🔧 Testing Fix #5: Enhanced Emotion Parsing")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test both parsing methods
        test_prompt = "I'm feeling really happy and excited"
        
        # Test keyword-based parsing (fallback)
        model.disable_neural_generation()
        result_keyword = model.generate_from_prompt(test_prompt, "Pop", 1)[0]
        print("   ✅ Keyword-based emotion parsing works")
        
        # Test that emotions are properly weighted
        emotions = result_keyword['emotion_weights']
        assert isinstance(emotions, dict), "Emotions should be a dictionary"
        assert 'Joy' in emotions, "Joy emotion should be detected"
        print(f"   ✅ Joy emotion detected: {emotions.get('Joy', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced emotion parsing test failed: {e}")
        return False


def test_wolfram_integration():
    """Test Fix #6: Wolfram|Alpha integration"""
    print("\n🔧 Testing Fix #6: Wolfram Integration")
    
    try:
        from wolfram_validator import WolframMusicValidator
        
        # Test initialization
        validator = WolframMusicValidator()
        print(f"   ✅ Wolfram validator initialized")
        print(f"   📊 Wolfram enabled: {validator.enabled}")
        
        if validator.enabled:
            # Test validation with simple progression
            test_chords = ["I", "V", "vi", "IV"]
            result = validator.validate_progression(test_chords, "major", "C")
            assert 'validation_status' in result, "Validation status missing"
            print(f"   ✅ Progression validation works: {result['validation_status']}")
        else:
            print("   ⚠️ Wolfram disabled (set WOLFRAM_APP_ID to test)")
        
        # Test integration function
        from wolfram_validator import integrate_wolfram_validation
        mock_result = {
            'chords': ['I', 'V', 'vi', 'IV'],
            'primary_mode': 'Ionian'
        }
        enhanced_result = integrate_wolfram_validation(mock_result, validator)
        assert 'wolfram_validation' in enhanced_result, "Wolfram validation not integrated"
        print("   ✅ Wolfram integration function works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Wolfram integration test failed: {e}")
        return False


def test_end_to_end_pipeline():
    """Test Fix #7: Complete end-to-end pipeline"""
    print("\n🔧 Testing Fix #7: End-to-End Pipeline")
    
    try:
        from chord_progression_model import ChordProgressionModel
        from wolfram_validator import WolframMusicValidator, integrate_wolfram_validation
        
        # Initialize components
        model = ChordProgressionModel()
        validator = WolframMusicValidator()
        
        # Test complete pipeline
        test_prompt = "romantic but anxious"
        test_genre = "Jazz"
        
        print(f"   🎯 Testing prompt: '{test_prompt}' in {test_genre}")
        
        # Generate progression
        results = model.generate_from_prompt(test_prompt, test_genre, 1)
        assert len(results) > 0, "No results generated"
        result = results[0]
        
        # Verify result structure
        required_fields = ['chords', 'emotion_weights', 'primary_mode', 'genre', 'metadata']
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        print(f"   ✅ Generated: {' → '.join(result['chords'])}")
        print(f"   ✅ Mode: {result['primary_mode']}")
        print(f"   ✅ Top emotion: {max(result['emotion_weights'], key=result['emotion_weights'].get)}")
        
        # Test with Wolfram validation
        if validator.enabled:
            enhanced_result = integrate_wolfram_validation(result, validator)
            assert 'wolfram_validation' in enhanced_result, "Wolfram validation not added"
            print("   ✅ Wolfram validation integrated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end pipeline test failed: {e}")
        return False


def run_all_tests():
    """Run all fix verification tests"""
    print("🧪 Running VirtualAssistance Audit Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_model_loading_fix,
        test_genre_mapping_fix,
        test_mode_mapping_fix,
        test_neural_generation_integration,
        test_enhanced_emotion_parsing,
        test_wolfram_integration,
        test_end_to_end_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All audit fixes verified successfully!")
        print("\n✅ Your VirtualAssistance system is now fully functional:")
        print("   • Model loading bug fixed")
        print("   • Neural pipeline connected")
        print("   • Genre and mode mapping working")
        print("   • Wolfram validation available")
        print("   • Enhanced emotion parsing")
        print("   • End-to-end pipeline tested")
    else:
        print(f"⚠️ {failed} test(s) failed - some fixes may need attention")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 