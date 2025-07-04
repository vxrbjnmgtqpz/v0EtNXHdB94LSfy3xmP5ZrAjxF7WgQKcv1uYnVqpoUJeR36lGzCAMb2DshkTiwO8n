#!/usr/bin/env python3
"""
Transcendence Integration Test Script
Tests that all system components properly recognize and work with the new Transcendence emotion.
"""

import json
import sys
from pathlib import Path

# Test 1: Database Validation
def test_database_integration():
    """Test that the emotion database properly includes Transcendence"""
    print("🔍 Testing Database Integration...")
    
    try:
        with open('emotion_progression_database.json', 'r') as f:
            db = json.load(f)
        
        # Check database info
        info = db.get('database_info', {})
        print(f"  ✓ Database version: {info.get('version')}")
        print(f"  ✓ Total emotions: {info.get('total_emotions')}")
        print(f"  ✓ Total sub-emotions: {info.get('total_sub_emotions')}")
        
        # Check Transcendence exists
        emotions = db.get('emotions', {})
        if 'Transcendence' not in emotions:
            print("  ❌ Transcendence emotion not found in database!")
            return False
        
        # Check Transcendence structure
        transcendence = emotions['Transcendence']
        sub_emotions = transcendence.get('sub_emotions', {})
        print(f"  ✓ Transcendence has {len(sub_emotions)} sub-emotions")
        
        # Check some key sub-emotions
        expected_sub_emotions = [
            'Lucid_Wonder', 'Ego_Death', 'Dreamflight', 'Sacred_Dissonance',
            'Cosmic_Unity', 'Divine_Ecstasy', 'Harmonic_Nirvana'
        ]
        
        for sub_emotion in expected_sub_emotions:
            if sub_emotion in sub_emotions:
                print(f"    ✓ {sub_emotion} found")
            else:
                print(f"    ❌ {sub_emotion} missing")
                return False
        
        # Check keywords
        parser_rules = db.get('parser_rules', {})
        emotion_keywords = parser_rules.get('emotion_keywords', {})
        if 'Transcendence' not in emotion_keywords:
            print("  ❌ Transcendence keywords not found!")
            return False
        
        transcendence_keywords = emotion_keywords['Transcendence']
        print(f"  ✓ Transcendence has {len(transcendence_keywords.get('primary', []))} primary keywords")
        print(f"  ✓ Transcendence has {len(transcendence_keywords.get('sub_emotion_keywords', {}))} sub-emotion keyword sets")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False

# Test 2: Neural Network Integration
def test_neural_network_integration():
    """Test that neural networks include Transcendence"""
    print("\n🧠 Testing Neural Network Integration...")
    
    try:
        # Import and check neural progression analyzer
        from neural_progression_analyzer import NeuralProgressionAnalyzer, EmotionAnalysisEngine
        
        analyzer = NeuralProgressionAnalyzer()
        if 'Transcendence' not in analyzer.emotion_labels:
            print("  ❌ NeuralProgressionAnalyzer missing Transcendence!")
            return False
        
        print(f"  ✓ NeuralProgressionAnalyzer has {len(analyzer.emotion_labels)} emotions including Transcendence")
        
        engine = EmotionAnalysisEngine()
        if 'Transcendence' not in engine.emotion_labels:
            print("  ❌ EmotionAnalysisEngine missing Transcendence!")
            return False
        
        print(f"  ✓ EmotionAnalysisEngine has {len(engine.emotion_labels)} emotions including Transcendence")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Neural network test failed: {e}")
        return False

# Test 3: Interpolation Engine Integration
def test_interpolation_integration():
    """Test that interpolation engine includes Transcendence"""
    print("\n🌊 Testing Interpolation Engine Integration...")
    
    try:
        from emotion_interpolation_engine import EmotionInterpolationEngine
        
        engine = EmotionInterpolationEngine()
        
        if 'Transcendence' not in engine.emotion_labels:
            print("  ❌ InterpolationEngine missing Transcendence!")
            return False
        
        print(f"  ✓ InterpolationEngine has {len(engine.emotion_labels)} emotions including Transcendence")
        
        # Check if Transcendence has a mode mapping
        if 'Transcendence' not in engine.emotion_modes:
            print("  ❌ Transcendence mode mapping missing!")
            return False
        
        print(f"  ✓ Transcendence mode: {engine.emotion_modes['Transcendence']}")
        
        # Check compatibility matrix
        compatibility_matrix = engine._build_compatibility_matrix()
        if 'Transcendence' not in compatibility_matrix:
            print("  ❌ Transcendence compatibility mapping missing!")
            return False
        
        print("  ✓ Transcendence compatibility matrix generated")
        
        # Check consonance mapping
        consonance_map = engine._build_emotion_consonance_map()
        if 'Transcendence' not in consonance_map:
            print("  ❌ Transcendence consonance mapping missing!")
            return False
        
        print(f"  ✓ Transcendence consonance value: {consonance_map['Transcendence']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Interpolation engine test failed: {e}")
        return False

# Test 4: Chord Models Integration
def test_chord_models_integration():
    """Test that chord models include Transcendence"""
    print("\n🎵 Testing Chord Models Integration...")
    
    try:
        from chord_progression_model import ChordProgressionModel, EmotionParser
        from individual_chord_model import IndividualChordModel, EmotionParser as IndividualEmotionParser
        
        # Test progression model
        prog_parser = EmotionParser()
        if 'Transcendence' not in prog_parser.emotion_labels:
            print("  ❌ ProgressionModel EmotionParser missing Transcendence!")
            return False
        
        print(f"  ✓ ProgressionModel has {len(prog_parser.emotion_labels)} emotions including Transcendence")
        
        # Test individual chord model
        ind_parser = IndividualEmotionParser()
        if 'Transcendence' not in ind_parser.emotion_labels:
            print("  ❌ IndividualChordModel EmotionParser missing Transcendence!")
            return False
        
        print(f"  ✓ IndividualChordModel has {len(ind_parser.emotion_labels)} emotions including Transcendence")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Chord models test failed: {e}")
        return False

# Test 5: Functionality Test
def test_transcendence_functionality():
    """Test that Transcendence emotion actually works"""
    print("\n✨ Testing Transcendence Functionality...")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test transcendent progression generation
        transcendent_queries = [
            "mystical ethereal progression",
            "cosmic unity spiritual music", 
            "lucid dream floating melody",
            "transcendent enlightenment harmony"
        ]
        
        for query in transcendent_queries:
            try:
                result = model.generate_progressions(query, num_progressions=1)
                if result and len(result) > 0:
                    print(f"  ✓ Generated progression for: '{query}'")
                else:
                    print(f"  ⚠️  No progression generated for: '{query}'")
            except Exception as e:
                print(f"  ❌ Failed to generate progression for '{query}': {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🌀 Transcendence Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_database_integration,
        test_neural_network_integration,
        test_interpolation_integration,
        test_chord_models_integration,
        test_transcendence_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("  ❌ Test failed!")
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Transcendence is fully integrated!")
        return True
    else:
        print("⚠️  Some tests failed. Integration is incomplete.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
