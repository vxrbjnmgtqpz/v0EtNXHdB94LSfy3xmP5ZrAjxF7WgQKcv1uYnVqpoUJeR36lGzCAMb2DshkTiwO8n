#!/usr/bin/env python3
"""
Final Transcendence Integration Test
Tests all aspects of the Transcendence emotion integration across the system.
"""

import requests
import json
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chord_progression_model import ChordProgressionModel
from individual_chord_model import IndividualChordModel
from emotion_interpolation_engine import EmotionInterpolationEngine

def test_server_endpoints():
    """Test server endpoints for Transcendence functionality"""
    print("🔍 Testing Server Endpoints...")
    
    base_url = "http://localhost:5004"
    
    # Test chat endpoint with Transcendence prompt
    try:
        response = requests.post(f"{base_url}/chat/integrated", 
                               json={"message": "mystic insight and harmonic nirvana"},
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat endpoint responding")
            transcendence_weight = data.get('emotion', {}).get('Transcendence', 0)
            if transcendence_weight > 0:
                print(f"✅ Transcendence detected: {transcendence_weight:.3f}")
                print(f"Sub-emotion: {data.get('primary_result', {}).get('detected_sub_emotion', 'None')}")
            else:
                print("⚠️ Transcendence not detected in chat response")
            print(f"Response preview: {data.get('message', '')[:200]}...")
        else:
            print(f"❌ Chat endpoint error: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Test progression endpoint doesn't exist - skip it
    print("⚠️ Progression endpoint test skipped (endpoint doesn't exist)")

def test_models():
    """Test all models for Transcendence support"""
    print("\n🔍 Testing Models...")
    
    # Test ChordProgressionModel
    try:
        cpm = ChordProgressionModel()
        
        # Check if Transcendence is in emotion labels
        if "Transcendence" in cpm.emotion_labels:
            print("✅ ChordProgressionModel has Transcendence in emotion_labels")
        else:
            print("❌ ChordProgressionModel missing Transcendence in emotion_labels")
        
        # Test progression generation
        progression = cpm.generate_progression("Transcendence", length=4)
        if progression:
            print(f"✅ ChordProgressionModel generated progression: {progression}")
        else:
            print("❌ ChordProgressionModel failed to generate progression")
            
    except Exception as e:
        print(f"❌ ChordProgressionModel error: {e}")
    
    # Test IndividualChordModel
    try:
        icm = IndividualChordModel()
        
        # Check if Transcendence is in emotion labels
        if "Transcendence" in icm.emotion_labels:
            print("✅ IndividualChordModel has Transcendence in emotion_labels")
        else:
            print("❌ IndividualChordModel missing Transcendence in emotion_labels")
        
        # Test chord analysis
        chord_analysis = icm.analyze_chord("Cmaj7")
        if chord_analysis and len(chord_analysis.get('emotion_weights', {})) >= 23:
            print("✅ IndividualChordModel returns 23+ emotions")
            if 'Transcendence' in chord_analysis['emotion_weights']:
                print("✅ IndividualChordModel includes Transcendence in analysis")
            else:
                print("❌ IndividualChordModel missing Transcendence in analysis")
        else:
            print("❌ IndividualChordModel analysis incomplete")
            
    except Exception as e:
        print(f"❌ IndividualChordModel error: {e}")
    
    # Test EmotionInterpolationEngine
    try:
        eie = EmotionInterpolationEngine()
        
        # Check if Transcendence is in emotion labels
        if "Transcendence" in eie.emotion_labels:
            print("✅ EmotionInterpolationEngine has Transcendence in emotion_labels")
        else:
            print("❌ EmotionInterpolationEngine missing Transcendence in emotion_labels")
            
    except Exception as e:
        print(f"❌ EmotionInterpolationEngine error: {e}")

def test_database_integrity():
    """Test database files for Transcendence data"""
    print("\n🔍 Testing Database Integrity...")
    
    # Test emotion progression database
    try:
        with open('emotion_progression_database.json', 'r') as f:
            emotion_db = json.load(f)
        
        if "Transcendence" in emotion_db:
            print("✅ Transcendence found in emotion_progression_database.json")
            
            transcendence_data = emotion_db["Transcendence"]
            sub_emotions = len(transcendence_data.get("sub_emotions", {}))
            progressions = len(transcendence_data.get("progression_pools", {}).get("base_progressions", []))
            
            print(f"   Sub-emotions: {sub_emotions}")
            print(f"   Base progressions: {progressions}")
            
            if sub_emotions >= 20:
                print("✅ Transcendence has sufficient sub-emotions")
            else:
                print("⚠️ Transcendence may need more sub-emotions")
                
            if progressions >= 30:
                print("✅ Transcendence has sufficient progressions")
            else:
                print("⚠️ Transcendence may need more progressions")
        else:
            print("❌ Transcendence not found in emotion_progression_database.json")
            
    except Exception as e:
        print(f"❌ Error reading emotion progression database: {e}")
    
    # Test individual chord database
    try:
        with open('individual_chord_database_updated.json', 'r') as f:
            chord_db = json.load(f)
        
        # Check a few sample chords
        sample_chords = ["C", "Dm", "G7", "Cmaj7", "Am"]
        transcendence_found = 0
        
        for chord in sample_chords:
            if chord in chord_db:
                chord_data = chord_db[chord]
                if len(chord_data.get('emotion_weights', [])) >= 23:
                    transcendence_found += 1
        
        if transcendence_found == len(sample_chords):
            print("✅ Individual chord database has 23+ emotions for all sample chords")
        else:
            print(f"⚠️ Only {transcendence_found}/{len(sample_chords)} sample chords have 23+ emotions")
            
    except Exception as e:
        print(f"❌ Error reading individual chord database: {e}")

def main():
    """Run all tests"""
    print("🎵 Final Transcendence Integration Test")
    print("=" * 50)
    
    test_server_endpoints()
    test_models()
    test_database_integrity()
    
    print("\n" + "=" * 50)
    print("🎯 Test Complete!")
    
    # Final API test with specific Transcendence query
    print("\n🔍 Final API Test with Transcendence Query...")
    try:
        response = requests.post("http://localhost:5004/chat/integrated",
                               json={"message": "mystic insight and inner rebirth"},
                               timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("✅ Final API test successful")
            transcendence_weight = data.get('emotion', {}).get('Transcendence', 0)
            sub_emotion = data.get('primary_result', {}).get('detected_sub_emotion', '')
            print(f"🎯 Transcendence weight: {transcendence_weight:.3f}")
            print(f"🎭 Sub-emotion detected: {sub_emotion}")
            print(f"🎵 Generated progression: {data.get('chords', [])}")
            print(f"Response: {data.get('message', '')}")
        else:
            print(f"❌ Final API test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Final API test error: {e}")

if __name__ == "__main__":
    main()
