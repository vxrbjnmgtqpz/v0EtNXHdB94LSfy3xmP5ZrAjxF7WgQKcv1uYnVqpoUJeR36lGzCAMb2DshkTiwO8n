#!/usr/bin/env python3
"""
Test direct access to Transcendence chord progressions
"""

import json
import sys

def test_direct_transcendence_access():
    """Test accessing Transcendence progressions directly from the database"""
    
    try:
        # Load the database
        with open('emotion_progression_database.json', 'r') as f:
            db = json.load(f)
        
        # Access Transcendence directly
        transcendence = db['emotions']['Transcendence']
        print("✅ Successfully loaded Transcendence emotion")
        
        # Test access to Lucid_Wonder
        lucid_wonder = transcendence['sub_emotions']['Lucid_Wonder']
        print("✅ Successfully accessed Lucid_Wonder sub-emotion")
        
        print(f"Mode: {lucid_wonder['mode']}")
        print(f"Description: {lucid_wonder['description']}")
        
        progressions = lucid_wonder['progression_pool']
        print(f"Found {len(progressions)} progressions:")
        
        for i, prog in enumerate(progressions):
            chords = prog['chords']
            genres = prog.get('genres', {})
            print(f"  {i+1}. {chords}")
            print(f"     Genres: {genres}")
        
        # Test other transcendence emotions
        print(f"\n🔍 Testing other Transcendence sub-emotions:")
        sub_emotions = list(transcendence['sub_emotions'].keys())
        print(f"Available: {sub_emotions}")
        
        # Test Cosmic_Unity
        cosmic_unity = transcendence['sub_emotions']['Cosmic_Unity']
        cosmic_progressions = cosmic_unity['progression_pool']
        print(f"\nCosmic_Unity has {len(cosmic_progressions)} progressions:")
        for i, prog in enumerate(cosmic_progressions):
            chords = prog['chords']
            print(f"  {i+1}. {chords}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_transcendence_access()
    sys.exit(0 if success else 1)
