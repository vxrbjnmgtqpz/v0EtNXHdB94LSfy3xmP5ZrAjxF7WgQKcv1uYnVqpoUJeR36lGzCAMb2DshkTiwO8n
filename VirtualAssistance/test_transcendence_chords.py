#!/usr/bin/env python3
"""
Quick test to verify Transcendence chord progressions are accessible
"""

import json
import sys

def test_transcendence_chords():
    """Test that Transcendence sub-emotions have chord progressions"""
    
    try:
        # Load the emotion database
        with open('emotion_progression_database.json', 'r') as f:
            db = json.load(f)
        
        transcendence = db['emotions']['Transcendence']
        print(f"Transcendence mode: {transcendence['mode']}")
        print(f"Transcendence description: {transcendence['description']}")
        
        sub_emotions = transcendence['sub_emotions']
        print(f"\nFound {len(sub_emotions)} sub-emotions:")
        
        total_progressions = 0
        missing_chords = []
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            progressions = sub_emotion_data.get('progression_pool', [])
            progression_count = len(progressions)
            total_progressions += progression_count
            
            print(f"\n{sub_emotion_name}:")
            print(f"  Mode: {sub_emotion_data['mode']}")
            print(f"  Progressions: {progression_count}")
            
            # Check each progression for chord data
            for i, progression in enumerate(progressions):
                if 'chords' not in progression:
                    missing_chords.append(f"{sub_emotion_name} progression {i+1}")
                    print(f"    ❌ Progression {i+1}: Missing chords")
                else:
                    chords = progression['chords']
                    print(f"    ✅ Progression {i+1}: {chords}")
        
        print(f"\n📊 SUMMARY:")
        print(f"Total sub-emotions: {len(sub_emotions)}")
        print(f"Total progressions: {total_progressions}")
        print(f"Missing chord data: {len(missing_chords)}")
        
        if missing_chords:
            print(f"\n❌ ISSUES FOUND:")
            for issue in missing_chords:
                print(f"  - {issue}")
            return False
        else:
            print(f"\n✅ ALL TRANSCENDENCE PROGRESSIONS HAVE CHORD DATA!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing transcendence chords: {e}")
        return False

if __name__ == "__main__":
    success = test_transcendence_chords()
    sys.exit(0 if success else 1)
