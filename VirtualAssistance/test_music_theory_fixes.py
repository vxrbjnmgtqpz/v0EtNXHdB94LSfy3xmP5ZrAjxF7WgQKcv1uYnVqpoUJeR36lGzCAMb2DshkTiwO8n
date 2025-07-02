#!/usr/bin/env python3
"""
Test script to verify the chord progression and individual chord emotion fixes
"""

import sys
import os
sys.path.append('/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance')

from chord_progression_model import EmotionMusicDatabase, ChordProgressionGenerator
from individual_chord_model import IndividualChordModel

def test_anger_progressions():
    """Test that Anger progressions no longer contain major I with minor iv"""
    print("=== Testing Anger Progressions ===")
    
    # Initialize database
    database = EmotionMusicDatabase()
    
    # Get Anger progressions
    anger_progressions = database.chord_progressions.get("Anger", [])
    
    print(f"Found {len(anger_progressions)} Anger progressions:")
    
    for i, prog in enumerate(anger_progressions):
        print(f"  {i+1}. {prog.chords} (Mode: {prog.mode})")
        
        # Check for invalid combinations
        has_major_I = "I" in prog.chords
        has_minor_iv = "iv" in prog.chords
        
        if has_major_I and has_minor_iv:
            print(f"    ❌ INVALID: Contains both major I and minor iv")
        else:
            print(f"    ✅ Valid progression")
    
def test_minor_chord_emotions():
    """Test that minor chords (like iv) evoke sadness, not joy"""
    print("\n=== Testing Minor Chord Emotions ===")
    
    # Initialize individual chord model
    chord_model = IndividualChordModel()
    database = chord_model.database.chord_emotion_map
    
    # Find iv chords in the database
    iv_chords = [chord for chord in database if chord.roman_numeral == "iv"]
    
    print(f"Found {len(iv_chords)} iv chord entries:")
    
    for chord in iv_chords:
        print(f"\nChord: {chord.roman_numeral} ({chord.symbol}) in {chord.mode_context}")
        
        sadness = chord.emotion_weights.get('Sadness', 0)
        joy = chord.emotion_weights.get('Joy', 0)
        
        print(f"  Sadness: {sadness:.2f}, Joy: {joy:.2f}")
        
        if sadness > joy:
            print("  ✅ Correctly associates minor chord with sadness")
        else:
            print("  ❌ ISSUE: Minor chord has more joy than sadness")

def test_ionian_chords():
    """Test that Ionian mode only contains appropriate chords"""
    print("\n=== Testing Ionian Mode Chords ===")
    
    chord_model = IndividualChordModel()
    database = chord_model.database.chord_emotion_map
    
    # Find all chords in Ionian mode
    ionian_chords = [chord for chord in database if chord.mode_context == "Ionian"]
    
    print(f"Found {len(ionian_chords)} chord entries in Ionian mode:")
    
    expected_chords = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    invalid_chords = ["iv"]  # Minor iv should not exist in Ionian
    
    found_chords = set(chord.roman_numeral for chord in ionian_chords)
    
    print(f"Chords found in Ionian: {sorted(found_chords)}")
    
    for chord_name in expected_chords:
        if chord_name in found_chords:
            print(f"  ✅ {chord_name}: Found (expected)")
        else:
            print(f"  ⚠️ {chord_name}: Missing (should be present)")
    
    for chord_name in invalid_chords:
        if chord_name in found_chords:
            print(f"  ❌ {chord_name}: Found (should NOT be present in Ionian)")
        else:
            print(f"  ✅ {chord_name}: Correctly absent from Ionian")

if __name__ == "__main__":
    print("Testing Music Theory Fixes")
    print("=" * 50)
    
    test_anger_progressions()
    test_minor_chord_emotions()
    test_ionian_chords()
    
    print("\n" + "=" * 50)
    print("Test completed!")
