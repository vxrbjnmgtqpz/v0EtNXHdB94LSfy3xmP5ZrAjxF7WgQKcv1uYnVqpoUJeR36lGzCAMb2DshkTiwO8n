#!/usr/bin/env python3
"""
Validate Fixed Chord Progression Database
Test the chord progression model with the audit-corrected progressions
"""

import json
import sys
import os

# Add the project root to Python path
sys.path.append('/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance')

from chord_progression_model import EmotionMusicDatabase, ChordProgression

def validate_fixed_progressions():
    """Test the chord progression model with fixed database"""
    
    print("=== Validating Fixed Chord Progression Database ===\n")
    
    # Initialize the database
    try:
        db = EmotionMusicDatabase()
        print("✓ Successfully loaded chord progression database")
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        return False
    
    # Test specific progressions that were fixed
    test_cases = [
        # JOY fixes
        ("Joy", "joy_008", ["I", "ii", "IV", "V"], "Should now use ii instead of iii"),
        ("Joy", "joy_009", ["I", "V", "vi", "ii", "IV", "I", "ii", "V"], "Should use ii instead of iii"),
        ("Joy", "joy_011", ["I", "IV", "ii", "V"], "Should use IV instead of vi"),
        
        # SADNESS fixes
        ("Sadness", "sad_003", ["i", "♭VII", "♭VI", "i"], "Should end on i not ♭VII"),
        ("Sadness", "sad_007", ["i", "iv", "i", "i"], "Should end on i for melancholy"),
        
        # TRUST fixes
        ("Trust", "trust_009", ["i", "IV", "V", "ii"], "Should use V instead of vi°"),
        ("Trust", "trust_004", ["i", "IV", "ii", "♭VII", "i"], "Should resolve to i"),
        
        # LOVE fixes  
        ("Love", "love_003", ["I", "♭VII", "V", "I"], "Should use V instead of v"),
        ("Love", "love_005", ["I", "♭VII", "IV", "v", "I"], "Should resolve to I"),
        
        # ANGER fixes
        ("Anger", "anger_006", ["I", "♭iii", "♭II", "I"], "Should use ♭iii instead of ♭III"),
        ("Anger", "anger_003", ["I", "v", "♭II", "I"], "Should use v instead of V"),
        
        # FEAR fixes
        ("Fear", "fear_010", ["i", "♭vi", "♭VII", "i"], "Should use ♭vi instead of ♭VI"),
        ("Fear", "fear_005", ["i", "♭II", "♭vi", "i"], "Should use ♭vi and end on i"),
        
        # DISGUST fixes
        ("Disgust", "disgust_004", ["♭v", "i°", "♭vi", "i°"], "Should use ♭vi and end on i°"),
        ("Disgust", "disgust_007", ["♭II", "♭v", "♭vi", "i°"], "Should use ♭vi instead of ♭VI"),
        
        # ANTICIPATION fixes
        ("Anticipation", "anticipation_002", ["i", "ii°", "V", "i"], "Should use ii° instead of IV"),
        
        # SHAME fixes
        ("Shame", "shame_010", ["i", "♭III", "iv", "V"], "Should use ♭III instead of ♭III+"),
        
        # ENVY fixes
        ("Envy", "envy_009", ["i", "V7", "♭II", "♯iv°"], "Should use V7 instead of V"),
    ]
    
    # Validate each test case
    passed = 0
    total = len(test_cases)
    
    for emotion, prog_id, expected_chords, description in test_cases:
        result = validate_progression(db, emotion, prog_id, expected_chords, description)
        if result:
            passed += 1
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Passed: {passed}/{total} progressions")
    
    if passed == total:
        print("✓ All audit fixes successfully applied and validated!")
        return True
    else:
        print(f"✗ {total - passed} progressions still need attention")
        return False

def validate_progression(db, emotion, prog_id, expected_chords, description):
    """Validate a specific progression fix"""
    
    try:
        # Get progressions for the emotion
        progressions = db.get_progressions_for_emotion(emotion)
        
        # Find the specific progression
        target_prog = None
        for prog in progressions:
            if prog.progression_id == prog_id:
                target_prog = prog
                break
        
        if target_prog is None:
            print(f"✗ {prog_id}: Progression not found")
            return False
        
        # Check if chords match expected
        if target_prog.chords == expected_chords:
            print(f"✓ {prog_id}: {target_prog.chords} - {description}")
            return True
        else:
            print(f"✗ {prog_id}: Expected {expected_chords}, got {target_prog.chords}")
            return False
            
    except Exception as e:
        print(f"✗ {prog_id}: Error - {e}")
        return False

def test_emotional_alignment():
    """Test if fixed progressions better align with their target emotions"""
    
    print("\n=== Testing Emotional Alignment ===\n")
    
    # Load database
    db = EmotionMusicDatabase()
    
    # Test examples from each emotion category
    test_emotions = [
        ("Joy", "Should sound bright and uplifting"),
        ("Sadness", "Should sound melancholy and resolved in minor"),
        ("Trust", "Should sound warm and supportive"),
        ("Love", "Should sound tender and resolved"),
        ("Anger", "Should sound aggressive and unstable"),
        ("Fear", "Should sound tense and ominous"),
        ("Disgust", "Should sound dissonant and unsettled"),
        ("Anticipation", "Should sound hopeful but unresolved"),
        ("Shame", "Should sound tragic and haunted"),
        ("Envy", "Should sound bitter and complex")
    ]
    
    for emotion, description in test_emotions:
        try:
            progressions = db.get_progressions_for_emotion(emotion)
            
            print(f"{emotion.upper()} ({description}):")
            
            # Show first 3 progressions as examples
            for i, prog in enumerate(progressions[:3]):
                chord_str = " - ".join(prog.chords)
                print(f"  {prog.progression_id}: {chord_str}")
            
            print()
            
        except Exception as e:
            print(f"Error testing {emotion}: {e}")

def main():
    """Main validation function"""
    
    # First validate the specific fixes
    validation_passed = validate_fixed_progressions()
    
    # Then test overall emotional alignment
    test_emotional_alignment()
    
    if validation_passed:
        print("🎵 Chord progression database is now emotionally aligned!")
        return True
    else:
        print("⚠️ Some issues remain in the chord progression database")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
