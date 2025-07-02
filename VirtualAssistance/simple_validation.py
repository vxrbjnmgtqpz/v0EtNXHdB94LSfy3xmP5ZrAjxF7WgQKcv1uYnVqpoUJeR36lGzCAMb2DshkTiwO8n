#!/usr/bin/env python3
"""
Simple validation of fixed chord progression database
Tests without requiring PyTorch dependencies
"""

import json
import os

def validate_chord_progression_fixes():
    """Validate the fixes applied to the chord progression database"""
    
    print("=== Validating Fixed Chord Progression Database ===\n")
    
    # Load the fixed database
    database_path = "/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance/emotion_progression_database.json"
    
    try:
        with open(database_path, 'r') as f:
            database = json.load(f)
        print("✓ Successfully loaded fixed chord progression database")
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        return False
    
    # Test cases with expected fixes
    test_cases = [
        # JOY fixes - should be brighter, less melancholy
        ("Joy", "joy_008", ["I", "ii", "IV", "V"], "Fixed: iii→ii for less somber tone"),
        ("Joy", "joy_009", ["I", "V", "vi", "ii", "IV", "I", "ii", "V"], "Fixed: iii→ii to maintain joy"),
        ("Joy", "joy_011", ["I", "IV", "ii", "V"], "Fixed: vi→IV for warmer progression"),
        
        # SADNESS fixes - should resolve properly to minor
        ("Sadness", "sad_003", ["i", "♭VII", "♭VI", "i"], "Fixed: ends on i for melancholy"),
        ("Sadness", "sad_007", ["i", "iv", "i", "i"], "Fixed: ends on i for mournful feel"),
        
        # TRUST fixes - should be more stable, less dissonant
        ("Trust", "trust_009", ["i", "IV", "V", "ii"], "Fixed: vi°→V removes dissonance"),
        ("Trust", "trust_004", ["i", "IV", "ii", "♭VII", "i"], "Fixed: added resolution to i"),
        
        # LOVE fixes - should be warmer, more resolved
        ("Love", "love_003", ["I", "♭VII", "V", "I"], "Fixed: v→V for brighter resolution"),
        ("Love", "love_005", ["I", "♭VII", "IV", "v", "I"], "Fixed: added resolution to I"),
        
        # ANGER fixes - should be more menacing, unstable
        ("Anger", "anger_006", ["I", "♭iii", "♭II", "I"], "Fixed: ♭III→♭iii for menace"),
        ("Anger", "anger_003", ["I", "v", "♭II", "I"], "Fixed: V→v maintains tension"),
        
        # FEAR fixes - should be more ominous, less consonant
        ("Fear", "fear_010", ["i", "♭vi", "♭VII", "i"], "Fixed: ♭VI→♭vi removes warmth"),
        ("Fear", "fear_005", ["i", "♭II", "♭vi", "i"], "Fixed: ♭VI→♭vi, ends on i"),
        
        # DISGUST fixes - should be consistently dissonant
        ("Disgust", "disgust_004", ["♭v", "i°", "♭vi", "i°"], "Fixed: ♭VI→♭vi, ends on i°"),
        ("Disgust", "disgust_007", ["♭II", "♭v", "♭vi", "i°"], "Fixed: ♭VI→♭vi for sourness"),
        
        # ANTICIPATION fixes - should maintain suspense
        ("Anticipation", "anticipation_002", ["i", "ii°", "V", "i"], "Fixed: IV→ii° for tension"),
        
        # SHAME fixes - should emphasize tragedy
        ("Shame", "shame_010", ["i", "♭III", "iv", "V"], "Fixed: ♭III+→♭III for focus"),
        
        # ENVY fixes - should be more complex, twisted
        ("Envy", "envy_009", ["i", "V7", "♭II", "♯iv°"], "Fixed: V→V7 for complexity"),
    ]
    
    # Validate each test case
    passed = 0
    failed = 0
    
    for emotion, prog_id, expected_chords, description in test_cases:
        if validate_specific_progression(database, emotion, prog_id, expected_chords, description):
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Success Rate: {passed}/{passed + failed} ({100 * passed / (passed + failed):.1f}%)")
    
    if failed == 0:
        print("\n🎵 All chord progression fixes successfully validated!")
        show_emotional_improvements(database)
        return True
    else:
        print(f"\n⚠️ {failed} progressions need additional attention")
        return False

def validate_specific_progression(database, emotion, prog_id, expected_chords, description):
    """Validate a specific progression against expected chords"""
    
    try:
        # Navigate to the emotion's progressions
        emotion_data = database["emotions"][emotion]
        progressions = emotion_data["progression_pool"]
        
        # Find the specific progression
        for prog in progressions:
            if prog["progression_id"] == prog_id:
                actual_chords = prog["chords"]
                
                if actual_chords == expected_chords:
                    print(f"✓ {prog_id}: {' - '.join(actual_chords)} | {description}")
                    return True
                else:
                    print(f"✗ {prog_id}: Expected {expected_chords}, got {actual_chords}")
                    return False
        
        print(f"✗ {prog_id}: Progression not found in {emotion}")
        return False
        
    except Exception as e:
        print(f"✗ {prog_id}: Error validating - {e}")
        return False

def show_emotional_improvements(database):
    """Show examples of how each emotion's progressions were improved"""
    
    print("\n=== EMOTIONAL ALIGNMENT IMPROVEMENTS ===\n")
    
    improvements = {
        "Joy": "Replaced melancholy minor chords (iii, vi) with brighter alternatives (ii, IV)",
        "Sadness": "Ensured progressions resolve to minor tonic (i) instead of major chords",
        "Trust": "Removed dissonant diminished chords, added proper resolutions for stability",
        "Love": "Used major dominants (V) instead of minor (v) for warmer resolutions",
        "Anger": "Emphasized minor and unstable chords to maintain aggressive tension",
        "Fear": "Replaced consonant major chords with minor versions for ominous atmosphere",
        "Disgust": "Maintained consistent dissonance, avoided pleasant major chord relief",
        "Anticipation": "Replaced stable chords with diminished ones to preserve suspense",
        "Shame": "Simplified augmented chords to focus on tragedy rather than grandeur",
        "Envy": "Added complexity to dominants (V7) for more twisted, bitter sound"
    }
    
    for emotion, improvement in improvements.items():
        print(f"🎭 {emotion.upper()}")
        print(f"   Improvement: {improvement}")
        
        # Show an example progression
        try:
            emotion_data = database["emotions"][emotion]
            first_progression = emotion_data["progression_pool"][0]
            chord_sequence = " → ".join(first_progression["chords"])
            print(f"   Example: {chord_sequence}")
            print()
        except:
            print()

def analyze_emotional_consistency():
    """Analyze the overall emotional consistency of the fixed database"""
    
    print("=== EMOTIONAL CONSISTENCY ANALYSIS ===\n")
    
    database_path = "/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance/emotion_progression_database.json"
    
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    for emotion_name, emotion_data in database["emotions"].items():
        mode = emotion_data["mode"]
        progressions = emotion_data["progression_pool"]
        
        print(f"🎼 {emotion_name.upper()} ({mode} mode)")
        print(f"   Total progressions: {len(progressions)}")
        
        # Analyze chord usage patterns
        all_chords = []
        for prog in progressions:
            all_chords.extend(prog["chords"])
        
        # Count unique chords
        unique_chords = list(set(all_chords))
        major_chords = [c for c in unique_chords if not any(x in c for x in ['i', '°', '♭', '#', '+']) and c != 'I']
        minor_chords = [c for c in unique_chords if 'i' in c.lower() and c != 'I']
        
        print(f"   Unique chords used: {len(unique_chords)}")
        print(f"   Major chords: {len(major_chords)} | Minor chords: {len(minor_chords)}")
        print(f"   Sample progression: {' - '.join(progressions[0]['chords'])}")
        print()

def main():
    """Main validation function"""
    
    # Validate the specific fixes
    validation_success = validate_chord_progression_fixes()
    
    # Analyze overall emotional consistency
    analyze_emotional_consistency()
    
    return validation_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
