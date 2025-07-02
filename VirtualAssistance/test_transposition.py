#!/usr/bin/env python3
"""
Test transposition functionality specifically
"""

from individual_chord_model import IndividualChordModel

def test_transposition():
    """Test the enhanced transposition functionality"""
    print("=== Transposition Test ===\n")
    
    model = IndividualChordModel()
    
    # Test different keys with complex chords
    print("1. Complex Chord Transposition:")
    test_keys = ["C", "G", "D", "A", "E", "F#", "F", "Bb", "Eb"]
    
    for key in test_keys:
        # Test with jazz chord that should transpose
        result = model.generate_chord_from_prompt("sophisticated jazz feeling", key=key, 
                                                context_preference="Jazz", num_options=1)[0]
        print(f"   Key {key}: {result['chord_symbol']} ({result['roman_numeral']})")
    
    print("\n2. Different Emotions in Different Keys:")
    emotions_and_keys = [
        ("happy and bright", "G"),
        ("melancholy", "D"),
        ("romantic jazz", "F"),
        ("dark and mysterious", "A"),
        ("bluesy sadness", "E")
    ]
    
    for emotion, key in emotions_and_keys:
        result = model.generate_chord_from_prompt(emotion, key=key, num_options=1)[0]
        print(f"   '{emotion}' in {key}: {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")

if __name__ == "__main__":
    test_transposition()
