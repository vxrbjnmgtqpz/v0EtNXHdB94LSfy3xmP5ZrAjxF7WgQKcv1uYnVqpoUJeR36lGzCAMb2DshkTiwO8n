#!/usr/bin/env python3
"""
Extended test for individual chord model - edge cases and integration testing
"""

import json
from individual_chord_model import IndividualChordModel

def test_edge_cases():
    """Test edge cases and robustness of the individual chord model"""
    print("=== Individual Chord Model Edge Case Tests ===\n")
    
    model = IndividualChordModel()
    
    # Test 1: Empty and nonsensical inputs
    print("1. Edge Case Inputs:")
    edge_prompts = [
        "",  # Empty
        "asdfghjkl",  # Nonsensical
        "12345",  # Numbers only
        "The weather is nice today",  # No emotional content
        "I don't know how I feel"  # Ambiguous
    ]
    
    for prompt in edge_prompts:
        try:
            result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
            print(f"   '{prompt}' → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")
        except Exception as e:
            print(f"   '{prompt}' → ERROR: {e}")
    
    # Test 2: Different keys
    print("\n2. Transposition to Different Keys:")
    prompt = "happy and bright"
    for key in ["C", "G", "F", "D"]:
        result = model.generate_chord_from_prompt(prompt, key=key, num_options=1)[0]
        print(f"   In key {key}: {result['chord_symbol']} ({result['roman_numeral']})")
    
    # Test 3: Multiple options
    print("\n3. Multiple Chord Options:")
    result = model.generate_chord_from_prompt("melancholy jazz", num_options=5)
    for i, chord in enumerate(result, 1):
        print(f"   {i}. {chord['chord_symbol']} ({chord['roman_numeral']}) - {chord['mode_context']}/{chord['style_context']}")
        print(f"      Score: {chord['emotional_score']:.3f}")
    
    # Test 4: Context filtering
    print("\n4. Context-Specific Filtering:")
    prompt = "I feel sad"
    modes = ["Ionian", "Aeolian", "Dorian"]
    styles = ["Jazz", "Blues", "Classical"]
    
    print("   Mode filtering:")
    for mode in modes:
        try:
            result = model.generate_chord_from_prompt(prompt, mode_preference=mode, num_options=1)[0]
            print(f"   {mode}: {result['chord_symbol']} ({result['roman_numeral']})")
        except:
            print(f"   {mode}: No suitable chords found")
    
    print("   Style filtering:")
    for style in styles:
        try:
            result = model.generate_chord_from_prompt(prompt, style_preference=style, num_options=1)[0]
            print(f"   {style}: {result['chord_symbol']} ({result['roman_numeral']})")
        except:
            print(f"   {style}: No suitable chords found")
    
    # Test 5: Mixed emotions
    print("\n5. Mixed Emotional Content:")
    mixed_prompts = [
        "happy but also nervous",
        "sad but beautiful",
        "angry love song",
        "fearful anticipation",
        "disgusted but curious"
    ]
    
    for prompt in mixed_prompts:
        result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
        # Get the top 3 emotion weights
        emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:3]
        emotions_str = ", ".join([f"{e}:{w:.2f}" for e, w in emotions if w > 0])
        print(f"   '{prompt}'")
        print(f"   → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")
        print(f"     Emotions: {emotions_str}")
    
    # Test 6: Consistency check
    print("\n6. Consistency Check (same prompt multiple times):")
    prompt = "melancholy and reflective"
    results = []
    for i in range(5):
        result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
        results.append(result['chord_symbol'])
    
    print(f"   Prompt: '{prompt}'")
    print(f"   Results: {results}")
    print(f"   Consistency: {'GOOD' if len(set(results)) <= 2 else 'VARIABLE'}")

if __name__ == "__main__":
    test_edge_cases()
