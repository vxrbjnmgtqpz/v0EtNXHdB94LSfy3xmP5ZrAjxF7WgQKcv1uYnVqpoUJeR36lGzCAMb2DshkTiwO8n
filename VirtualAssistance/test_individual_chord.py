#!/usr/bin/env python3
"""
Test script for individual chord model functionality
"""

import json
from individual_chord_model import IndividualChordModel

def test_chord_generation():
    """Test various aspects of the individual chord model"""
    print("=== Individual Chord Model Test Suite ===\n")
    
    model = IndividualChordModel()
    
    # Test 1: Basic emotion-to-chord mapping
    print("1. Basic Emotion-to-Chord Mapping:")
    basic_prompts = [
        "I'm feeling happy",
        "I'm sad and lonely", 
        "I feel angry",
        "I'm in love"
    ]
    
    for prompt in basic_prompts:
        result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
        print(f"   '{prompt}' → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")
    
    # Test 2: Context-specific generation
    print("\n2. Context-Specific Generation:")
    contexts = model.get_available_contexts()
    print(f"   Available modes: {contexts['modes']}")
    print(f"   Available styles: {contexts['styles']}")
    
    prompt = "I feel mysterious and dark"
    for mode in ["Aeolian", "Dorian"]:
        if mode in contexts['modes']:
            result = model.generate_chord_from_prompt(prompt, mode_preference=mode, num_options=1)[0]
            print(f"   '{prompt}' in {mode} mode → {result['chord_symbol']} ({result['roman_numeral']})")
    
    for style in ["Jazz", "Blues"]:
        if style in contexts['styles']:
            result = model.generate_chord_from_prompt(prompt, style_preference=style, num_options=1)[0]
            print(f"   '{prompt}' in {style} style → {result['chord_symbol']} ({result['roman_numeral']})")
    
    # Test 3: JSON output format
    print("\n3. JSON Output Format:")
    result = model.generate_chord_from_prompt("joyful and bright", num_options=2)
    print(json.dumps(result, indent=2))
    
    # Test 4: Emotional analysis
    print("\n4. Emotional Analysis:")
    analysis = model.analyze_emotional_content("I feel bittersweet - happy but also nostalgic")
    print(f"   Dominant emotions: {analysis['dominant_emotions']}")
    print(f"   Primary emotion: {analysis['primary_emotion']}")
    print(f"   Emotional complexity: {analysis['emotional_complexity']}")
    
    # Test 5: Complex prompts
    print("\n5. Complex Prompts:")
    complex_prompts = [
        "bluesy melancholy with a hint of hope",
        "jazz sophistication meets romantic warmth",
        "mysterious dark tension building to anticipation",
        "playful joy with unexpected harmonic twists"
    ]
    
    for prompt in complex_prompts:
        result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
        print(f"   '{prompt}'")
        print(f"   → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")
        print(f"     Score: {result['emotional_score']:.3f}")

if __name__ == "__main__":
    test_chord_generation()
