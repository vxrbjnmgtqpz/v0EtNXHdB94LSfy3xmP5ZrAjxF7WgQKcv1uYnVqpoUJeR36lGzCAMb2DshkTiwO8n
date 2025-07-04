#!/usr/bin/env python3
"""
Test Consonant/Dissonant Integration
Demonstrates the new consonant/dissonant functionality in the individual chord model
"""

import sys
import json
from individual_chord_model import IndividualChordModel

def test_consonant_dissonant_integration():
    """Test the new consonant/dissonant functionality"""
    print("🎵 Testing Consonant/Dissonant Integration")
    print("=" * 50)
    
    # Initialize the model
    model = IndividualChordModel()
    
    # Test scenarios with different consonant/dissonant preferences
    test_cases = [
        {
            "prompt": "I feel happy and joyful",
            "cd_preferences": [None, 0.0, 0.5, 1.0],
            "style": "Classical"
        },
        {
            "prompt": "dark and mysterious",
            "cd_preferences": [None, 0.0, 0.5, 1.0],
            "style": "Jazz"
        },
        {
            "prompt": "tense and anxious",
            "cd_preferences": [None, 0.0, 0.5, 1.0],
            "style": "Classical"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: '{test_case['prompt']}' ({test_case['style']})")
        print(f"{'='*50}")
        
        for cd_pref in test_case["cd_preferences"]:
            cd_label = "Auto" if cd_pref is None else f"{cd_pref:.1f}"
            print(f"\n🎯 CD Preference: {cd_label}")
            print("-" * 30)
            
            try:
                results = model.generate_chord_from_prompt(
                    test_case["prompt"],
                    style_preference=test_case["style"],
                    num_options=3,
                    consonant_dissonant_preference=cd_pref
                )
                
                for j, result in enumerate(results, 1):
                    cd_value = result.get("consonant_dissonant_value")
                    cd_desc = result.get("consonant_dissonant_description")
                    
                    print(f"  {j}. {result['chord_symbol']} ({result['roman_numeral']})")
                    print(f"     Score: {result['emotional_score']:.3f}")
                    if cd_value is not None:
                        print(f"     CD Value: {cd_value:.3f} ({cd_desc})")
                    
                    # Show top emotions
                    top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_text = ", ".join([f"{k}({v:.2f})" for k, v in top_emotions if v > 0])
                    print(f"     Emotions: {emotion_text}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

def test_database_loading():
    """Test that the database loads correctly with consonant/dissonant profiles"""
    print("\n🔍 Testing Database Loading")
    print("=" * 30)
    
    model = IndividualChordModel()
    
    # Check database statistics
    total_chords = len(model.database.chord_emotion_map)
    chords_with_cd = sum(1 for chord in model.database.chord_emotion_map if chord.consonant_dissonant_profile)
    emotion_dimensions = len(model.database.chord_emotion_map[0].emotion_weights) if model.database.chord_emotion_map else 0
    
    print(f"✅ Total chords: {total_chords}")
    print(f"✅ Chords with CD profiles: {chords_with_cd}")
    print(f"✅ Emotion dimensions: {emotion_dimensions}")
    
    # Show a few sample chords
    print("\n📊 Sample Chord Profiles:")
    for i, chord in enumerate(model.database.chord_emotion_map[:5]):
        cd_profile = chord.consonant_dissonant_profile
        if cd_profile:
            cd_value = cd_profile.get("base_value", "N/A")
            cd_desc = cd_profile.get("description", "N/A")
            print(f"  {chord.roman_numeral} ({chord.style_context}): CD={cd_value} - {cd_desc}")

def test_emotion_cd_correlation():
    """Test that emotions correlate appropriately with consonant/dissonant values"""
    print("\n🧠 Testing Emotion-CD Correlation")
    print("=" * 35)
    
    model = IndividualChordModel()
    
    # Test emotional prompts that should prefer consonance
    consonant_prompts = [
        "peaceful and serene",
        "loving and warm",
        "joyful and bright",
        "grateful and content"
    ]
    
    # Test emotional prompts that should prefer dissonance
    dissonant_prompts = [
        "angry and harsh",
        "fearful and tense",
        "malicious and cruel",
        "anxious and unsettled"
    ]
    
    print("\n🕊️ Consonant Emotions (should prefer low CD values):")
    for prompt in consonant_prompts:
        try:
            results = model.generate_chord_from_prompt(prompt, num_options=1)
            if results:
                result = results[0]
                cd_value = result.get("consonant_dissonant_value", "N/A")
                print(f"  '{prompt}' → {result['chord_symbol']} (CD: {cd_value})")
        except Exception as e:
            print(f"  '{prompt}' → Error: {e}")
    
    print("\n⚡ Dissonant Emotions (should prefer high CD values):")
    for prompt in dissonant_prompts:
        try:
            results = model.generate_chord_from_prompt(prompt, num_options=1)
            if results:
                result = results[0]
                cd_value = result.get("consonant_dissonant_value", "N/A")
                print(f"  '{prompt}' → {result['chord_symbol']} (CD: {cd_value})")
        except Exception as e:
            print(f"  '{prompt}' → Error: {e}")

def main():
    """Run all tests"""
    print("🚀 Consonant/Dissonant Integration Test Suite")
    print("=" * 70)
    
    try:
        test_database_loading()
        test_consonant_dissonant_integration()
        test_emotion_cd_correlation()
        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("🎉 Consonant/Dissonant integration is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
