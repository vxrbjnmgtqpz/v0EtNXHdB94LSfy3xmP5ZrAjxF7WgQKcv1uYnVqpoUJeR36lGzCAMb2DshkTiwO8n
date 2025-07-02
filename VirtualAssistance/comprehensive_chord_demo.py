#!/usr/bin/env python3
"""
Comprehensive Demo of the Individual Chord Model
Showcases all major features and capabilities
"""

import json
from individual_chord_model import IndividualChordModel

def comprehensive_demo():
    """Complete demonstration of individual chord model capabilities"""
    print("=" * 60)
    print("    INDIVIDUAL CHORD MODEL - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    model = IndividualChordModel()
    
    # 1. Basic Emotion-to-Chord Mapping
    print("\n🎵 1. BASIC EMOTION-TO-CHORD MAPPING")
    print("-" * 40)
    basic_emotions = [
        "I feel joyful and energetic",
        "I'm deeply sad and melancholy", 
        "I feel romantic and tender",
        "I'm anxious and tense",
        "I feel angry and aggressive",
        "I'm in awe of something beautiful"
    ]
    
    for emotion in basic_emotions:
        result = model.generate_chord_from_prompt(emotion, num_options=1)[0]
        print(f"'{emotion}'")
        print(f"  → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']} ({result['style_context']})")
        print(f"    Score: {result['emotional_score']:.3f}")
    
    # 2. Context-Aware Generation
    print("\n🎼 2. CONTEXT-AWARE GENERATION")
    print("-" * 40)
    contexts = model.get_available_contexts()
    print(f"Available modes: {', '.join(contexts['modes'])}")
    print(f"Available styles: {', '.join(contexts['styles'])}")
    
    prompt = "I feel melancholy but sophisticated"
    print(f"\nPrompt: '{prompt}'")
    print("Mode-specific results:")
    for mode in ["Ionian", "Aeolian", "Dorian"]:
        if mode in contexts['modes']:
            result = model.generate_chord_from_prompt(prompt, mode_preference=mode, num_options=1)[0]
            print(f"  {mode:8}: {result['chord_symbol']:8} ({result['roman_numeral']:5}) - Score: {result['emotional_score']:.3f}")
    
    print("Style-specific results:")
    for style in ["Jazz", "Blues", "Classical"]:
        if style in contexts['styles']:
            result = model.generate_chord_from_prompt(prompt, style_preference=style, num_options=1)[0]
            print(f"  {style:8}: {result['chord_symbol']:8} ({result['roman_numeral']:5}) - Score: {result['emotional_score']:.3f}")
    
    # 3. Multi-Key Transposition
    print("\n🎹 3. MULTI-KEY TRANSPOSITION")
    print("-" * 40)
    keys = ["C", "G", "D", "A", "F", "Bb"]
    prompt = "happy and bright"
    print(f"Prompt: '{prompt}' across different keys:")
    
    for key in keys:
        result = model.generate_chord_from_prompt(prompt, key=key, num_options=1)[0]
        print(f"  Key {key:2}: {result['chord_symbol']:6} ({result['roman_numeral']:3}) - {result['mode_context']}/{result['style_context']}")
    
    # 4. Complex Emotional Prompts
    print("\n🎭 4. COMPLEX EMOTIONAL PROMPTS")
    print("-" * 40)
    complex_prompts = [
        "bittersweet nostalgia with jazz sophistication",
        "dark mysterious tension building to hopeful resolution", 
        "playful happiness with unexpected harmonic color",
        "romantic warmth tinged with melancholy",
        "bluesy sadness with a hint of defiant strength"
    ]
    
    for prompt in complex_prompts:
        result = model.generate_chord_from_prompt(prompt, num_options=1)[0]
        
        # Get top emotions
        emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)
        top_emotions = [f"{e}({w:.2f})" for e, w in emotions[:3] if w > 0.05]
        
        print(f"'{prompt}'")
        print(f"  → {result['chord_symbol']} ({result['roman_numeral']}) - {result['mode_context']}/{result['style_context']}")
        print(f"    Emotions: {', '.join(top_emotions)}")
        print(f"    Score: {result['emotional_score']:.3f}")
    
    # 5. Multiple Chord Options
    print("\n🎶 5. MULTIPLE CHORD OPTIONS")
    print("-" * 40)
    prompt = "contemplative and introspective"
    results = model.generate_chord_from_prompt(prompt, num_options=5)
    
    print(f"Prompt: '{prompt}' - Top 5 options:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['chord_symbol']:8} ({result['roman_numeral']:6}) - {result['mode_context']}/{result['style_context']:8} - Score: {result['emotional_score']:.3f}")
    
    # 6. JSON Output Format
    print("\n📋 6. STRUCTURED JSON OUTPUT")
    print("-" * 40)
    result = model.generate_chord_from_prompt("mysterious jazz feeling", num_options=1)[0]
    
    # Pretty print select fields
    print("Sample JSON output structure:")
    output = {
        "chord_id": result["chord_id"],
        "prompt": result["prompt"], 
        "chord_symbol": result["chord_symbol"],
        "roman_numeral": result["roman_numeral"],
        "mode_context": result["mode_context"],
        "style_context": result["style_context"],
        "emotional_score": result["emotional_score"],
        "key": result["key"],
        "generation_timestamp": result["metadata"]["timestamp"]
    }
    print(json.dumps(output, indent=2))
    
    # 7. Emotional Analysis
    print("\n🧠 7. EMOTIONAL ANALYSIS")
    print("-" * 40)
    analysis_prompts = [
        "I feel happy but also nervous about the future",
        "melancholy beauty with a touch of hope",
        "intense passion mixed with uncertainty"
    ]
    
    for prompt in analysis_prompts:
        analysis = model.analyze_emotional_content(prompt)
        print(f"'{prompt}'")
        print(f"  Primary: {analysis['primary_emotion'][0]} ({analysis['primary_emotion'][1]:.2f})")
        print(f"  Dominant: {', '.join([f'{e}({w:.2f})' for e, w in analysis['dominant_emotions']])}")
        print(f"  Complexity: {analysis['emotional_complexity']} emotions detected")
    
    # 8. Robustness Test
    print("\n🛡️  8. ROBUSTNESS TEST")
    print("-" * 40)
    edge_cases = [
        "",  # Empty input
        "xyz123",  # No emotional content
        "The sky is blue and grass is green",  # Neutral content
        "I feel everything and nothing at once"  # Paradoxical
    ]
    
    print("Testing edge cases (should gracefully handle):")
    for case in edge_cases:
        try:
            result = model.generate_chord_from_prompt(case, num_options=1)[0]
            print(f"  '{case}' → {result['chord_symbol']} ({result['roman_numeral']}) ✓")
        except Exception as e:
            print(f"  '{case}' → ERROR: {e} ✗")
    
    print("\n" + "=" * 60)
    print("    DEMO COMPLETE - Individual Chord Model Functioning!")
    print("=" * 60)

if __name__ == "__main__":
    comprehensive_demo()
