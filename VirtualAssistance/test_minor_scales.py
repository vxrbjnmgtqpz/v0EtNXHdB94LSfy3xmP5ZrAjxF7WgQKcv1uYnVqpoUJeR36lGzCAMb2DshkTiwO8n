#!/usr/bin/env python3
"""
Test the expanded individual chord model with Harmonic Minor and Melodic Minor contexts
"""

from individual_chord_model import IndividualChordModel

def test_minor_scales():
    """Test the new Harmonic Minor and Melodic Minor chord contexts"""
    print("=" * 70)
    print("    HARMONIC MINOR & MELODIC MINOR CHORD TESTING")
    print("=" * 70)
    
    model = IndividualChordModel()
    
    # Check available contexts
    contexts = model.get_available_contexts()
    print(f"📋 Available contexts: {', '.join(contexts)}")
    print()
    
    # Test 1: Harmonic Minor Context
    print("🎭 1. HARMONIC MINOR CONTEXT")
    print("-" * 40)
    
    harmonic_prompts = [
        "dramatic and tragic",
        "operatic tension",
        "exotic and mysterious", 
        "classical drama",
        "dark and foreboding"
    ]
    
    for prompt in harmonic_prompts:
        result = model.generate_chord_from_prompt(
            prompt, 
            context_preference="Harmonic Minor", 
            num_options=3
        )
        print(f"'{prompt}':")
        for i, chord in enumerate(result, 1):
            print(f"  {i}. {chord['chord_symbol']} ({chord['roman_numeral']}) - Score: {chord['emotional_score']:.3f}")
        print()
    
    # Test 2: Melodic Minor Context
    print("🎵 2. MELODIC MINOR CONTEXT")
    print("-" * 40)
    
    melodic_prompts = [
        "smooth and sophisticated",
        "modern jazz feeling",
        "contemplative beauty",
        "ascending hope",
        "elegant melancholy"
    ]
    
    for prompt in melodic_prompts:
        result = model.generate_chord_from_prompt(
            prompt,
            context_preference="Melodic Minor",
            num_options=3
        )
        print(f"'{prompt}':")
        for i, chord in enumerate(result, 1):
            print(f"  {i}. {chord['chord_symbol']} ({chord['roman_numeral']}) - Score: {chord['emotional_score']:.3f}")
        print()
    
    # Test 3: Comparison across minor contexts
    print("🔄 3. MINOR CONTEXT COMPARISON")
    print("-" * 40)
    
    test_emotion = "deep sadness with dramatic tension"
    minor_contexts = ["Aeolian", "Harmonic Minor", "Melodic Minor"]
    
    print(f"Emotion: '{test_emotion}' across minor contexts:")
    for context in minor_contexts:
        result = model.generate_chord_from_prompt(
            test_emotion,
            context_preference=context,
            num_options=1
        )
        if result:
            chord = result[0]
            print(f"  {context:15}: {chord['chord_symbol']:8} ({chord['roman_numeral']:5}) - Score: {chord['emotional_score']:.3f}")
    
    # Test 4: Unique chord characteristics
    print("\n🎼 4. UNIQUE CHORD CHARACTERISTICS")
    print("-" * 40)
    
    # Test augmented chords (♭III+)
    aug_result = model.generate_chord_from_prompt(
        "awe-inspiring and mysterious",
        context_preference="Harmonic Minor",
        num_options=1
    )
    if aug_result:
        print(f"Augmented chord test: {aug_result[0]['chord_symbol']} ({aug_result[0]['roman_numeral']})")
    
    # Test diminished variations
    dim_contexts = ["Harmonic Minor", "Melodic Minor"]
    for context in dim_contexts:
        dim_result = model.generate_chord_from_prompt(
            "fearful and tense",
            context_preference=context,
            num_options=1
        )
        if dim_result:
            print(f"{context} diminished: {dim_result[0]['chord_symbol']} ({dim_result[0]['roman_numeral']})")
    
    # Test V7 dominants
    print("\nDominant V7 in minor contexts:")
    for context in ["Harmonic Minor", "Melodic Minor"]:
        dom_result = model.generate_chord_from_prompt(
            "building anticipation",
            context_preference=context,
            num_options=1
        )
        if dom_result:
            print(f"  {context}: {dom_result[0]['chord_symbol']} ({dom_result[0]['roman_numeral']})")
    
    # Test 5: Cross-context emotional mapping
    print(f"\n🧠 5. CROSS-CONTEXT EMOTIONAL ANALYSIS")
    print("-" * 40)
    
    complex_emotions = [
        "tragic beauty",
        "sophisticated melancholy", 
        "operatic passion",
        "modern classical feeling"
    ]
    
    for emotion in complex_emotions:
        print(f"'{emotion}':")
        # Try all contexts and see which gives the best match
        best_results = []
        for context in contexts:
            try:
                result = model.generate_chord_from_prompt(
                    emotion,
                    context_preference=context,
                    num_options=1
                )
                if result:
                    best_results.append((context, result[0]))
            except:
                continue
        
        # Sort by emotional score
        best_results.sort(key=lambda x: x[1]['emotional_score'], reverse=True)
        
        # Show top 3 matches
        for i, (context, chord) in enumerate(best_results[:3], 1):
            print(f"  {i}. {context:15}: {chord['chord_symbol']:8} ({chord['roman_numeral']:5}) - {chord['emotional_score']:.3f}")
        print()
    
    print("=" * 70)
    print("    HARMONIC MINOR & MELODIC MINOR TESTING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    test_minor_scales()
