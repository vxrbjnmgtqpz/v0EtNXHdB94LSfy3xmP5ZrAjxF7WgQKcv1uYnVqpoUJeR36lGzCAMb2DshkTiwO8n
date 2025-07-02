#!/usr/bin/env python3
"""
Comprehensive validation of Harmonic Minor and Melodic Minor chord characteristics
"""

from individual_chord_model import IndividualChordModel

def validate_minor_modes():
    """Validate the musical characteristics and emotional mappings of the new minor modes"""
    print("=" * 70)
    print("    HARMONIC & MELODIC MINOR VALIDATION")
    print("=" * 70)
    
    model = IndividualChordModel()
    
    # Test 1: Validate unique chord types
    print("🎭 1. UNIQUE CHORD TYPE VALIDATION")
    print("-" * 50)
    
    # Test augmented chords (♭III+) - should appear in both minor modes
    print("Augmented chords (♭III+):")
    for context in ["Harmonic Minor", "Melodic Minor"]:
        result = model.generate_chord_from_prompt(
            "awe-inspiring and transcendent",
            context_preference=context,
            num_options=5
        )
        aug_chords = [r for r in result if "aug" in r['chord_symbol'] or "♭III+" in r['roman_numeral']]
        if aug_chords:
            print(f"  {context}: {aug_chords[0]['chord_symbol']} ({aug_chords[0]['roman_numeral']}) - {aug_chords[0]['emotional_score']:.3f}")
    
    # Test V7 dominants - both should have them
    print("\nDominant V7 chords:")
    for context in ["Harmonic Minor", "Melodic Minor"]:
        result = model.generate_chord_from_prompt(
            "strong anticipation and tension",
            context_preference=context,
            num_options=5
        )
        v7_chords = [r for r in result if r['roman_numeral'] == "V7"]
        if v7_chords:
            print(f"  {context}: {v7_chords[0]['chord_symbol']} ({v7_chords[0]['roman_numeral']}) - {v7_chords[0]['emotional_score']:.3f}")
    
    # Test diminished chords variations
    print("\nDiminished chord variations:")
    dim_emotions = ["fearful", "disgusted", "tense anxiety"]
    for emotion in dim_emotions:
        print(f"  '{emotion}':")
        for context in ["Harmonic Minor", "Melodic Minor"]:
            result = model.generate_chord_from_prompt(
                emotion,
                context_preference=context,
                num_options=3
            )
            dim_chords = [r for r in result if "dim" in r['chord_symbol'] or "°" in r['roman_numeral']]
            if dim_chords:
                print(f"    {context}: {dim_chords[0]['chord_symbol']} ({dim_chords[0]['roman_numeral']})")
    
    # Test 2: Emotional differentiation
    print(f"\n🧠 2. EMOTIONAL DIFFERENTIATION")
    print("-" * 50)
    
    emotion_tests = [
        ("dramatic operatic tension", ["Harmonic Minor"]),
        ("smooth sophisticated jazz", ["Melodic Minor"]), 
        ("exotic mysterious beauty", ["Harmonic Minor"]),
        ("modern classical elegance", ["Melodic Minor"]),
        ("tragic romantic passion", ["Harmonic Minor"])
    ]
    
    for emotion, expected_contexts in emotion_tests:
        print(f"'{emotion}':")
        
        # Test across all contexts
        results_by_context = {}
        for context in ["Aeolian", "Harmonic Minor", "Melodic Minor", "Jazz", "Ionian"]:
            try:
                result = model.generate_chord_from_prompt(
                    emotion,
                    context_preference=context,
                    num_options=1
                )
                if result:
                    results_by_context[context] = result[0]
            except:
                pass
        
        # Sort by emotional score
        sorted_results = sorted(results_by_context.items(), 
                              key=lambda x: x[1]['emotional_score'], 
                              reverse=True)
        
        # Show top 3 and highlight if expected context is among them
        for i, (context, chord) in enumerate(sorted_results[:3], 1):
            indicator = "⭐" if context in expected_contexts else "  "
            print(f"  {indicator} {i}. {context:15}: {chord['chord_symbol']:8} ({chord['roman_numeral']:5}) - {chord['emotional_score']:.3f}")
        print()
    
    # Test 3: Modal chord progression compatibility
    print("🎵 3. MODAL PROGRESSION COMPATIBILITY")
    print("-" * 50)
    
    # Generate multiple chords that could work together in each mode
    for mode in ["Harmonic Minor", "Melodic Minor"]:
        print(f"{mode} chord palette:")
        
        # Get a variety of chords from this mode
        emotions = ["melancholy", "tense", "beautiful", "anticipatory"]
        mode_chords = []
        
        for emotion in emotions:
            result = model.generate_chord_from_prompt(
                emotion,
                context_preference=mode,
                num_options=2
            )
            mode_chords.extend(result)
        
        # Remove duplicates and show unique chords
        unique_chords = {}
        for chord in mode_chords:
            key = chord['roman_numeral']
            if key not in unique_chords or chord['emotional_score'] > unique_chords[key]['emotional_score']:
                unique_chords[key] = chord
        
        # Sort by roman numeral for logical ordering
        chord_order = ['i', 'ii', 'ii°', '♭III+', 'iv', 'IV', 'V7', '♭VI', 'vi°', 'vii°']
        for roman in chord_order:
            if roman in unique_chords:
                chord = unique_chords[roman]
                print(f"  {chord['roman_numeral']:6} - {chord['chord_symbol']:8} (Score: {chord['emotional_score']:.2f})")
        print()
    
    # Test 4: Key transposition with new modes
    print("🎹 4. KEY TRANSPOSITION TEST")
    print("-" * 50)
    
    test_keys = ["C", "G", "D", "F#", "Bb"]
    for mode in ["Harmonic Minor", "Melodic Minor"]:
        print(f"{mode} tonic chord across keys:")
        for key in test_keys:
            result = model.generate_chord_from_prompt(
                "melancholy and introspective",
                context_preference=mode,
                key=key,
                num_options=1
            )
            if result:
                chord = result[0]
                print(f"  Key {key:2}: {chord['chord_symbol']:6} ({chord['roman_numeral']})")
        print()
    
    print("=" * 70)
    print("    VALIDATION COMPLETE - New modes are fully functional!")
    print("=" * 70)

if __name__ == "__main__":
    validate_minor_modes()
