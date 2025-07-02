#!/usr/bin/env python3
"""
Test the newly added Hungarian Minor and Locrian contexts
"""

from individual_chord_model import IndividualChordModel

def test_extreme_modes():
    """Test Hungarian Minor and Locrian - the most exotic and emotionally extreme modes"""
    print("=" * 80)
    print("    HUNGARIAN MINOR & LOCRIAN - EXTREME MODES TESTING")
    print("=" * 80)
    
    model = IndividualChordModel()
    
    # Check all available contexts
    contexts = model.get_available_contexts()
    print(f"📋 All Available Contexts ({len(contexts)}): {', '.join(contexts)}")
    print()
    
    # Test 1: Hungarian Minor - Exotic and Dramatic
    print("🎭 1. HUNGARIAN MINOR - EXOTIC DRAMA")
    print("-" * 60)
    
    hungarian_prompts = [
        "exotic passionate romance",
        "gypsy dramatic intensity", 
        "mysterious Eastern European beauty",
        "tragic melodramatic love",
        "Romani folk intensity",
        "theatrical operatic passion"
    ]
    
    for prompt in hungarian_prompts:
        results = model.generate_chord_from_prompt(
            prompt, 
            context_preference="Hungarian Minor", 
            num_options=3
        )
        print(f"'{prompt}':")
        for i, chord in enumerate(results, 1):
            emotions = sorted(chord['emotion_weights'].items(), key=lambda x: x[1], reverse=True)
            top_emotion = emotions[0] if emotions[0][1] > 0 else ("Joy", 0.0)
            print(f"  {i}. {chord['chord_symbol']:8} ({chord['roman_numeral']:6}) - {top_emotion[0]}({top_emotion[1]:.2f}) - Score: {chord['emotional_score']:.3f}")
        print()
    
    # Test 2: Locrian - Chaos and Instability
    print("🌪️  2. LOCRIAN - CHAOS AND INSTABILITY")
    print("-" * 60)
    
    locrian_prompts = [
        "anxious unstable fear",
        "collapsing chaotic despair",
        "nervous breakdown tension",
        "existential dread and void",
        "unstable reality crumbling",
        "psychotic dissonance"
    ]
    
    for prompt in locrian_prompts:
        results = model.generate_chord_from_prompt(
            prompt,
            context_preference="Locrian",
            num_options=3
        )
        print(f"'{prompt}':")
        for i, chord in enumerate(results, 1):
            emotions = sorted(chord['emotion_weights'].items(), key=lambda x: x[1], reverse=True)
            top_emotion = emotions[0] if emotions[0][1] > 0 else ("Joy", 0.0)
            print(f"  {i}. {chord['chord_symbol']:8} ({chord['roman_numeral']:6}) - {top_emotion[0]}({top_emotion[1]:.2f}) - Score: {chord['emotional_score']:.3f}")
        print()
    
    # Test 3: Cross-Modal Comparison - Same emotion, different modes
    print("🔄 3. CROSS-MODAL EMOTIONAL COMPARISON")
    print("-" * 60)
    
    test_emotions = [
        "deep sadness",
        "intense fear", 
        "beautiful awe",
        "dramatic tension"
    ]
    
    extreme_modes = ["Hungarian Minor", "Locrian", "Harmonic Minor", "Phrygian"]
    
    for emotion in test_emotions:
        print(f"'{emotion}' across extreme modes:")
        
        results_by_mode = {}
        for mode in extreme_modes:
            try:
                result = model.generate_chord_from_prompt(
                    emotion,
                    context_preference=mode,
                    num_options=1
                )
                if result:
                    results_by_mode[mode] = result[0]
            except:
                continue
        
        # Sort by emotional score
        sorted_results = sorted(results_by_mode.items(), 
                              key=lambda x: x[1]['emotional_score'], 
                              reverse=True)
        
        for i, (mode, chord) in enumerate(sorted_results, 1):
            print(f"  {i}. {mode:15}: {chord['chord_symbol']:8} ({chord['roman_numeral']:6}) - Score: {chord['emotional_score']:.3f}")
        print()
    
    # Test 4: Unique Chord Characteristics
    print("🎼 4. UNIQUE CHORD CHARACTERISTICS")
    print("-" * 60)
    
    # Test Hungarian Minor's augmented chord
    print("Hungarian Minor ♭III+ (augmented) - Maximum Aesthetic Awe:")
    aug_result = model.generate_chord_from_prompt(
        "transcendent exotic beauty",
        context_preference="Hungarian Minor",
        num_options=5
    )
    aug_chords = [r for r in aug_result if "aug" in r['chord_symbol'] or "♭III+" in r['roman_numeral']]
    if aug_chords:
        chord = aug_chords[0]
        print(f"  {chord['chord_symbol']} ({chord['roman_numeral']}) - Aesthetic Awe: {chord['emotion_weights']['Aesthetic Awe']:.2f}")
    
    # Test Locrian's unstable tonic
    print("\nLocrian i° (diminished tonic) - Maximum Instability:")
    dim_result = model.generate_chord_from_prompt(
        "complete instability and fear",
        context_preference="Locrian",
        num_options=5
    )
    dim_chords = [r for r in dim_result if r['roman_numeral'] == "i°"]
    if dim_chords:
        chord = dim_chords[0]
        print(f"  {chord['chord_symbol']} ({chord['roman_numeral']}) - Fear: {chord['emotion_weights']['Fear']:.2f}")
    
    # Test Locrian's ♭II - Anger/Aggression
    print("\nLocrian ♭II (Neapolitan) - Maximum Aggression:")
    neap_result = model.generate_chord_from_prompt(
        "aggressive angry confrontation",
        context_preference="Locrian",
        num_options=5
    )
    neap_chords = [r for r in neap_result if r['roman_numeral'] == "♭II"]
    if neap_chords:
        chord = neap_chords[0]
        print(f"  {chord['chord_symbol']} ({chord['roman_numeral']}) - Anger: {chord['emotion_weights']['Anger']:.2f}")
    
    # Test 5: Emotional Extremes
    print(f"\n🌡️  5. EMOTIONAL EXTREMES TEST")
    print("-" * 60)
    
    # Find the most extreme emotions across all modes
    all_modes = contexts
    extreme_tests = {
        "Maximum Fear": "terrifying horror nightmare",
        "Maximum Aesthetic Awe": "transcendent divine beauty", 
        "Maximum Sadness": "devastating tragic loss",
        "Maximum Anger": "furious rage violence"
    }
    
    for test_name, prompt in extreme_tests.items():
        print(f"{test_name}:")
        
        best_result = None
        best_score = 0
        best_mode = ""
        target_emotion = test_name.split()[1]  # Extract emotion name
        
        for mode in all_modes:
            try:
                result = model.generate_chord_from_prompt(
                    prompt,
                    context_preference=mode,
                    num_options=1
                )
                if result and result[0]['emotion_weights'].get(target_emotion, 0) > best_score:
                    best_score = result[0]['emotion_weights'][target_emotion]
                    best_result = result[0]
                    best_mode = mode
            except:
                continue
        
        if best_result:
            print(f"  Winner: {best_mode} - {best_result['chord_symbol']} ({best_result['roman_numeral']}) - {target_emotion}: {best_score:.2f}")
        print()
    
    # Test 6: Modal Progression Potential
    print("🎵 6. MODAL PROGRESSION BUILDING")
    print("-" * 60)
    
    for mode in ["Hungarian Minor", "Locrian"]:
        print(f"{mode} chord palette:")
        
        # Get a variety of chords from this mode
        emotions = ["melancholy", "tense", "beautiful", "dramatic", "fearful"]
        mode_chords = []
        
        for emotion in emotions:
            try:
                result = model.generate_chord_from_prompt(
                    emotion,
                    context_preference=mode,
                    num_options=3
                )
                mode_chords.extend(result)
            except:
                continue
        
        # Remove duplicates and show unique chords
        unique_chords = {}
        for chord in mode_chords:
            key = chord['roman_numeral']
            if key not in unique_chords or chord['emotional_score'] > unique_chords[key]['emotional_score']:
                unique_chords[key] = chord
        
        # Show all available chords in this mode
        for roman, chord in sorted(unique_chords.items()):
            print(f"  {chord['roman_numeral']:6} - {chord['chord_symbol']:8} (Score: {chord['emotional_score']:.2f})")
        print()
    
    print("=" * 80)
    print("    EXTREME MODES TESTING COMPLETE!")
    print("    The full spectrum of Western harmony is now available!")
    print("=" * 80)

if __name__ == "__main__":
    test_extreme_modes()
