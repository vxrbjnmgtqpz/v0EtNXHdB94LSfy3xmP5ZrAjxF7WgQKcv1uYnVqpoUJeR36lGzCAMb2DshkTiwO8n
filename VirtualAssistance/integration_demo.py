#!/usr/bin/env python3
"""
Integration Demo: Chord Progression + Individual Chord Models
Shows how both models can work together for comprehensive chord generation
"""

import json
from chord_progression_model import ChordProgressionModel
from individual_chord_model import IndividualChordModel

def integrated_demo():
    """Demonstrate both progression and individual chord models working together"""
    print("=" * 70)
    print("    INTEGRATED CHORD GENERATION DEMO")
    print("    Progression Model + Individual Chord Model")
    print("=" * 70)
    
    # Initialize both models
    progression_model = ChordProgressionModel()
    chord_model = IndividualChordModel()
    
    # Test scenarios
    scenarios = [
        {
            "emotion": "happy and energetic",
            "description": "Upbeat song opening"
        },
        {
            "emotion": "melancholy and reflective", 
            "description": "Contemplative ballad"
        },
        {
            "emotion": "romantic and warm",
            "description": "Love song verse"
        },
        {
            "emotion": "mysterious and dark",
            "description": "Tension building"
        },
        {
            "emotion": "playful jazz feeling",
            "description": "Jazz improvisation"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎵 SCENARIO {i}: {scenario['description']}")
        print(f"Emotion: '{scenario['emotion']}'")
        print("-" * 50)
        
        # Get chord progression
        try:
            prog_results = progression_model.generate_from_prompt(
                scenario['emotion'], num_progressions=1
            )
            prog_result = prog_results[0] if prog_results else None
            print(f"📋 PROGRESSION: {' - '.join(prog_result['chords'])}")
            print(f"   Mode: {prog_result['primary_mode']}")
            print(f"   Genre: {prog_result.get('genre', 'N/A')}")
        except Exception as e:
            print(f"📋 PROGRESSION: Error - {e}")
            prog_result = None
        
        # Get individual chord options
        try:
            chord_results = chord_model.generate_chord_from_prompt(
                scenario['emotion'], num_options=3
            )
            print(f"🎹 INDIVIDUAL CHORDS:")
            for j, chord in enumerate(chord_results, 1):
                print(f"   {j}. {chord['chord_symbol']} ({chord['roman_numeral']}) - {chord['mode_context']} - Score: {chord['emotional_score']:.3f}")
        except Exception as e:
            print(f"🎹 INDIVIDUAL CHORDS: Error - {e}")
        
        # Suggest workflow
        print(f"💡 WORKFLOW SUGGESTION:")
        if prog_result:
            print(f"   → Start with progression: {' - '.join(prog_result['chords'])}")
            print(f"   → Add individual chord embellishments from the options above")
            print(f"   → Consider the {prog_result['primary_mode']} modal context")
        else:
            print(f"   → Use individual chords as building blocks")
            print(f"   → Combine multiple chords to create a custom progression")
    
    # Demonstrate contextual awareness
    print(f"\n🎼 CONTEXTUAL COMPARISON")
    print("-" * 50)
    
    test_emotion = "sad but beautiful"
    print(f"Emotion: '{test_emotion}' across different contexts")
    
    # Progression model (mode-based)
    try:
        prog_results = progression_model.generate_from_prompt(test_emotion, num_progressions=1)
        prog = prog_results[0] if prog_results else None
        if prog:
            print(f"📋 Progression Model → Mode: {prog['primary_mode']}, Progression: {' - '.join(prog['chords'])}")
        else:
            print(f"📋 Progression Model → No results")
    except Exception as e:
        print(f"📋 Progression Model → Error: {e}")
    
    # Individual chord model (context-aware)
    contexts = ["Ionian", "Aeolian", "Jazz", "Blues"]
    print(f"🎹 Individual Chord Model:")
    for context in contexts:
        try:
            chord = chord_model.generate_chord_from_prompt(
                test_emotion, context_preference=context, num_options=1
            )[0]
            print(f"   {context:8}: {chord['chord_symbol']} ({chord['roman_numeral']})")
        except:
            print(f"   {context:8}: No suitable chord found")
    
    # JSON output comparison
    print(f"\n📊 JSON OUTPUT COMPARISON")
    print("-" * 50)
    
    emotion = "romantic jazz"
    
    try:
        prog_results = progression_model.generate_from_prompt(emotion, num_progressions=1)
        prog_json = prog_results[0] if prog_results else {}
        print("📋 Progression Model JSON structure:")
        print(f"   Keys: {list(prog_json.keys())}")
        
        chord_json = chord_model.generate_chord_from_prompt(emotion, num_options=1)[0]
        print("🎹 Individual Chord Model JSON structure:")
        print(f"   Keys: {list(chord_json.keys())}")
        
        print("\n📈 Both models provide:")
        print("   ✓ Emotion analysis and scoring")
        print("   ✓ Roman numeral notation")
        print("   ✓ Modal/contextual information")
        print("   ✓ Structured JSON output")
        print("   ✓ Generation metadata")
        
    except Exception as e:
        print(f"JSON comparison error: {e}")
    
    print(f"\n🔧 INTEGRATION POSSIBILITIES:")
    print("-" * 50)
    print("1. 📝 Unified API: Single endpoint for both progression and chord requests")
    print("2. 🎛️  Hybrid Mode: Generate progression + suggest chord substitutions")
    print("3. 🔄 Interactive: Start with progression, refine with individual chords")
    print("4. 🎨 Creative: Use individual chords to create custom progressions")
    print("5. 🎵 Real-time: Live chord suggestion based on current progression context")
    
    print("\n" + "=" * 70)
    print("    INTEGRATION DEMO COMPLETE")
    print("    Both models are functional and complementary!")
    print("=" * 70)

if __name__ == "__main__":
    integrated_demo()
