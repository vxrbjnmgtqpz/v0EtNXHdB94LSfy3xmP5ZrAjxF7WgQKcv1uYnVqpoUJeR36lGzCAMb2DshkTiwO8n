#!/usr/bin/env python3
"""
Full Integration Demo: Complete Music Theory System
Demonstrates the complete architecture for contextual chord-emotion translation:
1. Individual chord model - maps emotions to single chords
2. Chord progression model - maps emotions to progressions  
3. Neural progression analyzer - provides contextual weighting and novel generation

This shows how individual chord feelings and progression feelings are integrated
to provide better emotional translation of both known and novel progressions.
"""

import json
from individual_chord_model import IndividualChordModel
from chord_progression_model import ChordProgressionModel
from neural_progression_analyzer import ContextualProgressionIntegrator

def compare_models_demo():
    """Compare outputs from all three models for the same emotional prompt"""
    print("=" * 80)
    print("    FULL INTEGRATION DEMO: CONTEXTUAL CHORD-EMOTION TRANSLATION")
    print("=" * 80)
    
    # Initialize all models
    individual_model = IndividualChordModel()
    progression_model = ChordProgressionModel()
    integrator = ContextualProgressionIntegrator()
    
    # Load trained neural analyzer if available
    try:
        integrator.load_model('trained_neural_analyzer.pth')
        print("✅ Loaded trained neural analyzer")
    except:
        print("⚠️  Using untrained neural analyzer (will show random patterns)")
    
    print()
    
    # Test scenarios that demonstrate the value of integration
    scenarios = [
        {
            "emotion": "melancholy but hopeful",
            "description": "Complex mixed emotion - shows how context affects individual chords"
        },
        {
            "emotion": "dark mysterious tension",
            "description": "Progression-dependent emotion - individual chords change meaning in context"
        },
        {
            "emotion": "joyful celebration",
            "description": "Clear positive emotion - shows consistency across models"
        },
        {
            "emotion": "bittersweet nostalgia with jazz sophistication",
            "description": "Multi-faceted emotion - demonstrates neural extrapolation"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🎵 SCENARIO {i}: {scenario['description']}")
        print(f"Emotion: '{scenario['emotion']}'")
        print("=" * 60)
        
        # 1. Individual Chord Model
        print("\n1️⃣ INDIVIDUAL CHORD MODEL (Base Emotions)")
        print("-" * 40)
        try:
            individual_results = individual_model.generate_chord_from_prompt(
                scenario['emotion'], num_options=3
            )
            for j, chord in enumerate(individual_results, 1):
                emotions = sorted(chord['emotion_weights'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
                emotion_str = ', '.join([f"{e}({w:.2f})" for e, w in emotions])
                print(f"   {j}. {chord['chord_symbol']:6} ({chord['roman_numeral']:4}) - {emotion_str}")
                print(f"      Context: {chord['mode_context']}/{chord['style_context']} - Score: {chord['emotional_score']:.3f}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Chord Progression Model
        print("\n2️⃣ CHORD PROGRESSION MODEL (Progression-Level Emotions)")
        print("-" * 40)
        try:
            progression_results = progression_model.generate_from_prompt(
                scenario['emotion'], num_progressions=2
            )
            for j, prog in enumerate(progression_results, 1):
                print(f"   {j}. {' → '.join(prog['chords'])}")
                print(f"      Mode: {prog['primary_mode']}, Genre: {prog.get('genre', 'N/A')}")
                if 'emotion_weights' in prog:
                    emotions = sorted(prog['emotion_weights'].items(),
                                    key=lambda x: x[1], reverse=True)[:3]
                    emotion_str = ', '.join([f"{e}({w:.2f})" for e, w in emotions])
                    print(f"      Emotions: {emotion_str}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 3. Neural Integration - Contextual Analysis
        print("\n3️⃣ NEURAL INTEGRATION (Contextual Weighting)")
        print("-" * 40)
        
        # Use the first progression if available, or create a test progression
        test_progression = None
        if 'progression_results' in locals() and progression_results:
            test_progression = progression_results[0]['chords']
        else:
            # Fall back to a common progression
            test_progression = ['I', 'vi', 'IV', 'V']
        
        print(f"   Analyzing progression: {' → '.join(test_progression)}")
        
        try:
            analysis = integrator.analyze_progression_context(test_progression)
            
            # Show overall progression emotions
            top_emotions = sorted(analysis.overall_emotion_weights.items(),
                                key=lambda x: x[1], reverse=True)[:3]
            print(f"   Overall emotions: {', '.join([f'{e}({w:.3f})' for e, w in top_emotions])}")
            print(f"   Novel pattern score: {analysis.novel_pattern_score:.3f}")
            print(f"   Generation confidence: {analysis.generation_confidence:.3f}")
            
            # Show how individual chords are reweighted in context
            print(f"   Contextual chord analysis:")
            for chord_analysis in analysis.contextual_chord_analyses:
                print(f"     {chord_analysis.roman_numeral}: {chord_analysis.functional_role}, "
                      f"tension {chord_analysis.harmonic_tension:.2f}, "
                      f"weight {chord_analysis.contextual_weight:.2f}")
                
                # Compare base vs contextual emotions
                base_top = sorted(chord_analysis.base_emotions.items(),
                                key=lambda x: x[1], reverse=True)[0]
                context_top = sorted(chord_analysis.contextual_emotions.items(),
                                   key=lambda x: x[1], reverse=True)[0]
                print(f"       Base: {base_top[0]}({base_top[1]:.2f}) → Context: {context_top[0]}({context_top[1]:.2f})")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        # 4. Novel Progression Generation
        print("\n4️⃣ NOVEL PROGRESSION GENERATION")
        print("-" * 40)
        try:
            novel_analysis = integrator.generate_novel_progression(scenario['emotion'], length=5)
            print(f"   Generated: {' → '.join(novel_analysis.chords)}")
            
            top_emotions = sorted(novel_analysis.overall_emotion_weights.items(),
                                key=lambda x: x[1], reverse=True)[:3]
            print(f"   Emotions: {', '.join([f'{e}({w:.3f})' for e, w in top_emotions])}")
            print(f"   Novelty: {novel_analysis.novel_pattern_score:.3f} (higher = more novel)")
            print(f"   Confidence: {novel_analysis.generation_confidence:.3f}")
            
            # Show harmonic flow
            print(f"   Harmonic flow: {' → '.join([f'{t:.2f}' for t in novel_analysis.harmonic_flow])}")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n💡 INTEGRATION INSIGHTS:")
        print(f"   → Individual model provides base chord-emotion mappings")
        print(f"   → Progression model provides sequence-level emotion context")
        print(f"   → Neural integration contextually reweights individual chords")
        print(f"   → System can extrapolate to novel progressions not in database")
        print()

def workflow_demo():
    """Demonstrate a practical workflow using the integrated system"""
    print("\n" + "=" * 80)
    print("    PRACTICAL WORKFLOW DEMO")
    print("=" * 80)
    
    integrator = ContextualProgressionIntegrator()
    
    # Load trained model if available
    try:
        integrator.load_model('trained_neural_analyzer.pth')
    except:
        pass
    
    # Example workflow: Create music for different sections of a song
    sections = [
        {
            "name": "Verse",
            "emotion": "contemplative and intimate",
            "length": 4
        },
        {
            "name": "Chorus", 
            "emotion": "uplifting and memorable",
            "length": 4
        },
        {
            "name": "Bridge",
            "emotion": "contrasting tension and release",
            "length": 6
        }
    ]
    
    song_progressions = {}
    
    for section in sections:
        print(f"\n🎼 {section['name'].upper()} SECTION")
        print(f"Target emotion: '{section['emotion']}'")
        print("-" * 50)
        
        # Generate progression for this section
        analysis = integrator.generate_novel_progression(
            section['emotion'], 
            length=section['length']
        )
        
        song_progressions[section['name']] = analysis
        
        print(f"Generated progression: {' | '.join(analysis.chords)}")
        
        # Show emotional profile
        top_emotions = sorted(analysis.overall_emotion_weights.items(),
                            key=lambda x: x[1], reverse=True)[:3]
        print(f"Emotional profile: {', '.join([f'{e}({w:.2f})' for e, w in top_emotions])}")
        
        # Show harmonic analysis
        print(f"Harmonic flow: {' → '.join([f'{t:.2f}' for t in analysis.harmonic_flow])}")
        print(f"Novelty score: {analysis.novel_pattern_score:.2f} (uniqueness)")
        print(f"Confidence: {analysis.generation_confidence:.2f} (model certainty)")
        
        # Show functional analysis
        print("Functional analysis:")
        for i, chord_analysis in enumerate(analysis.contextual_chord_analyses):
            role_emoji = {
                "tonic": "🏠", "dominant": "⚡", "subdominant": "🌉", 
                "tonic_substitute": "🏠*", "transitional": "↔️"
            }.get(chord_analysis.functional_role, "❓")
            
            print(f"  {i+1}. {chord_analysis.roman_numeral} {role_emoji} "
                  f"({chord_analysis.functional_role}, "
                  f"tension: {chord_analysis.harmonic_tension:.2f})")
    
    # Show song structure summary
    print(f"\n🎵 COMPLETE SONG STRUCTURE")
    print("-" * 50)
    for section_name, analysis in song_progressions.items():
        print(f"{section_name:8}: {' | '.join(analysis.chords)}")
    
    print(f"\n✨ This demonstrates how the integrated system can:")
    print(f"   • Generate contextually appropriate progressions for different song sections")
    print(f"   • Maintain emotional coherence while providing harmonic variety") 
    print(f"   • Provide detailed harmonic analysis for composition decisions")
    print(f"   • Create novel progressions that aren't just database lookups")

def architecture_summary():
    """Summarize the complete architecture"""
    print(f"\n" + "=" * 80)
    print("    ARCHITECTURE SUMMARY")
    print("=" * 80)
    
    print("""
🏗️  SYSTEM ARCHITECTURE:

1️⃣ INDIVIDUAL CHORD MODEL (individual_chord_model.py)
   • Database: individual_chord_database.json
   • Function: Maps emotions → single chords with context
   • Contexts: Mode (Ionian, Aeolian, etc.) + Style (Jazz, Blues, etc.)
   • Output: Chord symbol, roman numeral, emotion weights, contextual metadata

2️⃣ CHORD PROGRESSION MODEL (chord_progression_model.py)  
   • Database: emotion_progression_database.json
   • Function: Maps emotions → chord sequences
   • Features: BERT text encoding, modal fingerprints, genre weighting
   • Output: Chord progressions with mode and genre information

3️⃣ NEURAL PROGRESSION ANALYZER (neural_progression_analyzer.py)
   • Architecture: LSTM + Attention + Multiple prediction heads
   • Function: Contextual integration layer
   • Features:
     - Analyzes progressions through individual chord context
     - Provides contextual weighting for chord emotions
     - Generates novel progressions via learned patterns
     - Estimates pattern novelty and generation confidence

🔗 INTEGRATION WORKFLOW:
   1. Individual model provides base chord-emotion mappings
   2. Progression model provides sequence-level emotional context
   3. Neural analyzer contextually reweights individual chord emotions
   4. System can extrapolate to novel progressions not in database
   
✨ KEY INNOVATION:
   Separates individual chord feelings from progression feelings, then
   intelligently weights individual chords based on their progression context.
   This enables accurate emotion translation for both known and novel progressions.

📊 TRAINING DATA:
   • 144 progression samples from emotion database
   • 12-dimensional emotion vectors (Joy, Sadness, Fear, etc.)
   • Context-aware chord vocabularies from both models
   • Attention-based contextual weighting learning
""")

if __name__ == "__main__":
    compare_models_demo()
    workflow_demo()
    architecture_summary()
