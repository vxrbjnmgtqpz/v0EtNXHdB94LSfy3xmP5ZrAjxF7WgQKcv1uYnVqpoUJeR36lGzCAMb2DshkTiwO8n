#!/usr/bin/env python3
"""
Style-Specific Chord Progression Legality Demo
Demonstrates the enhanced theory engine's style-aware progression generation
"""

from enhanced_solfege_theory_engine import EnhancedSolfegeTheoryEngine

def main():
    # Initialize the enhanced theory engine
    engine = EnhancedSolfegeTheoryEngine()
    
    print("=" * 70)
    print("ENHANCED SOLFEGE THEORY ENGINE - STYLE-SPECIFIC LEGALITY DEMO")
    print("=" * 70)
    print()
    
    # Demo 1: Style Comparison in Major Key
    print("🎵 DEMO 1: Different Styles in Ionian Mode (Major Key)")
    print("-" * 50)
    comparison = engine.compare_style_progressions('Ionian', 4)
    for style, progression in comparison.items():
        if progression:  # Only show styles that generated progressions
            print(f"{style:12}: {' → '.join(progression)}")
    print()
    
    # Demo 2: Style Comparison in Minor Key  
    print("🎵 DEMO 2: Different Styles in Aeolian Mode (Natural Minor)")
    print("-" * 58)
    comparison = engine.compare_style_progressions('Aeolian', 4)
    for style, progression in comparison.items():
        if progression:
            print(f"{style:12}: {' → '.join(progression)}")
    print()
    
    # Demo 3: Modal Jazz Progressions
    print("🎵 DEMO 3: Jazz Progressions in Different Modes")
    print("-" * 44)
    modes = ['Ionian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian', 'Locrian']
    for mode in modes:
        progression = engine.generate_legal_progression('Jazz', mode, 4)
        if progression:
            print(f"{mode:12}: {' → '.join(progression)}")
        else:
            print(f"{mode:12}: (No progression generated)")
    print()
    
    # Demo 4: Blues Progressions in Different Keys
    print("🎵 DEMO 4: Blues Progressions in Different Modes")
    print("-" * 45)
    for mode in ['Ionian', 'Dorian', 'Mixolydian']:
        progression = engine.generate_legal_progression('Blues', mode, 6)
        if progression:
            print(f"{mode:12}: {' → '.join(progression)}")
    print()
    
    # Demo 5: Extended Progressions
    print("🎵 DEMO 5: Extended Progressions (8 chords)")
    print("-" * 39)
    extended_styles = ['Classical', 'Jazz', 'Pop', 'Rock']
    for style in extended_styles:
        progression = engine.generate_legal_progression(style, 'Ionian', 8)
        if progression:
            # Break into two lines for readability
            part1 = ' → '.join(progression[:4])
            part2 = ' → '.join(progression[4:])
            print(f"{style:10}: {part1}")
            print(f"{'':10}  {part2}")
        print()
    
    print("=" * 70)
    print("All style-specific legality rules are now operational!")
    print("The theory engine supports 8 styles across 7 musical modes.")
    print("=" * 70)

if __name__ == "__main__":
    main()
