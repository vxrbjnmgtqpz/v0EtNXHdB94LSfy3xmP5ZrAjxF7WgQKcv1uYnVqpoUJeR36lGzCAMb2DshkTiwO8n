#!/usr/bin/env python3
"""
Consolidated Syntax Files Demo
Demonstrates the benefits of having all style syntax data in a single file
"""

from enhanced_solfege_theory_engine import EnhancedSolfegeTheoryEngine
import json
import os

def demonstrate_consolidated_syntax():
    """Show the consolidated syntax file in action"""
    
    print("=" * 70)
    print("CONSOLIDATED SYNTAX FILES DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize engine
    engine = EnhancedSolfegeTheoryEngine()
    print()
    
    # Show file size comparison
    script_dir = os.path.dirname(os.path.abspath(__file__))
    syntax_all_path = os.path.join(script_dir, "syntaxAll.json")
    
    if os.path.exists(syntax_all_path):
        size_kb = os.path.getsize(syntax_all_path) / 1024
        print(f"📁 syntaxAll.json file size: {size_kb:.1f} KB")
        
        # Load and show structure
        with open(syntax_all_path, 'r') as f:
            syntax_data = json.load(f)
        
        print(f"📊 Contains data for {len(syntax_data)} styles:")
        for style_name, style_data in syntax_data.items():
            modes = list(style_data.keys())
            total_chords = 0
            for mode_data in style_data.values():
                for function_data in mode_data.values():
                    for chord_list in function_data.values():
                        if isinstance(chord_list, list):
                            total_chords += len(chord_list)
            print(f"   • {style_name:12}: {len(modes)} modes, ~{total_chords} chord definitions")
        print()
    
    # Demo 1: Cross-style progression comparison
    print("🎵 DEMO 1: All Styles in Ionian Mode (Major Key)")
    print("-" * 50)
    comparison = engine.compare_style_progressions('Ionian', 4)
    for style, progression in comparison.items():
        if progression:
            print(f"{style:12}: {' → '.join(progression)}")
    print()
    
    # Demo 2: Style consistency across modes
    print("🎵 DEMO 2: Jazz Across Different Modes")
    print("-" * 36)
    jazz_modes = ['Ionian', 'Dorian', 'Mixolydian', 'Aeolian']
    for mode in jazz_modes:
        progression = engine.generate_legal_progression('Jazz', mode, 4)
        if progression:
            print(f"{mode:12}: {' → '.join(progression)}")
        else:
            print(f"{mode:12}: (No progression available)")
    print()
    
    # Demo 3: Blues progressions across modes
    print("🎵 DEMO 3: Blues Across Different Modes")
    print("-" * 37)
    blues_modes = ['Ionian', 'Dorian', 'Mixolydian', 'Aeolian']
    for mode in blues_modes:
        progression = engine.generate_legal_progression('Blues', mode, 4)
        if progression:
            print(f"{mode:12}: {' → '.join(progression)}")
        else:
            print(f"{mode:12}: (No progression available)")
    print()
    
    # Demo 4: Extended progressions
    print("🎵 DEMO 4: Extended 8-Chord Progressions")
    print("-" * 38)
    extended_styles = ['Classical', 'Jazz', 'Blues', 'Cinematic']
    for style in extended_styles:
        progression = engine.generate_legal_progression(style, 'Ionian', 8)
        if progression:
            # Split into two lines for readability
            part1 = ' → '.join(progression[:4])
            part2 = ' → '.join(progression[4:])
            print(f"{style:12}: {part1}")
            print(f"{'':12}  {part2}")
        print()
    
    print("=" * 70)
    print("✅ CONSOLIDATION BENEFITS:")
    print("• Single file contains all style syntax data")
    print("• Simplified loading and management")
    print("• Consistent structure across all styles")
    print("• Easy to extend with new styles")
    print("• Reduced file system complexity")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_consolidated_syntax()
