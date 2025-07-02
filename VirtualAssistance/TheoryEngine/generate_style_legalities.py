#!/usr/bin/env python3
"""
Generate style-specific legality rules for all music styles
Creates harmonic progression rules based on each style's chord characteristics
"""

import json
import os
from typing import Dict, List, Set

class StyleLegalityGenerator:
    """Generate style-specific chord progression legality rules"""
    
    def __init__(self):
        self.base_dir = "/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance/TheoryEngine"
        
        # Load existing classical rules as foundation
        with open(os.path.join(self.base_dir, "legalityClassical.json"), 'r') as f:
            self.classical_rules = json.load(f)
        
        # Style characteristics for progression logic
        self.style_characteristics = {
            "Blues": {
                "emphasis": ["dominant_7th", "blues_scale", "circular_progression"],
                "common_moves": ["I7-IV7-V7-I7", "turnarounds", "blues_substitutions"],
                "avoid": ["fancy_extensions", "modal_interchange"]
            },
            "Jazz": {
                "emphasis": ["extended_chords", "ii-V-I", "tritone_substitution", "modal_interchange"],
                "common_moves": ["ii7-V7-IM7", "circle_of_fifths", "chromatic_movement"],
                "avoid": ["simple_triads", "predictable_patterns"]
            },
            "Pop": {
                "emphasis": ["simple_progressions", "strong_hooks", "vi-IV-I-V", "emotional_impact"],
                "common_moves": ["I-V-vi-IV", "vi-IV-I-V", "I-vi-IV-V"],
                "avoid": ["complex_jazz_chords", "dissonant_intervals"]
            },
            "Rock": {
                "emphasis": ["power_chords", "strong_rhythm", "modal_progressions", "driving_force"],
                "common_moves": ["I-bVII-IV", "i-bVII-bVI-bVII", "chromatic_riffs"],
                "avoid": ["gentle_resolutions", "complex_extensions"]
            },
            "Folk": {
                "emphasis": ["simple_harmony", "natural_voice_leading", "pentatonic_influence"],
                "common_moves": ["I-IV-V-I", "vi-IV-I-V", "modal_progressions"],
                "avoid": ["chromatic_harmony", "complex_substitutions"]
            },
            "RnB": {
                "emphasis": ["groove_oriented", "extended_chords", "gospel_influence", "smooth_voice_leading"],
                "common_moves": ["ii-V progressions", "chromatic_bass", "sixth_chords"],
                "avoid": ["harsh_dissonance", "angular_movement"]
            },
            "Cinematic": {
                "emphasis": ["emotional_depth", "wide_voicings", "orchestral_movement", "dramatic_tension"],
                "common_moves": ["epic_progressions", "modal_mixture", "suspended_resolutions"],
                "avoid": ["simple_patterns", "repetitive_structure"]
            }
        }
    
    def generate_all_style_legalities(self):
        """Generate legality rules for all styles"""
        
        print("=== Generating Style-Specific Legality Rules ===\n")
        
        all_legalities = {}
        
        # Include existing classical rules
        all_legalities["Classical"] = self.classical_rules
        print("✓ Classical: Using existing comprehensive rules")
        
        # Generate rules for each style
        styles = ["Blues", "Jazz", "Pop", "Rock", "Folk", "RnB", "Cinematic"]
        
        for style in styles:
            print(f"Generating {style} legality rules...")
            legality_rules = self.generate_style_legality(style)
            all_legalities[style] = legality_rules
            print(f"✓ {style}: Generated rules for {len(legality_rules)} modes")
        
        # Save comprehensive legality file
        output_path = os.path.join(self.base_dir, "legalityAll.json")
        with open(output_path, 'w') as f:
            json.dump(all_legalities, f, indent=2)
        
        print(f"\n✓ Saved comprehensive legality rules to: {output_path}")
        return all_legalities
    
    def generate_style_legality(self, style: str) -> Dict:
        """Generate legality rules for a specific style"""
        
        # Load style syntax data
        syntax_file = os.path.join(self.base_dir, f"syntax{style}.json")
        with open(syntax_file, 'r') as f:
            style_data = json.load(f)
        
        mode_chord_data = style_data.get("modeChordData", {})
        
        legality_rules = {}
        
        # Generate rules for each mode
        for mode, mode_data in mode_chord_data.items():
            legality_rules[mode] = self.generate_mode_legality(style, mode, mode_data)
        
        return legality_rules
    
    def generate_mode_legality(self, style: str, mode: str, mode_data: Dict) -> Dict:
        """Generate legality rules for a specific mode within a style"""
        
        # Extract all chords from the mode data
        all_chords = set()
        chord_functions = {}
        
        for function, function_data in mode_data.items():
            chord_functions[function] = set()
            for chord_type, chords in function_data.items():
                for chord in chords:
                    all_chords.add(chord)
                    chord_functions[function].add(chord)
        
        # Start with classical rules as foundation and adapt
        classical_mode_rules = self.classical_rules.get(mode, {})
        
        legality_rules = {}
        
        for chord in all_chords:
            legality_rules[chord] = self.generate_chord_legality(
                style, mode, chord, chord_functions, all_chords, classical_mode_rules
            )
        
        return legality_rules
    
    def generate_chord_legality(self, style: str, mode: str, chord: str, 
                              chord_functions: Dict, all_chords: Set, classical_rules: Dict) -> List[str]:
        """Generate legality rules for a specific chord"""
        
        # Find the base chord (remove extensions)
        base_chord = self.get_base_chord(chord)
        
        # Start with classical rules if available
        base_targets = classical_rules.get(base_chord, [])
        
        # Apply style-specific modifications
        targets = set(base_targets)
        
        # Add style-specific progressions
        if style == "Blues":
            targets.update(self.get_blues_progressions(chord, all_chords))
        elif style == "Jazz":
            targets.update(self.get_jazz_progressions(chord, all_chords, chord_functions))
        elif style == "Pop":
            targets.update(self.get_pop_progressions(chord, all_chords))
        elif style == "Rock":
            targets.update(self.get_rock_progressions(chord, all_chords))
        elif style == "Folk":
            targets.update(self.get_folk_progressions(chord, all_chords))
        elif style == "RnB":
            targets.update(self.get_rnb_progressions(chord, all_chords))
        elif style == "Cinematic":
            targets.update(self.get_cinematic_progressions(chord, all_chords))
        
        # Filter to only include chords that exist in this style/mode
        valid_targets = [t for t in targets if t in all_chords or self.get_base_chord(t) in {self.get_base_chord(c) for c in all_chords}]
        
        # Ensure chord can resolve to itself (for pedal tones, etc.)
        if chord not in valid_targets:
            valid_targets.append(chord)
        
        return sorted(list(set(valid_targets)))
    
    def get_base_chord(self, chord: str) -> str:
        """Extract base chord symbol, removing extensions"""
        # Remove common extensions
        base = chord
        extensions = ['7', '9', '11', '13', 'add9', 'sus4', 'sus2', '+', '°', 'ø', 'M7', 'm7']
        
        for ext in sorted(extensions, key=len, reverse=True):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        
        return base if base else chord
    
    def get_blues_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate blues-specific progressions"""
        targets = set()
        
        # Blues is primarily I7-IV7-V7 based
        if "I7" in chord:
            targets.update(["IV7", "V7", "I7", "bVII7"])
        elif "IV7" in chord:
            targets.update(["I7", "V7", "IV7"])
        elif "V7" in chord:
            targets.update(["I7", "IV7"])
        elif "bVII7" in chord:
            targets.update(["I7", "IV7"])
        
        # Blues turnarounds
        if chord.startswith("I"):
            targets.update(["vi7", "ii7", "V7"])
        
        return targets
    
    def get_jazz_progressions(self, chord: str, all_chords: Set, chord_functions: Dict) -> Set[str]:
        """Generate jazz-specific progressions"""
        targets = set()
        
        # ii-V-I progressions
        if "ii" in chord:
            targets.update(["V7", "V7alt", "V9", "V13"])
        elif chord.startswith("V"):
            targets.update(["IM7", "I", "vi7"])
        elif chord.startswith("I") and ("M7" in chord or chord == "I"):
            targets.update(["vi7", "iiø7", "ii7"])
        
        # Circle of fifths movement
        if "iii" in chord:
            targets.update(["vi7", "VI7"])
        elif "vi" in chord:
            targets.update(["ii7", "II7"])
        
        # Tritone substitutions
        if "V7" in chord:
            targets.update(["bII7", "IM7"])
        
        # Extended harmony resolutions
        if "7" in chord or "9" in chord or "11" in chord or "13" in chord:
            targets.update([c for c in all_chords if "M7" in c or c.endswith("7")])
        
        return targets
    
    def get_pop_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate pop-specific progressions"""
        targets = set()
        
        # Classic pop progressions: I-V-vi-IV, vi-IV-I-V
        if chord == "I":
            targets.update(["V", "vi", "IV", "iii"])
        elif chord == "V":
            targets.update(["I", "vi", "IV"])
        elif chord == "vi":
            targets.update(["IV", "I", "V", "ii"])
        elif chord == "IV":
            targets.update(["I", "V", "vi", "ii"])
        elif chord == "ii":
            targets.update(["V", "vi"])
        elif chord == "iii":
            targets.update(["vi", "IV"])
        
        # Add common pop extensions
        if "add9" in chord or "sus" in chord:
            base = self.get_base_chord(chord)
            targets.update([c for c in all_chords if self.get_base_chord(c) == base])
        
        return targets
    
    def get_rock_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate rock-specific progressions"""
        targets = set()
        
        # Common rock progressions: I-bVII-IV, i-bVII-bVI
        if chord.startswith("I"):
            targets.update(["bVII", "IV", "V", "vi"])
        elif "bVII" in chord:
            targets.update(["I", "IV", "bVI", "i"])
        elif chord == "IV":
            targets.update(["I", "V", "bVII"])
        elif "bVI" in chord:
            targets.update(["bVII", "i", "IV"])
        
        # Modal rock progressions
        if chord.startswith("i"):
            targets.update(["bVII", "bVI", "iv", "v"])
        elif "iv" in chord:
            targets.update(["i", "bVII", "V"])
        
        # Power chord movement (chromatic)
        if chord in all_chords:
            # Add chromatic neighbors
            targets.update([c for c in all_chords])
        
        return targets
    
    def get_folk_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate folk-specific progressions"""
        targets = set()
        
        # Simple folk progressions: I-IV-V-I, vi-IV-I-V
        if chord == "I":
            targets.update(["IV", "V", "vi", "ii"])
        elif chord == "IV":
            targets.update(["I", "V", "vi"])
        elif chord == "V":
            targets.update(["I", "vi"])
        elif chord == "vi":
            targets.update(["IV", "I", "ii"])
        elif chord == "ii":
            targets.update(["V", "vi"])
        
        # Modal folk progressions
        if chord.startswith("i"):
            targets.update(["bVII", "bVI", "iv"])
        elif "bVII" in chord:
            targets.update(["i", "bVI"])
        elif "bVI" in chord:
            targets.update(["bVII", "i"])
        
        return targets
    
    def get_rnb_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate R&B-specific progressions"""
        targets = set()
        
        # R&B loves ii-V movements and sixth chords
        if "ii" in chord:
            targets.update(["V7", "V9", "IM7", "I6"])
        elif chord.startswith("V"):
            targets.update(["IM7", "I6", "vi7"])
        elif "IM7" in chord or "I6" in chord:
            targets.update(["vi7", "ii7", "IV6"])
        
        # Gospel-influenced progressions
        if chord == "I":
            targets.update(["I6", "IM7", "vi7", "IV6"])
        elif "vi" in chord:
            targets.update(["ii7", "V7", "I6"])
        
        # Smooth voice leading
        if "6" in chord or "7" in chord:
            targets.update([c for c in all_chords if "6" in c or "7" in c])
        
        return targets
    
    def get_cinematic_progressions(self, chord: str, all_chords: Set) -> Set[str]:
        """Generate cinematic-specific progressions"""
        targets = set()
        
        # Cinematic music uses wide harmonic movement
        if chord.startswith("I"):
            targets.update(["bVI", "iv", "V", "bII", "vi"])
        elif "bVI" in chord:
            targets.update(["I", "iv", "bVII"])
        elif "iv" in chord:
            targets.update(["I", "bVI", "V"])
        elif "bII" in chord:
            targets.update(["I", "V"])
        
        # Suspended resolutions
        if "sus" in chord:
            base = self.get_base_chord(chord)
            targets.update([c for c in all_chords if self.get_base_chord(c) == base])
        
        # Epic progressions
        if chord.startswith("vi"):
            targets.update(["IV", "I", "V", "bVI"])
        
        # Modal mixture
        targets.update([c for c in all_chords if "b" in c or "#" in c])
        
        return targets

def main():
    """Generate comprehensive style legality rules"""
    
    generator = StyleLegalityGenerator()
    all_legalities = generator.generate_all_style_legalities()
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated legality rules for {len(all_legalities)} styles:")
    for style, rules in all_legalities.items():
        print(f"  {style}: {len(rules)} modes")
    
    print("\n✓ All style-specific legality rules have been generated!")

if __name__ == "__main__":
    main()
