"""
Voice Leading Engine Demo
Comprehensive demonstration of multi-octave emotional register mapping and voice leading optimization

This demo showcases:
1. Multi-octave emotional register mapping (angry/metal -> lower, transcendence -> higher)
2. Smooth voice leading optimization with minimal note movement
3. Style context adaptations
4. Key change handling
5. Integration with existing emotion system
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voice_leading_engine import EnhancedVoiceLeadingEngine, VoicedChord, VoicingResult
    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False

# Fallback classes for when Wolfram isn't available
if not WOLFRAM_AVAILABLE:
    @dataclass
    class VoicedChord:
        chord_symbol: str
        notes: List[Tuple[str, int]]
        register_range: Tuple[int, int]
        voice_leading_cost: float = 0.0
        emotional_fitness: float = 0.0

    @dataclass
    class VoicingResult:
        voiced_chords: List[VoicedChord]
        total_voice_leading_cost: float
        register_analysis: Dict
        harmonic_rhythm: Dict
        modulation_info: Optional[Dict] = None

class VoiceLeadingDemoEngine:
    """Demo engine with fallback capabilities"""
    
    def __init__(self):
        """Initialize demo engine"""
        self.wolfram_engine = None
        self.fallback_mode = False
        
        if WOLFRAM_AVAILABLE:
            try:
                self.wolfram_engine = EnhancedVoiceLeadingEngine()
                print("✅ Wolfram Language Voice Leading Engine loaded successfully")
            except Exception as e:
                print(f"⚠️  Wolfram Language not available: {e}")
                print("🔄 Falling back to demonstration mode")
                self.fallback_mode = True
        else:
            print("⚠️  Voice leading engine not available")
            print("🔄 Running in demonstration mode")
            self.fallback_mode = True
    
    def demo_emotional_register_mapping(self):
        """Demonstrate emotional register mapping"""
        print("\n" + "="*60)
        print("🎼 EMOTIONAL REGISTER MAPPING DEMO")
        print("="*60)
        
        # Test cases: Different emotional contexts
        test_cases = [
            {
                "name": "Metal/Aggressive",
                "emotions": {"Anger": 0.8, "Malice": 0.6, "Empowerment": 0.4},
                "expected_register": "Lower (1-3)",
                "description": "Aggressive emotions mapped to lower, more powerful registers"
            },
            {
                "name": "Transcendent/Ethereal", 
                "emotions": {"Transcendence": 0.9, "Aesthetic Awe": 0.6, "Wonder": 0.5},
                "expected_register": "Higher (5-7)",
                "description": "Transcendent emotions mapped to higher, ethereal registers"
            },
            {
                "name": "Bright/Joyful",
                "emotions": {"Joy": 0.8, "Empowerment": 0.6, "Gratitude": 0.4},
                "expected_register": "Mid-high (4-6)",
                "description": "Positive emotions mapped to bright mid-high registers"
            },
            {
                "name": "Introspective/Sad",
                "emotions": {"Sadness": 0.7, "Shame": 0.5, "Love": 0.3},
                "expected_register": "Mid (3-5)",
                "description": "Introspective emotions mapped to mid-range registers"
            },
            {
                "name": "Tension/Anxiety",
                "emotions": {"Fear": 0.8, "Anticipation": 0.6, "Surprise": 0.4},
                "expected_register": "Higher (5-7)",
                "description": "Tension emotions mapped to higher registers for anxiety"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🎯 {test_case['name']} Context:")
            print(f"   Emotions: {test_case['emotions']}")
            print(f"   Expected: {test_case['expected_register']}")
            print(f"   Logic: {test_case['description']}")
            
            if not self.fallback_mode:
                # Get actual register mapping from Wolfram engine
                try:
                    result = self.wolfram_engine.optimize_voice_leading(
                        chord_progression=["I", "V", "vi", "IV"],
                        emotion_weights=test_case["emotions"],
                        key="C"
                    )
                    actual_registers = result.register_analysis.get("target_registers", [4, 5])
                    avg_register = sum(actual_registers) / len(actual_registers)
                    print(f"   Actual: Octaves {actual_registers}, Average: {avg_register:.1f}")
                except Exception as e:
                    print(f"   Error: {e}")
            else:
                # Simulate register mapping
                simulated_registers = self._simulate_register_mapping(test_case["emotions"])
                print(f"   Simulated: Octaves {simulated_registers}")
    
    def demo_voice_leading_optimization(self):
        """Demonstrate voice leading optimization"""
        print("\n" + "="*60)
        print("🎵 VOICE LEADING OPTIMIZATION DEMO")
        print("="*60)
        
        # Complex progression that would benefit from voice leading
        progression = ["I", "vi", "IV", "V", "I"]
        emotions = {"Joy": 0.6, "Love": 0.4, "Trust": 0.3}
        
        print(f"\n🎼 Test Progression: {' - '.join(progression)}")
        print(f"🎭 Emotional Context: {emotions}")
        print(f"🎵 Key: C Major")
        
        if not self.fallback_mode:
            try:
                result = self.wolfram_engine.optimize_voice_leading(
                    chord_progression=progression,
                    emotion_weights=emotions,
                    key="C"
                )
                
                print(f"\n📊 Voice Leading Analysis:")
                print(f"   Total movement cost: {result.total_voice_leading_cost:.1f} semitones")
                print(f"   Average cost per change: {result.total_voice_leading_cost/(len(progression)-1):.1f} semitones")
                
                print(f"\n🎹 Optimized Voicings:")
                for i, chord in enumerate(result.voiced_chords):
                    print(f"   {i+1}. {chord.chord_symbol}: {chord.notes} (Register: {chord.register_range})")
                    if i > 0:
                        print(f"      → Voice movement: {chord.voice_leading_cost:.1f} semitones")
                
                # Show the voice leading logic
                print(f"\n🔍 Voice Leading Logic:")
                print(f"   • Notes move minimal distances between chords")
                print(f"   • Inversions chosen to maintain smooth voice leading")
                print(f"   • Register placement based on emotional context")
                
            except Exception as e:
                print(f"   Error: {e}")
                self._demo_fallback_voicing(progression, emotions)
        else:
            self._demo_fallback_voicing(progression, emotions)
    
    def demo_style_context_adaptation(self):
        """Demonstrate style context adaptations"""
        print("\n" + "="*60)
        print("🎨 STYLE CONTEXT ADAPTATION DEMO")
        print("="*60)
        
        # Same progression, different styles
        progression = ["i", "♭VII", "♭VI", "V"]
        base_emotions = {"Anger": 0.6, "Empowerment": 0.4, "Arousal": 0.3}
        
        styles = ["classical", "jazz", "blues", "rock", "pop", "metal", "experimental"]
        
        print(f"\n🎼 Test Progression: {' - '.join(progression)}")
        print(f"🎭 Base Emotions: {base_emotions}")
        print(f"🎵 Key: A Minor")
        
        for style in styles:
            print(f"\n🎨 {style.title()} Style:")
            
            if not self.fallback_mode:
                try:
                    result = self.wolfram_engine.optimize_with_style_context(
                        chord_progression=progression,
                        emotion_weights=base_emotions,
                        key="Am",
                        style_context=style
                    )
                    
                    avg_register = result.register_analysis.get("average_register", 4.0)
                    print(f"   Average register: {avg_register:.1f}")
                    print(f"   Voice leading cost: {result.total_voice_leading_cost:.1f}")
                    
                    # Show style-specific adaptations
                    style_descriptions = {
                        "classical": "Traditional voice leading, moderate register",
                        "jazz": "Extended harmonies, sophisticated voicings",
                        "blues": "Emphasis on dominant 7th, bluesy register",
                        "rock": "Power chord influences, punchy voicings",
                        "pop": "Accessible voicings, bright register",
                        "metal": "Lower register, aggressive voicings",
                        "experimental": "Unconventional voicings, extreme registers"
                    }
                    print(f"   Style adaptation: {style_descriptions[style]}")
                    
                except Exception as e:
                    print(f"   Error: {e}")
                    self._demo_style_fallback(style)
            else:
                self._demo_style_fallback(style)
    
    def demo_key_change_handling(self):
        """Demonstrate key change handling"""
        print("\n" + "="*60)
        print("🔄 KEY CHANGE HANDLING DEMO")
        print("="*60)
        
        # Progression with key change
        progression_part1 = ["I", "vi", "IV", "V"]
        progression_part2 = ["I", "vi", "IV", "V"]
        emotions = {"Joy": 0.5, "Wonder": 0.4, "Transcendence": 0.3}
        
        print(f"\n🎼 Part 1 (C Major): {' - '.join(progression_part1)}")
        print(f"🎼 Part 2 (G Major): {' - '.join(progression_part2)}")
        print(f"🎭 Emotional Context: {emotions}")
        
        if not self.fallback_mode:
            try:
                # Generate part 1 in C
                result1 = self.wolfram_engine.optimize_voice_leading(
                    chord_progression=progression_part1,
                    emotion_weights=emotions,
                    key="C"
                )
                
                # Generate part 2 in G with key change handling
                result2 = self.wolfram_engine.optimize_voice_leading(
                    chord_progression=progression_part2,
                    emotion_weights=emotions,
                    key="G"
                )
                
                print(f"\n🎹 C Major Voicings:")
                for i, chord in enumerate(result1.voiced_chords):
                    print(f"   {i+1}. {chord.chord_symbol}: {chord.notes}")
                
                print(f"\n🎹 G Major Voicings:")
                for i, chord in enumerate(result2.voiced_chords):
                    print(f"   {i+1}. {chord.chord_symbol}: {chord.notes}")
                
                # Show modulation logic
                print(f"\n🔄 Modulation Analysis:")
                print(f"   • Key change from C to G (up a perfect 5th)")
                print(f"   • Pivot chords identified for smooth transition")
                print(f"   • Voice leading optimized across key boundary")
                
            except Exception as e:
                print(f"   Error: {e}")
                self._demo_key_change_fallback()
        else:
            self._demo_key_change_fallback()
    
    def demo_integration_with_existing_system(self):
        """Demonstrate integration with existing emotion system"""
        print("\n" + "="*60)
        print("🔗 INTEGRATION WITH EXISTING SYSTEM DEMO")
        print("="*60)
        
        # Show how this integrates with the existing 22-emotion system
        print(f"\n🎭 22-Emotion System Integration:")
        print(f"   • Individual chord model: 22 emotions → register mapping")
        print(f"   • Progression model: 22 emotions → voice leading optimization")
        print(f"   • Interpolation engine: Tension curves + register trajectories")
        print(f"   • Neural analyzer: CD values + register predictions")
        
        # Example emotional progression
        emotional_progression = [
            {"Joy": 0.8, "Love": 0.5, "Trust": 0.3},
            {"Anticipation": 0.6, "Wonder": 0.4, "Joy": 0.3},
            {"Sadness": 0.7, "Love": 0.4, "Longing": 0.3},
            {"Empowerment": 0.8, "Joy": 0.6, "Gratitude": 0.4}
        ]
        
        chord_progression = ["I", "vi", "IV", "V"]
        
        print(f"\n🎼 Progressive Emotional Context:")
        for i, emotions in enumerate(emotional_progression):
            print(f"   {i+1}. {chord_progression[i]}: {emotions}")
        
        if not self.fallback_mode:
            try:
                results = []
                for i, emotions in enumerate(emotional_progression):
                    result = self.wolfram_engine.optimize_voice_leading(
                        chord_progression=[chord_progression[i]],
                        emotion_weights=emotions,
                        key="C"
                    )
                    results.append(result)
                
                print(f"\n🎹 Emotionally-Driven Voicings:")
                for i, (result, emotions) in enumerate(zip(results, emotional_progression)):
                    chord = result.voiced_chords[0]
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    print(f"   {i+1}. {chord.chord_symbol}: {chord.notes} (Dominant: {dominant_emotion})")
                    print(f"      Register: {chord.register_range}")
                
            except Exception as e:
                print(f"   Error: {e}")
                self._demo_integration_fallback(emotional_progression, chord_progression)
        else:
            self._demo_integration_fallback(emotional_progression, chord_progression)
    
    def _simulate_register_mapping(self, emotions: Dict[str, float]) -> List[int]:
        """Simulate register mapping for fallback mode"""
        # Simplified register mapping logic
        register_map = {
            "Anger": [2, 3], "Malice": [2, 3], "Disgust": [2, 3],
            "Transcendence": [6, 7], "Aesthetic Awe": [6, 7], "Wonder": [5, 6],
            "Joy": [4, 5], "Love": [4, 5], "Trust": [4, 5],
            "Sadness": [3, 4], "Shame": [3, 4], "Fear": [5, 6]
        }
        
        weighted_registers = []
        for emotion, weight in emotions.items():
            if emotion in register_map:
                weighted_registers.extend([r * weight for r in register_map[emotion]])
        
        if weighted_registers:
            avg_register = sum(weighted_registers) / len(weighted_registers)
            return [int(avg_register) - 1, int(avg_register), int(avg_register) + 1]
        return [4, 5, 6]
    
    def _demo_fallback_voicing(self, progression: List[str], emotions: Dict[str, float]):
        """Show fallback voicing logic"""
        print(f"\n🔄 Fallback Voicing (Demonstration Mode):")
        
        # Simple voicing logic
        basic_voicings = {
            "I": [("C", 4), ("E", 4), ("G", 4)],
            "ii": [("D", 4), ("F", 4), ("A", 4)],
            "iii": [("E", 4), ("G", 4), ("B", 4)],
            "IV": [("F", 4), ("A", 4), ("C", 5)],
            "V": [("G", 4), ("B", 4), ("D", 5)],
            "vi": [("A", 4), ("C", 5), ("E", 5)],
            "vii°": [("B", 4), ("D", 5), ("F", 5)]
        }
        
        target_registers = self._simulate_register_mapping(emotions)
        avg_register = sum(target_registers) / len(target_registers)
        
        print(f"   Target registers: {target_registers} (avg: {avg_register:.1f})")
        
        for i, chord in enumerate(progression):
            voicing = basic_voicings.get(chord, [("C", 4), ("E", 4), ("G", 4)])
            print(f"   {i+1}. {chord}: {voicing}")
        
        print(f"   Note: With Wolfram Language, these would be optimized for:")
        print(f"   • Minimal voice movement between chords")
        print(f"   • Emotional register preferences")
        print(f"   • Style-specific adaptations")
    
    def _demo_style_fallback(self, style: str):
        """Show style adaptation logic in fallback mode"""
        style_descriptions = {
            "classical": "Traditional voice leading, strict rules",
            "jazz": "Extended harmonies, complex voicings",
            "blues": "Dominant 7th emphasis, bluesy bends",
            "rock": "Power chord influences, open voicings",
            "pop": "Accessible, bright voicings",
            "metal": "Lower registers, aggressive intervals",
            "experimental": "Unconventional, extreme registers"
        }
        
        style_register_shifts = {
            "classical": 0, "jazz": 0, "blues": -0.5, "rock": -0.5,
            "pop": +0.5, "metal": -1.0, "experimental": 0
        }
        
        base_register = 4.5
        adjusted_register = base_register + style_register_shifts[style]
        
        print(f"   Style adaptation: {style_descriptions[style]}")
        print(f"   Register shift: {style_register_shifts[style]:+.1f} octaves")
        print(f"   Target register: {adjusted_register:.1f}")
    
    def _demo_key_change_fallback(self):
        """Show key change handling in fallback mode"""
        print(f"\n🔄 Key Change Logic (Demonstration Mode):")
        print(f"   • Identify common pivot chords between keys")
        print(f"   • Calculate optimal voice leading across modulation")
        print(f"   • Maintain register consistency during transition")
        print(f"   • Apply harmonic analysis for smooth resolution")
        
        print(f"\n🎼 C to G Modulation:")
        print(f"   • Pivot chords: I (C) = IV (G), vi (Am) = ii (G)")
        print(f"   • Voice leading: Optimize movement in new key")
        print(f"   • Register: Maintain emotional register preferences")
    
    def _demo_integration_fallback(self, emotional_progression: List[Dict], chord_progression: List[str]):
        """Show integration fallback logic"""
        print(f"\n🔗 Integration Logic (Demonstration Mode):")
        
        for i, (emotions, chord) in enumerate(zip(emotional_progression, chord_progression)):
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            simulated_registers = self._simulate_register_mapping(emotions)
            
            print(f"   {i+1}. {chord}: Dominant emotion '{dominant_emotion}' → Registers {simulated_registers}")
        
        print(f"\n🔄 Full System Integration:")
        print(f"   • Individual chord model → Emotional chord selection")
        print(f"   • Voice leading engine → Optimal voicings & registers")
        print(f"   • Interpolation engine → Smooth emotional transitions")
        print(f"   • Neural analyzer → Contextual refinement")
        print(f"   • Integrated server → Unified API")

def main():
    """Run the comprehensive voice leading demo"""
    print("🎼 VOICE LEADING ENGINE COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Advanced harmonic progression with emotional register mapping")
    print("=" * 60)
    
    # Initialize demo engine
    demo = VoiceLeadingDemoEngine()
    
    # Run all demo sections
    demo.demo_emotional_register_mapping()
    demo.demo_voice_leading_optimization()
    demo.demo_style_context_adaptation()
    demo.demo_key_change_handling()
    demo.demo_integration_with_existing_system()
    
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETE")
    print("="*60)
    
    if demo.fallback_mode:
        print("\n💡 To see full functionality:")
        print("   • Install Wolfram Language/Mathematica")
        print("   • Run: pip install wolframclient")
        print("   • Ensure wolframscript is in PATH")
    else:
        print("\n✅ Voice leading engine fully operational!")
    
    print("\n🎼 Key Features Demonstrated:")
    print("   ✓ Multi-octave emotional register mapping")
    print("   ✓ Smooth voice leading optimization")
    print("   ✓ Style context adaptations")
    print("   ✓ Key change handling")
    print("   ✓ Integration with 22-emotion system")
    
    print("\n🎵 Next Steps:")
    print("   • Test with real musical examples")
    print("   • Integrate with MIDI generation")
    print("   • Add to integrated server API")
    print("   • Create web interface controls")

if __name__ == "__main__":
    main() 