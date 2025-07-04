"""
Voice Leading Engine - Python Integration Layer
Advanced harmonic progression with smooth voice leading and emotional register mapping

This module provides a Python interface to the Wolfram Language voice leading engine,
enabling sophisticated chord voicing optimization with emotional register mapping.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoicedChord:
    """Represents a chord with specific voicing and octave information"""
    chord_symbol: str
    notes: List[Tuple[str, int]]  # [(note_name, octave), ...]
    register_range: Tuple[int, int]  # (min_octave, max_octave)
    voice_leading_cost: float = 0.0
    emotional_fitness: float = 0.0

@dataclass
class VoicingResult:
    """Result of voice leading optimization"""
    voiced_chords: List[VoicedChord]
    total_voice_leading_cost: float
    register_analysis: Dict[str, Union[float, List[int]]]
    harmonic_rhythm: Dict[str, List[float]]
    modulation_info: Optional[Dict[str, str]] = None

class WolframVoiceLeadingEngine:
    """Python interface to Wolfram Language voice leading engine"""
    
    def __init__(self, wolfram_script_path: str = "wolframscript"):
        """
        Initialize the voice leading engine
        
        Args:
            wolfram_script_path: Path to wolframscript executable
        """
        self.wolfram_script_path = wolfram_script_path
        self.engine_path = os.path.join(os.path.dirname(__file__), "TheoryEngine", "VoiceLeadingEngine.wl")
        
        # Test Wolfram availability
        self._test_wolfram_availability()
    
    def _test_wolfram_availability(self) -> None:
        """Test if Wolfram Language is available"""
        try:
            result = subprocess.run(
                [self.wolfram_script_path, "-code", "2+2"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"Wolfram test failed: {result.stderr}")
                raise RuntimeError("Wolfram Language not available")
        except Exception as e:
            logger.error(f"Wolfram Language not available: {e}")
            raise RuntimeError("Wolfram Language required for voice leading engine")
    
    def optimize_voice_leading(
        self,
        chord_progression: List[str],
        emotion_weights: Dict[str, float],
        key: str = "C",
        style_context: Optional[str] = None
    ) -> VoicingResult:
        """
        Optimize voice leading for a chord progression
        
        Args:
            chord_progression: List of chord symbols (Roman numerals)
            emotion_weights: Dictionary of emotion names to weights
            key: Key signature
            style_context: Optional style context for register adjustment
            
        Returns:
            VoicingResult with optimized voicings
        """
        try:
            # Prepare Wolfram Language code
            wl_code = self._generate_wolfram_code(
                chord_progression, emotion_weights, key, style_context
            )
            
            # Execute Wolfram Language code
            result = self._execute_wolfram_code(wl_code)
            
            # Parse and return results
            return self._parse_wolfram_result(result, chord_progression)
            
        except Exception as e:
            logger.error(f"Voice leading optimization failed: {e}")
            # Return fallback simple voicing
            return self._create_fallback_voicing(chord_progression, emotion_weights, key)
    
    def _generate_wolfram_code(
        self,
        chord_progression: List[str],
        emotion_weights: Dict[str, float],
        key: str,
        style_context: Optional[str]
    ) -> str:
        """Generate Wolfram Language code for voice leading optimization"""
        
        # Convert Python data to WL format
        wl_progression = "{" + ", ".join(f'"{chord}"' for chord in chord_progression) + "}"
        
        # Convert emotion weights to WL Association
        wl_emotions = "<|" + ", ".join(
            f'"{emotion}" -> {weight}' 
            for emotion, weight in emotion_weights.items()
        ) + "|>"
        
        # Apply style context modifiers if provided
        style_modifier = ""
        if style_context:
            style_modifiers = {
                "classical": 1.0,
                "jazz": 0.8,
                "blues": 0.7,
                "rock": 0.9,
                "pop": 0.9,
                "metal": 0.6,
                "experimental": 0.5
            }
            modifier = style_modifiers.get(style_context.lower(), 1.0)
            style_modifier = f"""
            (* Apply style context modifier *)
            emotionWeights = Map[# * {modifier} &, emotionWeights];
            """
        
        wl_code = f"""
        (* Load voice leading engine *)
        Get["{self.engine_path}"];
        
        (* Define inputs *)
        progression = {wl_progression};
        emotionWeights = {wl_emotions};
        keySignature = "{key}";
        
        {style_modifier}
        
        (* Optimize voice leading *)
        voicingResult = OptimizeVoiceLeading[progression, emotionWeights, keySignature];
        
        (* Calculate additional analysis *)
        targetRegisters = MapEmotionToRegister[emotionWeights];
        harmonicAnalysis = analyzeHarmonicRhythm[voicingResult, emotionWeights];
        
        (* Calculate total voice leading cost *)
        totalCost = 0.0;
        If[Length[voicingResult] > 1,
            totalCost = Total[Table[
                CalculateVoiceDistance[
                    Map[noteToMIDI[#[[1]], #[[2]]] &, voicingResult[[i]]],
                    Map[noteToMIDI[#[[1]], #[[2]]] &, voicingResult[[i+1]]]
                ],
                {{i, 1, Length[voicingResult] - 1}}
            ]]
        ];
        
        (* Format output as JSON *)
        output = <|
            "voicings" -> voicingResult,
            "totalCost" -> totalCost,
            "targetRegisters" -> targetRegisters,
            "harmonicAnalysis" -> harmonicAnalysis,
            "key" -> keySignature
        |>;
        
        (* Export as JSON *)
        ExportString[output, "JSON"]
        """
        
        return wl_code
    
    def _execute_wolfram_code(self, wl_code: str) -> str:
        """Execute Wolfram Language code and return result"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.wl', delete=False) as f:
                f.write(wl_code)
                temp_file = f.name
            
            result = subprocess.run(
                [self.wolfram_script_path, "-file", temp_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.returncode != 0:
                logger.error(f"Wolfram execution failed: {result.stderr}")
                raise RuntimeError(f"Wolfram execution failed: {result.stderr}")
            
            return result.stdout.strip()
            
        except Exception as e:
            logger.error(f"Wolfram execution error: {e}")
            raise
    
    def _parse_wolfram_result(self, wl_result: str, chord_progression: List[str]) -> VoicingResult:
        """Parse Wolfram Language result into Python objects"""
        try:
            # Parse JSON output from Wolfram
            data = json.loads(wl_result)
            
            # Extract voicings
            voiced_chords = []
            for i, (chord_symbol, voicing) in enumerate(zip(chord_progression, data["voicings"])):
                notes = [(note_data[0], note_data[1]) for note_data in voicing]
                octaves = [note_data[1] for note_data in voicing]
                register_range = (min(octaves), max(octaves))
                
                voiced_chord = VoicedChord(
                    chord_symbol=chord_symbol,
                    notes=notes,
                    register_range=register_range,
                    voice_leading_cost=0.0,  # Will be calculated
                    emotional_fitness=1.0    # Placeholder
                )
                voiced_chords.append(voiced_chord)
            
            # Calculate individual voice leading costs
            for i in range(1, len(voiced_chords)):
                prev_midi = [self._note_to_midi(note, octave) for note, octave in voiced_chords[i-1].notes]
                curr_midi = [self._note_to_midi(note, octave) for note, octave in voiced_chords[i].notes]
                cost = self._calculate_voice_distance(prev_midi, curr_midi)
                voiced_chords[i].voice_leading_cost = cost
            
            # Create result
            result = VoicingResult(
                voiced_chords=voiced_chords,
                total_voice_leading_cost=data["totalCost"],
                register_analysis={
                    "target_registers": data["targetRegisters"],
                    "average_register": sum(data["targetRegisters"]) / len(data["targetRegisters"])
                },
                harmonic_rhythm=data["harmonicAnalysis"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse Wolfram result: {e}")
            return self._create_fallback_voicing(chord_progression, {}, "C")
    
    def _create_fallback_voicing(
        self,
        chord_progression: List[str],
        emotion_weights: Dict[str, float],
        key: str
    ) -> VoicingResult:
        """Create fallback voicing when Wolfram engine fails"""
        logger.warning("Using fallback voicing due to Wolfram engine failure")
        
        # Simple fallback: root position in octave 4
        basic_voicings = {
            "I": [("C", 4), ("E", 4), ("G", 4)],
            "ii": [("D", 4), ("F", 4), ("A", 4)],
            "iii": [("E", 4), ("G", 4), ("B", 4)],
            "IV": [("F", 4), ("A", 4), ("C", 5)],
            "V": [("G", 4), ("B", 4), ("D", 5)],
            "vi": [("A", 4), ("C", 5), ("E", 5)],
            "vii°": [("B", 4), ("D", 5), ("F", 5)]
        }
        
        voiced_chords = []
        for chord in chord_progression:
            voicing = basic_voicings.get(chord, [("C", 4), ("E", 4), ("G", 4)])
            voiced_chord = VoicedChord(
                chord_symbol=chord,
                notes=voicing,
                register_range=(4, 5),
                voice_leading_cost=0.0,
                emotional_fitness=0.5
            )
            voiced_chords.append(voiced_chord)
        
        return VoicingResult(
            voiced_chords=voiced_chords,
            total_voice_leading_cost=0.0,
            register_analysis={"target_registers": [4, 5], "average_register": 4.5},
            harmonic_rhythm={"tensions": [0.5] * len(chord_progression), "durations": [1.0] * len(chord_progression)}
        )
    
    def _note_to_midi(self, note: str, octave: int) -> int:
        """Convert note name and octave to MIDI number"""
        note_values = {
            "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
            "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
        }
        return 12 * (octave + 1) + note_values.get(note, 0)
    
    def _calculate_voice_distance(self, voicing1: List[int], voicing2: List[int]) -> float:
        """Calculate voice movement distance between two voicings"""
        if len(voicing1) != len(voicing2):
            # Handle different chord sizes - simple approach
            min_len = min(len(voicing1), len(voicing2))
            voicing1 = voicing1[:min_len]
            voicing2 = voicing2[:min_len]
        
        return sum(abs(v2 - v1) for v1, v2 in zip(voicing1, voicing2))

class EnhancedVoiceLeadingEngine(WolframVoiceLeadingEngine):
    """Enhanced voice leading engine with additional features"""
    
    def __init__(self, wolfram_script_path: str = "wolframscript"):
        super().__init__(wolfram_script_path)
        self.cache = {}  # Simple caching for repeated calculations
    
    def optimize_with_style_context(
        self,
        chord_progression: List[str],
        emotion_weights: Dict[str, float],
        key: str = "C",
        style_context: str = "classical",
        register_preference: Optional[Dict[str, float]] = None
    ) -> VoicingResult:
        """
        Enhanced optimization with style context and register preferences
        
        Args:
            chord_progression: List of chord symbols
            emotion_weights: Dictionary of emotion names to weights
            key: Key signature
            style_context: Musical style context
            register_preference: Optional manual register preferences
            
        Returns:
            VoicingResult with style-aware optimized voicings
        """
        # Apply style-specific emotion weight adjustments
        adjusted_emotions = self._apply_style_adjustments(emotion_weights, style_context)
        
        # Apply register preferences if provided
        if register_preference:
            adjusted_emotions = self._apply_register_preferences(adjusted_emotions, register_preference)
        
        return self.optimize_voice_leading(
            chord_progression=chord_progression,
            emotion_weights=adjusted_emotions,
            key=key,
            style_context=style_context
        )
    
    def _apply_style_adjustments(self, emotion_weights: Dict[str, float], style: str) -> Dict[str, float]:
        """Apply style-specific emotion weight adjustments"""
        style_modifiers = {
            "classical": {"Reverence": 1.2, "Aesthetic Awe": 1.1, "Transcendence": 1.1},
            "jazz": {"Anticipation": 1.2, "Surprise": 1.1, "Wonder": 1.1},
            "blues": {"Sadness": 1.2, "Empowerment": 1.1, "Arousal": 1.1},
            "rock": {"Anger": 1.2, "Empowerment": 1.3, "Arousal": 1.2},
            "pop": {"Joy": 1.2, "Love": 1.1, "Trust": 1.1},
            "metal": {"Anger": 1.5, "Malice": 1.3, "Fear": 1.2},
            "experimental": {"Dissociation": 1.3, "Wonder": 1.2, "Surprise": 1.4}
        }
        
        modifiers = style_modifiers.get(style, {})
        adjusted = emotion_weights.copy()
        
        for emotion, modifier in modifiers.items():
            if emotion in adjusted:
                adjusted[emotion] *= modifier
        
        return adjusted
    
    def _apply_register_preferences(
        self, 
        emotion_weights: Dict[str, float], 
        register_prefs: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply manual register preferences to emotion weights"""
        # This is a simplified approach - in practice, you'd want more sophisticated logic
        adjusted = emotion_weights.copy()
        
        # Boost high-register emotions if high registers preferred
        if register_prefs.get("high", 0) > 0.5:
            high_register_emotions = ["Transcendence", "Aesthetic Awe", "Wonder", "Fear", "Anticipation"]
            for emotion in high_register_emotions:
                if emotion in adjusted:
                    adjusted[emotion] *= 1.2
        
        # Boost low-register emotions if low registers preferred
        if register_prefs.get("low", 0) > 0.5:
            low_register_emotions = ["Anger", "Malice", "Metal", "Disgust"]
            for emotion in low_register_emotions:
                if emotion in adjusted:
                    adjusted[emotion] *= 1.2
        
        return adjusted
    
    def analyze_progression_complexity(self, result: VoicingResult) -> Dict[str, float]:
        """Analyze the complexity of a voiced progression"""
        analysis = {
            "average_voice_leading_cost": result.total_voice_leading_cost / max(1, len(result.voiced_chords) - 1),
            "register_spread": 0.0,
            "harmonic_complexity": 0.0,
            "emotional_coherence": 0.0
        }
        
        # Calculate register spread
        all_octaves = []
        for chord in result.voiced_chords:
            all_octaves.extend([octave for _, octave in chord.notes])
        
        if all_octaves:
            analysis["register_spread"] = max(all_octaves) - min(all_octaves)
        
        # Calculate harmonic complexity based on chord types
        complex_chords = ["dim", "aug", "7", "9", "11", "13", "sus", "add"]
        complex_count = sum(1 for chord in result.voiced_chords 
                          if any(marker in chord.chord_symbol for marker in complex_chords))
        analysis["harmonic_complexity"] = complex_count / len(result.voiced_chords)
        
        # Calculate emotional coherence (placeholder)
        analysis["emotional_coherence"] = 0.8  # Would be calculated based on emotion mapping
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    # Test the voice leading engine
    engine = EnhancedVoiceLeadingEngine()
    
    # Example 1: Metal progression with anger
    metal_progression = ["i", "♭VII", "♭VI", "♯iv°"]
    metal_emotions = {"Anger": 0.8, "Malice": 0.6, "Empowerment": 0.4}
    
    print("=== Metal Progression Analysis ===")
    try:
        result = engine.optimize_with_style_context(
            chord_progression=metal_progression,
            emotion_weights=metal_emotions,
            key="Am",
            style_context="metal"
        )
        
        for i, chord in enumerate(result.voiced_chords):
            print(f"Chord {i+1}: {chord.chord_symbol}")
            print(f"  Voicing: {chord.notes}")
            print(f"  Register: {chord.register_range}")
            print(f"  Voice leading cost: {chord.voice_leading_cost}")
            print()
        
        print(f"Total voice leading cost: {result.total_voice_leading_cost}")
        print(f"Register analysis: {result.register_analysis}")
        
        complexity = engine.analyze_progression_complexity(result)
        print(f"Complexity analysis: {complexity}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This demo requires Wolfram Language/Mathematica to be installed") 