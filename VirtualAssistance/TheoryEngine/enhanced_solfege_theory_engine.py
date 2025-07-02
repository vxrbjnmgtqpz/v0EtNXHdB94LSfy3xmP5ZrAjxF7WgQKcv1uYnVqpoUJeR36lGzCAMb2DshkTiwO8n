#!/usr/bin/env python3
"""
Enhanced Python interface for the Wolfram Language Solfege Theory Engine
Supports multiple musical styles, emotion-based generation, and comprehensive analysis
"""

import subprocess
import json
import os
import tempfile
import logging
from typing import Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSolfegeTheoryEngine:
    """
    Enhanced Python interface to the Wolfram Language Solfege Theory Engine
    Supports multi-style generation, emotional mapping, and advanced analysis
    """
    
    def __init__(self, wolfram_script_path: Optional[str] = None):
        """Initialize the theory engine interface"""
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if wolfram_script_path is None:
            wolfram_script_path = os.path.join(self.script_dir, "EnhancedSolfegeTheoryEngine.wl")
        
        self.wolfram_script_path = wolfram_script_path
        
        if not os.path.exists(self.wolfram_script_path):
            raise FileNotFoundError(f"Wolfram script not found: {self.wolfram_script_path}")
        
        # Available styles and modes
        self.available_styles = [
            "Blues", "Jazz", "Classical", "Pop", "Rock", "Folk", "RnB", "Cinematic"
        ]
        
        self.available_modes = [
            "Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"
        ]
        
        # Emotion to style mapping (matches Wolfram implementation)
        self.emotion_style_map = {
            "happy": "Pop",
            "sad": "Folk", 
            "energetic": "Rock",
            "peaceful": "Classical",
            "romantic": "Jazz",
            "melancholy": "Blues",
            "dramatic": "Cinematic",
            "soulful": "RnB",
            "nostalgic": "Folk",
            "uplifting": "Pop",
            "contemplative": "Classical",
            "passionate": "Jazz"
        }
        
        logger.info("Enhanced Solfege Theory Engine initialized")
        logger.info(f"Available styles: {self.available_styles}")
        logger.info(f"Available modes: {self.available_modes}")
    
    def _execute_wolfram_function(self, function_call: str) -> Dict[str, Any]:
        """Execute a function call in the Wolfram engine and return results"""
        
        # Create a temporary Wolfram script
        temp_script = f"""
        SetDirectory["{self.script_dir}"];
        Get["{self.wolfram_script_path}"];
        
        result = {function_call};
        
        (* Convert result to JSON-compatible format *)
        jsonResult = If[AssociationQ[result],
            ExportString[result, "JSON"],
            ExportString[Association["result" -> result], "JSON"]
        ];
        
        Print["JSON_OUTPUT_START"];
        Print[jsonResult];
        Print["JSON_OUTPUT_END"];
        """
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.wl', delete=False) as f:
                f.write(temp_script)
                temp_file = f.name
            
            # Execute with WolframScript
            result = subprocess.run(
                ['wolframscript', '-file', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode != 0:
                logger.error(f"Wolfram execution failed: {result.stderr}")
                return {"error": result.stderr}
            
            # Extract JSON output
            output_lines = result.stdout.split('\n')
            json_start = None
            json_end = None
            
            for i, line in enumerate(output_lines):
                if "JSON_OUTPUT_START" in line:
                    json_start = i + 1
                elif "JSON_OUTPUT_END" in line:
                    json_end = i
                    break
            
            if json_start is not None and json_end is not None:
                json_lines = output_lines[json_start:json_end]
                json_str = '\n'.join(json_lines)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    return {"error": f"JSON parsing failed: {e}"}
            
            return {"error": "No JSON output found"}
            
        except subprocess.TimeoutExpired:
            return {"error": "Wolfram execution timed out"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    
    def generate_midi_chord(self, mode: str, chord_symbol: str, root_note: int = 60) -> List[int]:
        """Generate MIDI notes for a chord in solfege system"""
        
        if mode not in self.available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available: {self.available_modes}")
        
        function_call = f'generateMIDIChord["{mode}", "{chord_symbol}", {root_note}]'
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error generating MIDI chord: {result['error']}")
            return []
        
        return result.get("result", [])
    
    def generate_style_progression(self, style: str, mode: str, length: int = 4, root_note: int = 60) -> List[str]:
        """Generate a chord progression for a specific style and mode"""
        
        if style not in self.available_styles:
            raise ValueError(f"Invalid style: {style}. Available: {self.available_styles}")
        
        if mode not in self.available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available: {self.available_modes}")
        
        function_call = f'generateStyleProgression["{style}", "{mode}", {length}, {root_note}]'
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error generating style progression: {result['error']}")
            return []
        
        return result.get("result", [])
    
    def generate_emotional_progression(self, emotion: str, mode: str = "Ionian", 
                                     length: int = 4, root_note: int = 60) -> Dict[str, Any]:
        """Generate a chord progression based on emotion"""
        
        if mode not in self.available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available: {self.available_modes}")
        
        function_call = f'generateEmotionalProgression["{emotion}", "{mode}", {length}, {root_note}]'
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error generating emotional progression: {result['error']}")
            return {"error": result["error"]}
        
        return result
    
    def progression_to_midi(self, progression: List[str], mode: str, root_note: int = 60) -> List[List[int]]:
        """Convert chord symbols to MIDI note arrays"""
        
        if mode not in self.available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available: {self.available_modes}")
        
        # Convert Python list to Wolfram list format
        wolfram_list = "{" + ", ".join([f'"{chord}"' for chord in progression]) + "}"
        function_call = f'progressionToMIDI[{wolfram_list}, "{mode}", {root_note}]'
        
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error converting progression to MIDI: {result['error']}")
            return []
        
        return result.get("result", [])
    
    def analyze_progression(self, progression: List[str], mode: str) -> Dict[str, Any]:
        """Analyze a chord progression for functional harmony and structure"""
        
        if mode not in self.available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available: {self.available_modes}")
        
        # Convert Python list to Wolfram list format
        wolfram_list = "{" + ", ".join([f'"{chord}"' for chord in progression]) + "}"
        function_call = f'analyzeChordProgression[{wolfram_list}, "{mode}"]'
        
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error analyzing progression: {result['error']}")
            return {"error": result["error"]}
        
        return result
    
    def get_emotion_style(self, emotion: str) -> str:
        """Get the musical style associated with an emotion"""
        return self.emotion_style_map.get(emotion.lower(), "Classical")
    
    def demonstrate(self) -> Dict[str, Any]:
        """Run the demonstration function"""
        
        function_call = 'demonstrateTheoryEngine[]'
        result = self._execute_wolfram_function(function_call)
        
        if "error" in result:
            logger.error(f"Error running demonstration: {result['error']}")
            return {"error": result["error"]}
        
        return result
    
    def generate_multi_style_comparison(self, emotion: str, length: int = 4) -> Dict[str, Any]:
        """Generate the same emotion in different styles for comparison"""
        
        results = {}
        primary_style = self.get_emotion_style(emotion)
        
        # Generate in primary style
        results["primary"] = self.generate_emotional_progression(emotion, "Ionian", length)
        
        # Generate in a few other styles for comparison
        comparison_styles = ["Jazz", "Rock", "Blues", "Classical"]
        comparison_styles = [s for s in comparison_styles if s != primary_style][:3]
        
        results["comparisons"] = {}
        for style in comparison_styles:
            prog = self.generate_style_progression(style, "Ionian", length)
            midi = self.progression_to_midi(prog, "Ionian")
            
            results["comparisons"][style] = {
                "style": style,
                "chordSymbols": prog,
                "midiChords": midi
            }
        
        return results
    
    def generate_legal_progression(self, style: str, mode: str = "Ionian", 
                                 length: int = 4, start_chord: str = "auto") -> List[str]:
        """
        Generate a style-aware progression using comprehensive legality rules
        
        Args:
            style: Musical style (Blues, Jazz, Classical, Pop, Rock, Folk, RnB, Cinematic)
            mode: Musical mode (Ionian, Dorian, etc.)
            length: Number of chords in progression
            start_chord: Starting chord ("auto" for intelligent selection)
            
        Returns:
            List of chord symbols
        """
        if style not in self.available_styles:
            raise ValueError(f"Style must be one of: {self.available_styles}")
        
        if mode not in self.available_modes:
            raise ValueError(f"Mode must be one of: {self.available_modes}")
        
        try:
            # Call Wolfram engine
            function_call = f'generateLegalProgression["{style}", "{mode}", {length}, "{start_chord}"]'
            
            result = self._execute_wolfram_function(function_call)
            
            # Parse result - handle both direct list and wrapped format
            if "error" in result:
                logger.error(f"Error from Wolfram: {result['error']}")
                return []
            elif "result" in result and isinstance(result["result"], list):
                return [str(chord).strip() for chord in result["result"]]
            elif isinstance(result, list):
                return [str(chord).strip() for chord in result]
            else:
                logger.warning(f"Unexpected result format from legal progression generation: {result}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating legal progression: {e}")
            return []
    
    def compare_style_progressions(self, mode: str = "Ionian", length: int = 4) -> Dict[str, List[str]]:
        """
        Generate progressions for all styles in the same mode for comparison
        
        Args:
            mode: Musical mode to use for all styles
            length: Number of chords per progression
            
        Returns:
            Dictionary mapping style names to chord progressions
        """
        try:
            function_call = f'compareStyleProgressions["{mode}", {length}]'
            
            result = self._execute_wolfram_function(function_call)
            
            # Parse the association result
            if "error" in result:
                logger.error(f"Error from Wolfram: {result['error']}")
                return {}
            elif "result" in result and isinstance(result["result"], dict):
                return {style: [str(chord).replace('Â°', '°').replace('â°', '°') for chord in progression] 
                       for style, progression in result["result"].items() if progression}
            elif isinstance(result, dict) and not "error" in result:
                return {style: [str(chord).replace('Â°', '°').replace('â°', '°') for chord in progression] 
                       for style, progression in result.items() if progression}
            else:
                logger.warning(f"Unexpected result format from style comparison: {result}")
                return {}
                
        except Exception as e:
            logger.error(f"Error comparing style progressions: {e}")
            return {}

def main():
    """Demo the enhanced theory engine"""
    print("=== Enhanced Solfege Theory Engine Demo ===")
    
    try:
        engine = EnhancedSolfegeTheoryEngine()
        
        # Test basic chord generation
        print("\n1. Basic chord generation:")
        chord = engine.generate_midi_chord("Ionian", "I", 60)
        print(f"I chord in Ionian: {chord}")
        
        # Test style progression
        print("\n2. Style-specific progression:")
        jazz_prog = engine.generate_style_progression("Jazz", "Dorian", 4, 60)
        print(f"Jazz progression in Dorian: {jazz_prog}")
        
        # Test emotional progression
        print("\n3. Emotion-based progression:")
        happy_prog = engine.generate_emotional_progression("happy", "Ionian", 4, 60)
        print(f"Happy progression: {happy_prog}")
        
        # Test multi-style comparison
        print("\n4. Multi-style comparison:")
        comparison = engine.generate_multi_style_comparison("romantic", 4)
        print(f"Romantic emotion in different styles:")
        if "primary" in comparison:
            print(f"  Primary: {comparison['primary'].get('chordSymbols', [])}")
        for style, data in comparison.get("comparisons", {}).items():
            print(f"  {style}: {data.get('chordSymbols', [])}")
        
        # Test legal progression generation
        print("\n5. Legal progression generation:")
        legal_prog = engine.generate_legal_progression("Jazz", "Ionian", 4)
        print(f"Legal progression in Jazz, Ionian: {legal_prog}")
        
        # Test style comparison progression
        print("\n6. Style comparison progression:")
        comparison_results = engine.compare_style_progressions("Ionian", 4)
        for style, progression in comparison_results.items():
            print(f"  {style}: {progression}")
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")

if __name__ == "__main__":
    main()
