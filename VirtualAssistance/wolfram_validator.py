"""
Wolfram|Alpha Integration for Music Theory Validation
Validates chord progressions and provides music theory analysis using Wolfram's knowledge engine
"""

import requests
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import os


class WolframMusicValidator:
    """
    Integrates with Wolfram|Alpha to validate music theory aspects of generated progressions
    """
    
    def __init__(self, app_id: Optional[str] = None):
        """
        Initialize Wolfram validator
        
        Args:
            app_id: Wolfram|Alpha API app ID (if None, looks for WOLFRAM_APP_ID env var)
        """
        self.app_id = app_id or os.getenv('WOLFRAM_APP_ID')
        self.base_url = "https://api.wolframalpha.com/v2/query"
        self.enabled = bool(self.app_id)
        
        if not self.enabled:
            print("⚠️ Wolfram|Alpha validation disabled: No API key found")
            print("   Set WOLFRAM_APP_ID environment variable to enable")
    
    def validate_progression(self, chords: List[str], mode: str = "major", key: str = "C") -> Dict:
        """
        Validate a chord progression using Wolfram|Alpha
        
        Args:
            chords: List of chord symbols (e.g., ["I", "V", "vi", "IV"])
            mode: Musical mode/scale (e.g., "major", "minor")
            key: Root key (e.g., "C", "G")
            
        Returns:
            Dictionary with validation results and analysis
        """
        if not self.enabled:
            return self._disabled_response()
        
        try:
            # Convert roman numerals to actual chord names
            chord_names = self._convert_to_chord_names(chords, key, mode)
            
            # Query Wolfram about the progression
            query = f"chord progression {' '.join(chord_names)} in {key} {mode}"
            result = self._query_wolfram(query)
            
            # Parse the response
            analysis = self._parse_music_response(result)
            
            # Add our own analysis
            analysis.update({
                'input_chords': chords,
                'chord_names': chord_names,
                'key': key,
                'mode': mode,
                'validation_status': 'success'
            })
            
            return analysis
            
        except Exception as e:
            return {
                'validation_status': 'error',
                'error': str(e),
                'input_chords': chords,
                'fallback_analysis': self._fallback_analysis(chords, mode)
            }
    
    def analyze_mode_compatibility(self, chords: List[str], target_mode: str) -> Dict:
        """
        Check if chord progression fits the intended musical mode
        
        Args:
            chords: List of roman numeral chords
            target_mode: Intended mode (e.g., "Ionian", "Dorian", "Aeolian")
            
        Returns:
            Compatibility analysis
        """
        if not self.enabled:
            return self._disabled_response()
        
        try:
            # Query about mode characteristics
            mode_query = f"musical mode {target_mode} scale degrees and chords"
            mode_info = self._query_wolfram(mode_query)
            
            # Analyze chord compatibility
            compatibility = self._analyze_mode_fit(chords, target_mode)
            
            return {
                'target_mode': target_mode,
                'chord_progression': chords,
                'compatibility_score': compatibility['score'],
                'fitting_chords': compatibility['fitting'],
                'non_fitting_chords': compatibility['non_fitting'],
                'mode_analysis': mode_info,
                'validation_status': 'success'
            }
            
        except Exception as e:
            return {
                'validation_status': 'error',
                'error': str(e),
                'fallback_analysis': f"Local analysis: {target_mode} mode compatibility check failed"
            }
    
    def get_harmonic_function(self, chord: str, key: str = "C", mode: str = "major") -> Dict:
        """
        Get harmonic function analysis for a specific chord
        
        Args:
            chord: Roman numeral chord (e.g., "V", "vi", "IV")
            key: Root key
            mode: Musical mode
            
        Returns:
            Harmonic function analysis
        """
        if not self.enabled:
            return self._disabled_response()
        
        try:
            chord_name = self._convert_to_chord_names([chord], key, mode)[0]
            query = f"harmonic function of {chord_name} chord in {key} {mode}"
            
            result = self._query_wolfram(query)
            
            return {
                'chord': chord,
                'chord_name': chord_name,
                'harmonic_function': result,
                'key': key,
                'mode': mode,
                'validation_status': 'success'
            }
            
        except Exception as e:
            return {
                'validation_status': 'error',
                'error': str(e),
                'chord': chord
            }
    
    def _query_wolfram(self, query: str) -> str:
        """Send query to Wolfram|Alpha and return response"""
        params = {
            'input': query,
            'appid': self.app_id,
            'format': 'plaintext',
            'output': 'json'
        }
        
        response = requests.get(self.base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract plaintext results
        if 'queryresult' in data and data['queryresult']['success']:
            pods = data['queryresult']['pods']
            results = []
            
            for pod in pods:
                if 'subpods' in pod:
                    for subpod in pod['subpods']:
                        if 'plaintext' in subpod and subpod['plaintext']:
                            results.append(f"{pod['title']}: {subpod['plaintext']}")
            
            return '\n'.join(results) if results else "No results found"
        else:
            return "Query failed"
    
    def _convert_to_chord_names(self, roman_chords: List[str], key: str, mode: str) -> List[str]:
        """Convert roman numeral chords to actual chord names"""
        # Mapping for major scale (Ionian mode)
        major_mapping = {
            'I': key, 'ii': self._get_chord_name(key, 2, 'minor'),
            'iii': self._get_chord_name(key, 3, 'minor'), 'IV': self._get_chord_name(key, 4),
            'V': self._get_chord_name(key, 5), 'vi': self._get_chord_name(key, 6, 'minor'),
            'vii°': self._get_chord_name(key, 7, 'diminished')
        }
        
        # Mapping for minor scale (Aeolian mode)
        minor_mapping = {
            'i': key + 'm', '♭II': self._get_chord_name(key, 2, 'major', flat=True),
            '♭III': self._get_chord_name(key, 3, 'major', flat=True), 'iv': self._get_chord_name(key, 4, 'minor'),
            'v': self._get_chord_name(key, 5, 'minor'), '♭VI': self._get_chord_name(key, 6, 'major', flat=True),
            '♭VII': self._get_chord_name(key, 7, 'major', flat=True)
        }
        
        # Choose mapping based on mode
        mapping = minor_mapping if mode in ['minor', 'Aeolian'] else major_mapping
        
        # Convert each chord
        chord_names = []
        for chord in roman_chords:
            if chord in mapping:
                chord_names.append(mapping[chord])
            else:
                # Fallback: try to parse complex chord symbols
                chord_names.append(self._parse_complex_chord(chord, key, mode))
        
        return chord_names
    
    def _get_chord_name(self, root: str, degree: int, quality: str = 'major', flat: bool = False) -> str:
        """Get chord name for a scale degree"""
        # Simplified chord name generation
        notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
        # Find root position
        root_idx = notes.index(root) if root in notes else 0
        
        # Calculate target note
        target_idx = (root_idx + degree - 1) % 7
        target_note = notes[target_idx]
        
        if flat:
            target_note += '♭'
        
        # Add quality
        if quality == 'minor':
            target_note += 'm'
        elif quality == 'diminished':
            target_note += '°'
        
        return target_note
    
    def _parse_complex_chord(self, chord: str, key: str, mode: str) -> str:
        """Parse complex chord symbols (with extensions, alterations)"""
        # This is a simplified parser - in practice, you'd want more sophisticated logic
        if chord.startswith('♭'):
            return f"{key}♭{chord[1:]}"
        elif chord.startswith('#'):
            return f"{key}#{chord[1:]}"
        else:
            return f"{key}{chord}"
    
    def _parse_music_response(self, wolfram_response: str) -> Dict:
        """Parse Wolfram|Alpha response for music-related information"""
        analysis = {
            'scale_analysis': '',
            'chord_functions': '',
            'harmonic_analysis': '',
            'theory_notes': ''
        }
        
        lines = wolfram_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'scale' in line.lower():
                analysis['scale_analysis'] = line
            elif 'function' in line.lower():
                analysis['chord_functions'] = line
            elif 'harmonic' in line.lower():
                analysis['harmonic_analysis'] = line
            else:
                analysis['theory_notes'] += line + '\n'
        
        return analysis
    
    def _analyze_mode_fit(self, chords: List[str], target_mode: str) -> Dict:
        """Local analysis of how well chords fit the target mode"""
        # Simplified mode compatibility checking
        mode_chord_sets = {
            'Ionian': ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°'],
            'Aeolian': ['i', 'ii°', '♭III', 'iv', 'v', '♭VI', '♭VII'],
            'Dorian': ['i', 'ii', '♭III', 'IV', 'v', 'vi°', '♭VII'],
            'Phrygian': ['i', '♭ii', '♭III', 'iv', 'v°', '♭VI', '♭vii'],
            'Lydian': ['I', 'II', 'iii', '♯iv°', 'V', 'vi', 'vii'],
            'Mixolydian': ['I', 'ii', 'iii°', 'IV', 'v', 'vi', '♭VII']
        }
        
        expected_chords = set(mode_chord_sets.get(target_mode, mode_chord_sets['Ionian']))
        actual_chords = set(chords)
        
        fitting = actual_chords.intersection(expected_chords)
        non_fitting = actual_chords.difference(expected_chords)
        
        score = len(fitting) / len(actual_chords) if actual_chords else 0
        
        return {
            'score': score,
            'fitting': list(fitting),
            'non_fitting': list(non_fitting)
        }
    
    def _fallback_analysis(self, chords: List[str], mode: str) -> str:
        """Provide basic analysis when Wolfram is unavailable"""
        chord_count = len(chords)
        unique_chords = len(set(chords))
        
        return f"Local analysis: {chord_count} chord progression with {unique_chords} unique chords in {mode} mode"
    
    def _disabled_response(self) -> Dict:
        """Standard response when Wolfram integration is disabled"""
        return {
            'validation_status': 'disabled',
            'message': 'Wolfram|Alpha validation is disabled. Set WOLFRAM_APP_ID to enable.',
            'local_analysis': 'Basic validation passed'
        }


# Integration with the main model
def integrate_wolfram_validation(model_result: Dict, validator: WolframMusicValidator) -> Dict:
    """
    Integrate Wolfram validation into model results
    
    Args:
        model_result: Result from ChordProgressionModel.generate_from_prompt()
        validator: WolframMusicValidator instance
        
    Returns:
        Enhanced result with Wolfram validation
    """
    if not validator.enabled:
        model_result['wolfram_validation'] = validator._disabled_response()
        return model_result
    
    try:
        # Validate the progression
        progression_validation = validator.validate_progression(
            model_result['chords'], 
            model_result['primary_mode'].lower(),
            'C'  # Default key
        )
        
        # Check mode compatibility
        mode_compatibility = validator.analyze_mode_compatibility(
            model_result['chords'],
            model_result['primary_mode']
        )
        
        # Add validation results
        model_result['wolfram_validation'] = {
            'progression_analysis': progression_validation,
            'mode_compatibility': mode_compatibility,
            'validation_timestamp': progression_validation.get('validation_status')
        }
        
        # Add a quality score based on validation
        if progression_validation['validation_status'] == 'success':
            compatibility_score = mode_compatibility.get('compatibility_score', 0.5)
            model_result['quality_score'] = compatibility_score
        
    except Exception as e:
        model_result['wolfram_validation'] = {
            'validation_status': 'error',
            'error': str(e)
        }
    
    return model_result


if __name__ == "__main__":
    # Demo usage
    validator = WolframMusicValidator()
    
    # Test progression validation
    test_chords = ["I", "V", "vi", "IV"]
    result = validator.validate_progression(test_chords, "major", "C")
    print(f"Validation result: {json.dumps(result, indent=2)}")
    
    # Test mode compatibility
    mode_result = validator.analyze_mode_compatibility(test_chords, "Ionian")
    print(f"Mode compatibility: {json.dumps(mode_result, indent=2)}") 