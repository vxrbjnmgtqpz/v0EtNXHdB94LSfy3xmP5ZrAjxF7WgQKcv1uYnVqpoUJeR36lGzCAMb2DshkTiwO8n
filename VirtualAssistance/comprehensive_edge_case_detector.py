#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE EDGE CASE DETECTOR
Systematically tests all aspects of the VirtualAssistance Model Stack
to identify edge cases, boundary conditions, and potential failure modes.
"""

import sys
import json
import traceback
from typing import List, Dict, Any, Tuple
import requests
import time
import gc
from datetime import datetime

# Add current directory to path for imports
sys.path.append('.')

class EdgeCaseDetector:
    """Comprehensive edge case detection for the music generation system"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'critical_errors': [],
            'warnings': [],
            'test_details': {}
        }
        
        print("🔍 Initializing Comprehensive Edge Case Detector...")
        
        # Initialize models safely
        try:
            from chord_progression_model import ChordProgressionModel
            from individual_chord_model import IndividualChordModel
            self.progression_model = ChordProgressionModel()
            self.individual_model = IndividualChordModel()
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            self.progression_model = None
            self.individual_model = None
            
        self.server_url = "http://localhost:5004"
        
    def log_test_result(self, test_name: str, passed: bool, error: str = None, details: Dict = None):
        """Log the result of a test."""
        self.results['total_tests'] += 1
        if passed:
            self.results['passed'] += 1
            print(f"  ✅ PASSED: {test_name}")
        else:
            self.results['failed'] += 1
            if error:
                if 'ValueError' in error or 'KeyError' in error or 'TypeError' in error:
                    self.results['critical_errors'].append({
                        'test': test_name,
                        'error': error,
                        'details': details
                    })
                    print(f"  💥 CRITICAL ERROR: {test_name} - {error}")
                else:
                    self.results['warnings'].append({
                        'test': test_name,
                        'error': error,
                        'details': details
                    })
                    print(f"  ⚠️ WARNING: {test_name} - {error}")
            else:
                print(f"  ❌ FAILED: {test_name}")
        
        self.results['test_details'][test_name] = {
            'passed': passed,
            'error': error,
            'details': details
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        try:
            print(f"\n🔍 Testing: {test_name}")
            result = test_func()
            if result is True:
                self.log_test_result(test_name, True)
            elif result is False:
                self.log_test_result(test_name, False)
            else:
                # Handle complex return values
                self.log_test_result(test_name, True, details={'result': str(result)})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log_test_result(test_name, False, error_msg, {
                'traceback': traceback.format_exc()
            })
    
    def test_input_validation_edge_cases(self):
        """Test edge cases in input validation"""
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a" * 1000,  # Very long string
            "🎵🎶🎵🎶🎵",  # Emoji only
            "I feel 💀☠️🔥",  # Mixed emoji and text
            "NULL",  # String that looks like null
            "undefined",  # String that looks like undefined
            "{{malicious}}",  # Template injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "\n\r\t",  # Control characters
            "feeling" * 100,  # Repeated words
        ]
        
        for i, test_input in enumerate(edge_cases):
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(test_input)
                    if result and len(result) > 0:
                        chords = result[0].get('chords', [])
                        if len(chords) == 4:  # Expected length
                            self.log_test_result(f"input_validation_{i}", True, message=f"Handled malformed input correctly: '{test_input[:30]}...'")
                        else:
                            self.log_test_result(f"input_validation_{i}", False, error=f"Unexpected chord count {len(chords)} for input: '{test_input[:30]}...'")
                    else:
                        self.log_test_result(f"input_validation_{i}", False, error="Empty result for input")
                else:
                    self.log_test_result(f"input_validation_{i}", False, error="Model not available for testing")
            except Exception as e:
                self.log_test_result(f"input_validation_{i}", False, error=f"Failed on input '{test_input[:30]}...': {e}")
    
    def test_chord_mapping_edge_cases(self):
        """Test edge cases in chord symbol mapping"""
        edge_chord_symbols = [
            # Extreme extensions
            "IM7♯11♯9♭13",
            "V7alt",
            "♭♭VII",  # Double flat
            "♯♯IV",   # Double sharp
            "XIII",   # Roman numeral beyond octave
            "♭IXsus4add9",  # Complex combination
            
            # Malformed symbols
            "I/",     # Trailing slash
            "/V",     # Leading slash
            "I//V",   # Double slash
            "I-V",    # Hyphen instead of slash
            "i(add♯9)", # Parentheses
            
            # Unicode variants
            "♭VII",   # Normal flat
            "bVII",   # ASCII b instead of ♭
            "VII♭",   # Flat after numeral
            
            # Case sensitivity issues
            "Imaj7",
            "IMAJ7", 
            "imaj7",
            
            # Spacing issues
            "I maj7",
            "I  maj7",
            "Imaj 7",
        ]
        
        for chord in edge_chord_symbols:
            try:
                # Test if this chord can be processed
                test_input = f"I want a progression with {chord} chord"
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(test_input)
                    if result:
                        self.log_test_result(f"chord_edge_{hash(chord)}", True, message=f"Processed chord reference: {chord}")
                    else:
                        self.log_test_result(f"chord_edge_{hash(chord)}", False, error="No result for chord")
                else:
                    self.log_test_result(f"chord_edge_{hash(chord)}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"chord_edge_{hash(chord)}", False, error=f"Exception with chord {chord}: {e}")
    
    def test_emotion_parsing_edge_cases(self):
        """Test edge cases in emotion parsing"""
        edge_emotion_inputs = [
            # Contradictory emotions
            "I feel happy and sad simultaneously",
            "joyful but also filled with rage",
            "peaceful yet aggressive",
            
            # Extreme intensity
            "EXTREMELY ABSOLUTELY UTTERLY DEVASTATED",
            "so happy i could die",
            "infinitely sad beyond measure",
            
            # Metaphorical emotions
            "feeling like a storm",
            "my heart is a black hole",
            "emotions are a rainbow",
            
            # Temporal emotions
            "I was happy but now I'm sad",
            "feeling nostalgic about future possibilities",
            
            # Abstract concepts
            "feeling the heat death of the universe",
            "emotionally quantum entangled",
            "vibrating at the frequency of cosmic dread",
            
            # Sub-emotion edge cases
            "malicious but not quite sadistic",
            "happy but not excited or euphoric",
            "sad but specifically melancholic not depressed",
        ]
        
        for emotion_input in edge_emotion_inputs:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(emotion_input)
                    if result and len(result) > 0:
                        detected_emotions = result[0].get('emotion_weights', {})
                        primary_emotion = max(detected_emotions, key=detected_emotions.get) if detected_emotions else "None"
                        chords = result[0].get('chords', [])
                        self.log_test_result(f"emotion_parsing_{hash(emotion_input)}", True, message=f"Parsed '{emotion_input[:30]}...' → {primary_emotion}, {len(chords)} chords")
                    else:
                        self.log_test_result(f"emotion_parsing_{hash(emotion_input)}", False, error="No result for emotion")
                else:
                    self.log_test_result(f"emotion_parsing_{hash(emotion_input)}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"emotion_parsing_{hash(emotion_input)}", False, error=f"Failed parsing '{emotion_input[:30]}...': {e}")
    
    def test_boundary_condition_edge_cases(self):
        """Test boundary conditions"""
        boundary_tests = [
            # Genre boundaries
            ("Unknown genre", "happy", "QuantumJazz"),
            ("Empty genre", "sad", ""),
            ("Very long genre", "angry", "Post-Progressive-Neo-Classical-Fusion-Metal"),
            
            # Emotion weight boundaries
            ("All emotions", "happy sad angry fearful disgusted surprised trusting anticipating shameful loving envious aesthetic malicious", "Pop"),
            ("No emotion words", "the quick brown fox jumps over the lazy dog", "Pop"),
            ("Only intensifiers", "very extremely absolutely completely utterly", "Pop"),
        ]
        
        for test_name, emotion, genre in boundary_tests:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(emotion, genre, 1)
                    if result:
                        chords = result[0].get('chords', [])
                        if len(chords) == 4:  # Expected length
                            self.log_test_result(f"boundary_{test_name}", True, message=f"Handled boundary case correctly: {len(chords)} chords")
                        else:
                            self.log_test_result(f"boundary_{test_name}", False, error=f"Unexpected chord count: {len(chords)}")
                    else:
                        self.log_test_result(f"boundary_{test_name}", False, error="No result generated")
                else:
                    self.log_test_result(f"boundary_{test_name}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"boundary_{test_name}", False, error=f"Boundary test failed: {e}")
    
    def test_malformed_input_edge_cases(self):
        """Test malformed and unusual inputs"""
        malformed_inputs = [
            # JSON-like strings
            '{"emotion": "happy", "genre": "pop"}',
            "['I', 'V', 'vi', 'IV']",
            
            # Code-like strings
            "function generateMusic() { return 'happy'; }",
            "SELECT * FROM emotions WHERE mood = 'sad';",
            
            # Mathematical expressions
            "happiness = joy + excitement - anxiety",
            "∑(sadness) + ∫(melancholy)dx",
            
            # Musical notation
            "C-Am-F-G",
            "I-vi-IV-V in C major",
            "♪♫♪♫ happy music ♪♫♪♫",
            
            # URLs and paths
            "http://example.com/emotion/happy",
            "/usr/bin/happy",
            "C:\\emotions\\sadness.exe",
        ]
        
        for malformed_input in malformed_inputs:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(malformed_input)
                    if result:
                        chords = result[0].get('chords', [])
                        self.log_test_result(f"malformed_{hash(malformed_input)}", True, message=f"Handled malformed input: '{malformed_input[:30]}...', got {len(chords)} chords")
                    else:
                        self.log_test_result(f"malformed_{hash(malformed_input)}", False, error="No result for malformed input")
                else:
                    self.log_test_result(f"malformed_{hash(malformed_input)}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"malformed_{hash(malformed_input)}", False, error=f"Failed on malformed input '{malformed_input[:30]}...': {e}")
    
    def test_unusual_chord_symbols(self):
        """Test handling of unusual chord symbols that might be generated"""
        unusual_chord_tests = [
            "I want music with a ♭♭♭VII chord",  # Triple flat
            "Create progression with ♯♯♯IV",      # Triple sharp
            "Generate chords including XXIII",    # Roman numeral beyond range
            "Music with √I chord",                # Mathematical symbols
            "Progression with ∞V",               # Infinity symbol
            "I need a πIII chord",               # Pi symbol
        ]
        
        for test_input in unusual_chord_tests:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(test_input)
                    if result:
                        chords = result[0].get('chords', [])
                        self.log_test_result(f"unusual_chord_{hash(test_input)}", True, message=f"Handled unusual chord request: {len(chords)} chords generated")
                    else:
                        self.log_test_result(f"unusual_chord_{hash(test_input)}", False, error="No result for unusual chord request")
                else:
                    self.log_test_result(f"unusual_chord_{hash(test_input)}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"unusual_chord_{hash(test_input)}", False, error=f"Exception with unusual chord request: {e}")
    
    def test_extreme_emotion_combinations(self):
        """Test extreme combinations of emotions and sub-emotions"""
        extreme_combinations = [
            # All Malice sub-emotions at once
            "I feel cruel sadistic vengeful callous manipulative and dominating",
            
            # Opposite emotion pairs
            "extremely joyful but also devastatingly sad",
            "trusting but also fearful",
            "loving but also envious",
            
            # All 13 core emotions
            "I feel joy sadness fear anger disgust surprise trust anticipation shame love envy aesthetic awe and malice",
            
            # Rapid emotion switching
            "happy sad happy sad happy sad happy sad",
            
            # Emotion intensity overload
            "EXTREMELY UTTERLY COMPLETELY ABSOLUTELY DEVASTATINGLY CRUSHINGLY OVERWHELMINGLY sad",
            
            # Meta-emotions
            "I feel emotional about feeling emotions",
            "anxious about being anxious about anxiety",
        ]
        
        for emotion_combo in extreme_combinations:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(emotion_combo)
                    if result:
                        emotion_weights = result[0].get('emotion_weights', {})
                        active_emotions = [e for e, w in emotion_weights.items() if w > 0.1]
                        chords = result[0].get('chords', [])
                        self.log_test_result(f"extreme_emotion_{hash(emotion_combo)}", True, message=f"Complex emotions: {len(active_emotions)} detected, {len(chords)} chords")
                    else:
                        self.log_test_result(f"extreme_emotion_{hash(emotion_combo)}", False, error="No result for complex emotions")
                else:
                    self.log_test_result(f"extreme_emotion_{hash(emotion_combo)}", False, error="Model not available")
            except Exception as e:
                self.log_test_result(f"extreme_emotion_{hash(emotion_combo)}", False, error=f"Failed on complex emotions: {e}")
    
    def test_server_stress_conditions(self):
        """Test server under stress conditions"""
        if not self.check_server_availability():
            self.log_test_result("server_stress", False, error="Server not available for stress testing")
            return
        
        # Rapid fire requests
        try:
            responses = []
            for i in range(5):  # Reduced for safety
                response = requests.post(f"{self.server_url}/chat/integrated", 
                                       json={"message": f"test message {i}"}, 
                                       timeout=10)
                responses.append(response.status_code)
                time.sleep(0.5)  # Small delay
            
            success_rate = sum(1 for r in responses if r == 200) / len(responses)
            if success_rate > 0.8:
                self.log_test_result("server_stress_rapid", True, message=f"Handled rapid requests: {success_rate*100:.1f}% success")
            else:
                self.log_test_result("server_stress_rapid", False, error=f"Low success rate: {success_rate*100:.1f}%")
                
        except Exception as e:
            self.log_test_result("server_stress_rapid", False, error=f"Rapid fire test failed: {e}")
    
    def test_audio_mapping_completeness(self):
        """Test completeness of audio mapping system"""
        # Test the chords we know should work
        essential_chords = [
            "I", "ii", "iii", "IV", "V", "vi", "vii°",
            "i", "♭III", "iv", "v", "♭VI", "♭VII",
            "IM7", "V7", "ii7", "vi7",
            "I6", "V6", "ii6", "vi6",
            "♯IV", "♯V", "♯vi"  # The ones we just added
        ]
        
        unmapped_chords = []
        for chord in essential_chords:
            try:
                # Test if these chords appear in actual generation
                test_input = f"I want a progression with {chord}"
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(test_input)
                    if result:
                        # Check if the generated progression is reasonable
                        chords = result[0].get('chords', [])
                        if len(chords) != 4:
                            unmapped_chords.append(f"{chord} (wrong length: {len(chords)})")
            except Exception as e:
                unmapped_chords.append(f"{chord} (error: {str(e)[:50]})")
        
        if not unmapped_chords:
            self.log_test_result("audio_mapping_complete", True, message="All essential chords processed correctly")
        else:
            self.log_test_result("audio_mapping_issues", False, error=f"Chord issues: {unmapped_chords[:5]}")  # Show first 5
    
    def test_progression_length_consistency(self):
        """Test that progression lengths are consistent"""
        test_inputs = [
            "happy music",
            "sad melody", 
            "angry progression",
            "malicious and evil",
            "I feel cruel and sadistic",
            "joyful and excited"
        ]
        
        lengths = []
        for test_input in test_inputs:
            try:
                if self.progression_model:
                    result = self.progression_model.generate_from_prompt(test_input)
                    if result:
                        chords = result[0].get('chords', [])
                        lengths.append(len(chords))
                    else:
                        lengths.append(0)
                else:
                    self.log_test_result("length_consistency", False, error="Model not available")
                    return
            except Exception as e:
                self.log_test_result("length_consistency", False, error=f"Length test failed: {e}")
                return
        
        # Check if all lengths are 4 (our expected default)
        if all(length == 4 for length in lengths):
            self.log_test_result("length_consistency", True, message=f"All progressions have consistent length: 4 chords")
        elif all(length == lengths[0] for length in lengths):
            self.log_test_result("length_consistency", False, error=f"Consistent but unexpected length: {lengths[0]} chords")
        else:
            self.log_test_result("length_consistency", False, error=f"Inconsistent lengths: {lengths}")
    
    def check_server_availability(self) -> bool:
        """Check if the server is available for testing"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_interpolation_edge_cases(self):
        """Test interpolation engine edge cases that commonly fail."""
        try:
            from emotion_interpolation_engine import EmotionInterpolationEngine
            engine = EmotionInterpolationEngine()
            
            # Test 1: Empty emotion weights (CRITICAL BUG FOUND)
            try:
                result = engine.create_emotion_state({})
                return False  # This should fail gracefully, not crash
            except ValueError as e:
                if "max() iterable argument is empty" in str(e):
                    return False  # This is the bug we found
                return True  # Other errors might be acceptable
            
        except Exception as e:
            raise e
    
    def test_interpolation_negative_weights(self):
        """Test interpolation with negative emotion weights."""
        try:
            from emotion_interpolation_engine import EmotionInterpolationEngine
            engine = EmotionInterpolationEngine()
            
            # Test negative weights
            result = engine.create_emotion_state({'Joy': -0.5, 'Sadness': 1.5})
            return result is not None
            
        except Exception as e:
            raise e
    
    def test_interpolation_none_values(self):
        """Test interpolation with None values."""
        try:
            from emotion_interpolation_engine import EmotionInterpolationEngine
            engine = EmotionInterpolationEngine()
            
            # Test None interpolation
            try:
                result = engine.interpolate_emotions(None, None, 0.5)
                return False  # Should handle gracefully
            except (TypeError, AttributeError):
                return True  # Expected to fail, but gracefully
            
        except Exception as e:
            raise e
    
    def test_wolfram_validator_import(self):
        """Test Wolfram validator import issues."""
        try:
            from wolfram_validator import WolframTheoryValidator
            return True
        except ImportError as e:
            return False  # This is a known issue
    
    def test_chord_progression_empty_input(self):
        """Test chord progression generation with empty input."""
        try:
            from chord_progression_model import ChordProgressionModel
            model = ChordProgressionModel()
            result = model.generate_from_prompt('', 'Pop')
            return len(result) > 0 and 'chords' in result[0]
        except Exception as e:
            raise e
    
    def test_chord_progression_unicode(self):
        """Test unicode and special character handling."""
        try:
            from chord_progression_model import ChordProgressionModel
            model = ChordProgressionModel()
            
            # Test unicode
            result1 = model.generate_from_prompt('Je suis heureux 🎵', 'Jazz')
            
            # Test special characters
            result2 = model.generate_from_prompt('I feel !@#$%^&*()', 'Rock')
            
            return (len(result1) > 0 and len(result2) > 0)
        except Exception as e:
            raise e
    
    def test_individual_chord_edge_cases(self):
        """Test individual chord model edge cases."""
        try:
            from individual_chord_model import IndividualChordModel
            model = IndividualChordModel()
            
            # Test various edge cases
            result1 = model.generate_chord_from_prompt('')
            result2 = model.generate_chord_from_prompt('test', mode_preference='InvalidMode')
            result3 = model.generate_chord_from_prompt('test', style_preference='InvalidStyle')
            
            return all(len(r) > 0 for r in [result1, result2, result3])
        except Exception as e:
            raise e
    
    def test_neural_analyzer_edge_cases(self):
        """Test neural analyzer with edge case inputs."""
        try:
            from neural_progression_analyzer import ContextualProgressionIntegrator
            analyzer = ContextualProgressionIntegrator()
            
            # Test various edge cases
            result1 = analyzer.analyze_progression_context([])
            result2 = analyzer.analyze_progression_context(['I'])
            result3 = analyzer.analyze_progression_context(['XYZ', 'ABC'])
            
            return all(r is not None for r in [result1, result2, result3])
        except Exception as e:
            raise e
    
    def test_extreme_parameter_values(self):
        """Test extreme parameter values."""
        try:
            from chord_progression_model import ChordProgressionModel
            model = ChordProgressionModel()
            
            # Test extreme values
            result1 = model.generate_from_prompt('happy', 'Pop', num_progressions=0)
            result2 = model.generate_from_prompt('happy', 'Pop', num_progressions=-5)
            result3 = model.generate_from_prompt('happy', 'Pop', num_progressions=1000)
            
            return (len(result1) == 0 and len(result2) >= 0 and len(result3) <= 100)
        except Exception as e:
            raise e
    
    def test_all_22_emotions(self):
        """Test all 22 emotions systematically."""
        try:
            from chord_progression_model import ChordProgressionModel
            model = ChordProgressionModel()
            
            emotions = ['Joy', 'Sadness', 'Fear', 'Anger', 'Disgust', 'Surprise', 
                       'Trust', 'Anticipation', 'Shame', 'Love', 'Envy', 'Aesthetic Awe',
                       'Malice', 'Arousal', 'Guilt', 'Reverence', 'Wonder', 'Dissociation', 
                       'Empowerment', 'Belonging', 'Ideology', 'Gratitude']
            
            failed_emotions = []
            for emotion in emotions:
                try:
                    result = model.generate_from_prompt(f'I feel {emotion.lower()}', 'Pop')
                    if len(result) == 0 or 'chords' not in result[0]:
                        failed_emotions.append(emotion)
                except Exception as e:
                    failed_emotions.append(f"{emotion} ({str(e)})")
            
            return len(failed_emotions) == 0
        except Exception as e:
            raise e
    
    def test_performance_edge_cases(self):
        """Test performance under stress conditions."""
        try:
            from chord_progression_model import ChordProgressionModel
            model = ChordProgressionModel()
            
            # Rapid-fire generation test
            start_time = time.time()
            for i in range(50):
                model.generate_from_prompt(f'Test {i}', 'Pop')
            rapid_time = time.time() - start_time
            
            # Large batch test
            start_time = time.time()
            large_batch = model.generate_from_prompt('happy', 'Pop', num_progressions=100)
            batch_time = time.time() - start_time
            
            # Performance should be reasonable
            return rapid_time < 10.0 and batch_time < 30.0 and len(large_batch) <= 100
        except Exception as e:
            raise e
    
    def run_comprehensive_tests(self):
        """Run all edge case tests."""
        print("🧪 COMPREHENSIVE EDGE CASE DETECTION")
        print("=" * 50)
        
        # Critical interpolation tests
        self.run_test("Interpolation Empty Weights Bug", self.test_interpolation_edge_cases)
        self.run_test("Interpolation Negative Weights", self.test_interpolation_negative_weights)
        self.run_test("Interpolation None Values", self.test_interpolation_none_values)
        
        # Import and dependency tests
        self.run_test("Wolfram Validator Import", self.test_wolfram_validator_import)
        
        # Core functionality tests
        self.run_test("Chord Progression Empty Input", self.test_chord_progression_empty_input)
        self.run_test("Unicode and Special Characters", self.test_chord_progression_unicode)
        self.run_test("Individual Chord Edge Cases", self.test_individual_chord_edge_cases)
        self.run_test("Neural Analyzer Edge Cases", self.test_neural_analyzer_edge_cases)
        self.run_test("Extreme Parameter Values", self.test_extreme_parameter_values)
        self.run_test("All 22 Emotions Support", self.test_all_22_emotions)
        self.run_test("Performance Edge Cases", self.test_performance_edge_cases)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive edge case report."""
        print("\n" + "=" * 60)
        print("📊 EDGE CASE DETECTION RESULTS")
        print("=" * 60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ✅")
        print(f"Failed: {self.results['failed']} ❌")
        print(f"Success Rate: {(self.results['passed']/self.results['total_tests']*100):.1f}%")
        
        if self.results['critical_errors']:
            print(f"\n🚨 CRITICAL ERRORS ({len(self.results['critical_errors'])}):")
            for i, error in enumerate(self.results['critical_errors'], 1):
                print(f"{i}. {error['test']}: {error['error']}")
        
        if self.results['warnings']:
            print(f"\n⚠️ WARNINGS ({len(self.results['warnings'])}):")
            for i, warning in enumerate(self.results['warnings'], 1):
                print(f"{i}. {warning['test']}: {warning['error']}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'edge_case_detection_report_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📁 Detailed report saved to: {filename}")
        
        return self.results

def main():
    """Run comprehensive edge case detection"""
    detector = EdgeCaseDetector()
    report = detector.run_comprehensive_tests()
    
    # Save report to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"edge_case_detection_report_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Full report saved to: {filename}")
    
    return report

if __name__ == "__main__":
    main() 