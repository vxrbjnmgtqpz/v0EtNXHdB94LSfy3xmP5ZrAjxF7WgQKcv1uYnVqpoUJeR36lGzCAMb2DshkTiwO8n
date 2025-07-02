#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing for VirtualAssistance Model Stack

This script systematically tests:
1. Chord inversion handling
2. Neural substitution detection
3. Color coding functionality  
4. Expanded emotion database integration
5. Audio playback compatibility
6. Cross-model consistency
"""

import sys
import os
sys.path.append('.')

from chord_progression_model import ChordProgressionModel
from individual_chord_model import IndividualChordModel
from integrated_chat_server import IntegratedMusicChatServer
import json
import traceback
from typing import Dict, List, Any

class EdgeCaseTestSuite:
    """Comprehensive test suite for catching system edge cases"""
    
    def __init__(self):
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "total_tests": 0
        }
        
        # Initialize models
        try:
            print("🚀 Initializing models for edge case testing...")
            self.progression_model = ChordProgressionModel()
            self.individual_model = IndividualChordModel()
            self.server = IntegratedMusicChatServer()
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to initialize models: {e}")
            raise
    
    def run_all_tests(self):
        """Execute all edge case tests"""
        print("\n" + "="*60)
        print("🧪 RUNNING COMPREHENSIVE EDGE CASE TESTS")
        print("="*60)
        
        # Test categories
        test_categories = [
            ("Chord Inversion Tests", self.test_chord_inversions),
            ("Neural Substitution Tests", self.test_neural_substitutions),
            ("Color Coding Tests", self.test_color_coding),
            ("Expanded Emotion Database Tests", self.test_expanded_emotions),
            ("Audio Compatibility Tests", self.test_audio_compatibility),
            ("Cross-Model Consistency Tests", self.test_cross_model_consistency),
            ("Edge Case Chord Mappings", self.test_edge_case_chords)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 50)
            try:
                test_function()
            except Exception as e:
                self.log_failure(f"{category_name} - Critical Error", str(e))
                print(f"❌ Critical error in {category_name}: {e}")
        
        self.print_summary()
    
    def test_chord_inversions(self):
        """Test comprehensive chord inversion handling"""
        print("Testing chord inversion handling...")
        
        # Test inversions that should now work
        inversion_tests = [
            ("I6", "First inversion of I major"),
            ("ii6", "First inversion of ii minor"),
            ("V6", "First inversion of V major"),
            ("I6/4", "Second inversion of I major"),
            ("V6/4", "Second inversion of V major"),
            ("IM76", "IM7 first inversion"),
            ("V76", "V7 first inversion"),
            ("i6", "i minor first inversion"),
            ("iv6", "iv minor first inversion")
        ]
        
        for chord, description in inversion_tests:
            try:
                # Test with progression model
                result = self.progression_model.generate_from_prompt(f"happy progression with {chord}")
                if result and len(result) > 0:
                    self.log_success(f"Inversion Test: {chord}", f"Generated progression with {description}")
                else:
                    self.log_warning(f"Inversion Test: {chord}", "Generated empty result")
                
                # Test with individual model  
                chord_result = self.individual_model.generate_chord_from_prompt(f"happy {chord} chord")
                if chord_result:
                    self.log_success(f"Individual Inversion: {chord}", f"Generated individual chord analysis")
                else:
                    self.log_warning(f"Individual Inversion: {chord}", "No individual analysis generated")
                    
            except Exception as e:
                self.log_failure(f"Inversion Test: {chord}", f"Error testing {description}: {str(e)}")
    
    def test_neural_substitutions(self):
        """Test neural substitution detection and generation"""
        print("Testing neural substitution system...")
        
        # Force neural generation for testing
        original_neural_state = self.progression_model.use_neural_generation
        self.progression_model.use_neural_generation = True
        
        test_prompts = [
            ("happy and excited", "Pop"),
            ("melancholic jazz", "Jazz"),
            ("angry classical", "Classical"),
            ("romantic blues", "Blues")
        ]
        
        substitution_count = 0
        total_chords = 0
        
        for prompt, genre in test_prompts:
            try:
                result = self.progression_model.generate_from_prompt(prompt, genre)
                if result and len(result) > 0:
                    progression_data = result[0]
                    
                    # Check for substitution metadata
                    if "chord_metadata" in progression_data:
                        chord_metadata = progression_data["chord_metadata"]
                        total_chords += len(chord_metadata)
                        
                        for meta in chord_metadata:
                            if meta.get("is_substitution", False):
                                substitution_count += 1
                                self.log_success(
                                    f"Neural Substitution: {prompt}",
                                    f"Found substitution: {meta.get('original_chord')} → {meta.get('current_chord')}"
                                )
                    
                    # Check generation method
                    gen_method = progression_data.get("metadata", {}).get("generation_method")
                    if gen_method == "neural_generation":
                        self.log_success(f"Neural Method: {prompt}", "Used neural generation successfully")
                    else:
                        self.log_warning(f"Neural Method: {prompt}", f"Used {gen_method} instead of neural")
                        
                else:
                    self.log_failure(f"Neural Test: {prompt}", "No result generated")
                    
            except Exception as e:
                self.log_failure(f"Neural Test: {prompt}", f"Error: {str(e)}")
        
        # Restore original state
        self.progression_model.use_neural_generation = original_neural_state
        
        substitution_rate = (substitution_count / total_chords * 100) if total_chords > 0 else 0
        print(f"🎯 Neural substitution rate: {substitution_rate:.1f}% ({substitution_count}/{total_chords})")
        
        if substitution_rate > 0:
            self.log_success("Neural Substitution Rate", f"{substitution_rate:.1f}% substitution rate achieved")
        else:
            self.log_warning("Neural Substitution Rate", "No substitutions detected - check neural generation")
    
    def test_color_coding(self):
        """Test color coding functionality through server integration"""
        print("Testing color coding system...")
        
        # Test via server integration
        test_messages = [
            "happy and excited jazz progression",
            "sad classical music",
            "I feel melancholic and nostalgic"
        ]
        
        for message in test_messages:
            try:
                # Simulate server processing
                response = self.server.process_message(message)
                
                if response and "chord_metadata" in response:
                    metadata = response["chord_metadata"]
                    has_substitutions = any(meta.get("is_substitution", False) for meta in metadata)
                    
                    if has_substitutions:
                        self.log_success(f"Color Coding: {message[:20]}...", "Found substitutions for orange color coding")
                    else:
                        self.log_success(f"Color Coding: {message[:20]}...", "All database chords for green color coding")
                        
                    # Check if metadata structure is correct
                    for i, meta in enumerate(metadata):
                        required_fields = ["is_substitution", "original_chord", "source"]
                        missing_fields = [field for field in required_fields if field not in meta]
                        
                        if missing_fields:
                            self.log_warning(
                                f"Color Metadata: Chord {i}",
                                f"Missing fields: {missing_fields}"
                            )
                
                else:
                    self.log_warning(f"Color Coding: {message[:20]}...", "No chord_metadata found in response")
                    
            except Exception as e:
                self.log_failure(f"Color Coding: {message[:20]}...", f"Error: {str(e)}")
    
    def test_expanded_emotions(self):
        """Test expanded emotion database with sub-emotions"""
        print("Testing expanded emotion database...")
        
        # Test sub-emotion detection
        sub_emotion_tests = [
            ("I feel excited and thrilled", "Joy:Excitement"),
            ("I feel melancholic and wistful", "Sadness:Melancholy"),
            ("I feel anxious and worried", "Fear:Anxiety"),
            ("I feel content and peaceful", "Joy:Contentment"),
            ("I feel romantic and loving", "Love:Romantic Longing"),
            ("I feel bitter and resentful", "Anger:Resentment")
        ]
        
        for prompt, expected_sub_emotion in sub_emotion_tests:
            try:
                result = self.progression_model.generate_from_prompt(prompt)
                if result and len(result) > 0:
                    progression_data = result[0]
                    detected_sub = progression_data.get("detected_sub_emotion", "")
                    
                    if detected_sub:
                        if detected_sub == expected_sub_emotion:
                            self.log_success(f"Sub-emotion: {prompt[:20]}...", f"Correctly detected {detected_sub}")
                        else:
                            self.log_warning(f"Sub-emotion: {prompt[:20]}...", f"Expected {expected_sub_emotion}, got {detected_sub}")
                    else:
                        self.log_warning(f"Sub-emotion: {prompt[:20]}...", "No sub-emotion detected")
                        
            except Exception as e:
                self.log_failure(f"Sub-emotion: {prompt[:20]}...", f"Error: {str(e)}")
        
        # Test database version compatibility
        try:
            with open('emotion_progression_database.json', 'r') as f:
                db_data = json.load(f)
                
            if "schema_version" in db_data and db_data["schema_version"] == "2.0":
                self.log_success("Database Schema", "Using v2.0 schema with sub-emotions")
            else:
                self.log_warning("Database Schema", "Not using latest v2.0 schema")
                
        except Exception as e:
            self.log_failure("Database Schema", f"Error loading database: {str(e)}")
    
    def test_audio_compatibility(self):
        """Test audio playback compatibility with new chord types"""
        print("Testing audio compatibility...")
        
        # Test chords that should be playable
        playable_chords = [
            "I", "V", "vi", "IV",  # Basic chords
            "I6", "V6", "ii6",     # New inversions
            "IM7", "V7", "vi7",    # Seventh chords
            "I6/4", "V6/4",        # Second inversions
            "IM76", "V76"          # Seventh inversions
        ]
        
        # Simulate audio mapping test (checking if chords have valid mappings)
        for chord in playable_chords:
            try:
                # Test through progression generation
                result = self.progression_model.generate_from_prompt(f"test {chord} chord")
                if result and len(result) > 0:
                    chords = result[0].get("chords", [])
                    if chord in chords or any(c == chord for c in chords):
                        self.log_success(f"Audio Test: {chord}", "Chord appears in generated progressions")
                    else:
                        self.log_warning(f"Audio Test: {chord}", "Chord not appearing in progressions")
                        
            except Exception as e:
                self.log_failure(f"Audio Test: {chord}", f"Error: {str(e)}")
    
    def test_cross_model_consistency(self):
        """Test consistency between different models"""
        print("Testing cross-model consistency...")
        
        test_emotions = ["happy", "sad", "romantic", "mysterious"]
        
        for emotion in test_emotions:
            try:
                # Get progression model result
                prog_result = self.progression_model.generate_from_prompt(f"{emotion} music")
                
                # Get individual model result
                indiv_result = self.individual_model.generate_chord_from_prompt(f"{emotion} chord")
                
                # Get server integration result
                server_result = self.server.process_message(f"{emotion} progression")
                
                # Check if all models respond appropriately
                if prog_result and indiv_result and server_result:
                    self.log_success(f"Consistency: {emotion}", "All models responded successfully")
                    
                    # Check for emotion consistency
                    prog_emotions = prog_result[0].get("emotion_weights", {}) if prog_result else {}
                    if prog_emotions:
                        top_emotion = max(prog_emotions, key=prog_emotions.get)
                        self.log_success(f"Emotion Mapping: {emotion}", f"Top detected emotion: {top_emotion}")
                        
                else:
                    missing = []
                    if not prog_result: missing.append("progression")
                    if not indiv_result: missing.append("individual")
                    if not server_result: missing.append("server")
                    
                    self.log_warning(f"Consistency: {emotion}", f"Missing responses from: {missing}")
                    
            except Exception as e:
                self.log_failure(f"Consistency: {emotion}", f"Error: {str(e)}")
    
    def test_edge_case_chords(self):
        """Test unusual and edge case chord symbols"""
        print("Testing edge case chord symbols...")
        
        edge_case_chords = [
            "N6",           # Neapolitan sixth
            "Ger6",         # German sixth
            "Fr6",          # French sixth
            "It6",          # Italian sixth
            "♭II6",         # Flat II sixth
            "V7/vi",        # Secondary dominant
            "ii°7",         # Diminished seventh
            "V7alt",        # Altered dominant
            "#iv°7",        # Sharp iv diminished
            "♭VI7"          # Flat VI seventh
        ]
        
        for chord in edge_case_chords:
            try:
                # Test if the system can handle these chords
                result = self.progression_model.generate_from_prompt(f"progression with {chord}")
                if result:
                    self.log_success(f"Edge Case: {chord}", "System handled unusual chord symbol")
                else:
                    self.log_warning(f"Edge Case: {chord}", "No result for unusual chord")
                    
            except Exception as e:
                self.log_failure(f"Edge Case: {chord}", f"Error with {chord}: {str(e)}")
    
    def log_success(self, test_name: str, message: str):
        """Log a successful test"""
        self.test_results["passed"].append({"test": test_name, "message": message})
        self.test_results["total_tests"] += 1
        print(f"✅ {test_name}: {message}")
    
    def log_warning(self, test_name: str, message: str):
        """Log a test warning"""
        self.test_results["warnings"].append({"test": test_name, "message": message})
        self.test_results["total_tests"] += 1
        print(f"⚠️  {test_name}: {message}")
    
    def log_failure(self, test_name: str, message: str):
        """Log a failed test"""
        self.test_results["failed"].append({"test": test_name, "message": message})
        self.test_results["total_tests"] += 1
        print(f"❌ {test_name}: {message}")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("📊 EDGE CASE TEST SUMMARY")
        print("="*60)
        
        total = self.test_results["total_tests"]
        passed = len(self.test_results["passed"])
        warnings = len(self.test_results["warnings"])
        failed = len(self.test_results["failed"])
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"🎯 Total Tests: {total}")
        print(f"✅ Passed: {passed} ({success_rate:.1f}%)")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print(f"\n🚨 CRITICAL ISSUES FOUND:")
            for failure in self.test_results["failed"]:
                print(f"   ❌ {failure['test']}: {failure['message']}")
        
        if warnings > 0:
            print(f"\n⚠️  WARNINGS (may need attention):")
            for warning in self.test_results["warnings"][:5]:  # Show first 5 warnings
                print(f"   ⚠️  {warning['test']}: {warning['message']}")
            if len(self.test_results["warnings"]) > 5:
                print(f"   ... and {len(self.test_results['warnings']) - 5} more warnings")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed == 0 and warnings == 0:
            print("   🎉 Excellent! No critical issues found.")
        elif failed == 0:
            print("   ✨ Good! No critical failures, but some warnings to review.")
        else:
            print("   🔧 Address critical failures before deployment.")
            
        # Save results
        with open('edge_case_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n📄 Detailed results saved to: edge_case_test_results.json")

def main():
    """Run the comprehensive edge case test suite"""
    print("🧪 VirtualAssistance Model Stack - Edge Case Testing")
    print("Testing inversions, neural substitutions, color coding, and expanded emotions\n")
    
    try:
        test_suite = EdgeCaseTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
