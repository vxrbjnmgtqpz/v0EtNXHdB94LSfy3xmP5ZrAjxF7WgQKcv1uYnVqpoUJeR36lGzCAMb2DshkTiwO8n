#!/usr/bin/env python3
"""
EDGE CASE ERROR HANDLING AUDIT AND TESTING
This script tests for comprehensive edge case handling and error catching
throughout the VirtualAssistance system.
"""

import os
import sys
import json
import traceback
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

class EdgeCaseErrorAudit:
    """Comprehensive edge case and error handling audit"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'critical_errors': [],
            'edge_cases_found': [],
            'error_handling_gaps': [],
            'recommendations': []
        }
        
        # Import models safely
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
            
    def run_edge_case_audit(self):
        """Run comprehensive edge case audit"""
        print("🔍 Starting Edge Case Error Handling Audit...")
        print("=" * 70)
        
        # 1. Test invalid inputs
        self._test_invalid_inputs()
        
        # 2. Test boundary conditions
        self._test_boundary_conditions()
        
        # 3. Test missing data handling
        self._test_missing_data_handling()
        
        # 4. Test malformed inputs
        self._test_malformed_inputs()
        
        # 5. Test resource exhaustion
        self._test_resource_exhaustion()
        
        # 6. Test transcendence integration edge cases
        self._test_transcendence_edge_cases()
        
        # 7. Generate final report
        self._generate_edge_case_report()
        
    def _run_test(self, test_name: str, test_func):
        """Run a test with error handling"""
        self.results['tests_run'] += 1
        try:
            print(f"\n🔍 Testing: {test_name}")
            result = test_func()
            if result:
                self.results['tests_passed'] += 1
                print(f"   ✅ PASSED: {test_name}")
            else:
                self.results['tests_failed'] += 1
                print(f"   ❌ FAILED: {test_name}")
                
        except Exception as e:
            self.results['tests_failed'] += 1
            error_detail = {
                'test': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.results['critical_errors'].append(error_detail)
            print(f"   💥 CRITICAL ERROR: {test_name} - {str(e)}")
            
    def _test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        print("\n1. 🚫 Testing Invalid Inputs...")
        
        if self.progression_model:
            # Test empty prompt
            self._run_test("Empty prompt handling", 
                          lambda: self._test_empty_prompt())
            
            # Test extremely long prompt
            self._run_test("Extremely long prompt handling",
                          lambda: self._test_long_prompt())
            
            # Test special characters
            self._run_test("Special character handling",
                          lambda: self._test_special_characters())
            
            # Test non-ASCII characters
            self._run_test("Non-ASCII character handling",
                          lambda: self._test_non_ascii())
            
    def _test_empty_prompt(self):
        """Test empty prompt handling"""
        try:
            result = self.progression_model.generate_from_prompt("", "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            if "empty" in str(e).lower() or "invalid" in str(e).lower():
                return True  # Good error handling
            else:
                raise  # Unexpected error
                
    def _test_long_prompt(self):
        """Test extremely long prompt handling"""
        long_prompt = "I want something very " * 1000 + "happy"
        try:
            result = self.progression_model.generate_from_prompt(long_prompt, "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            if "too long" in str(e).lower() or "length" in str(e).lower():
                return True  # Good error handling
            else:
                # Check if it handled gracefully
                return len(str(e)) < 500  # Reasonable error message
                
    def _test_special_characters(self):
        """Test special character handling"""
        special_prompt = "I want something @#$%^&*()_+-=[]{}|;':\",./<>?"
        try:
            result = self.progression_model.generate_from_prompt(special_prompt, "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "character" in str(e).lower() or "invalid" in str(e).lower()
            
    def _test_non_ascii(self):
        """Test non-ASCII character handling"""
        non_ascii_prompt = "I want something très émotionnel 音楽 🎵"
        try:
            result = self.progression_model.generate_from_prompt(non_ascii_prompt, "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "encoding" in str(e).lower() or "unicode" in str(e).lower()
            
    def _test_boundary_conditions(self):
        """Test boundary conditions"""
        print("\n2. 🎯 Testing Boundary Conditions...")
        
        if self.progression_model:
            # Test zero progressions requested
            self._run_test("Zero progressions requested",
                          lambda: self._test_zero_progressions())
            
            # Test negative progressions
            self._run_test("Negative progressions requested",
                          lambda: self._test_negative_progressions())
            
            # Test extremely large number of progressions
            self._run_test("Extremely large progression count",
                          lambda: self._test_large_progression_count())
            
    def _test_zero_progressions(self):
        """Test requesting zero progressions"""
        try:
            result = self.progression_model.generate_from_prompt("happy", "Pop", 0)
            return isinstance(result, list) and len(result) == 0
        except Exception as e:
            return "zero" in str(e).lower() or "invalid" in str(e).lower()
            
    def _test_negative_progressions(self):
        """Test requesting negative progressions"""
        try:
            result = self.progression_model.generate_from_prompt("happy", "Pop", -1)
            return isinstance(result, list) and len(result) >= 0
        except Exception as e:
            return "negative" in str(e).lower() or "invalid" in str(e).lower()
            
    def _test_large_progression_count(self):
        """Test requesting large number of progressions"""
        try:
            result = self.progression_model.generate_from_prompt("happy", "Pop", 1000)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "too many" in str(e).lower() or "limit" in str(e).lower()
            
    def _test_missing_data_handling(self):
        """Test missing data handling"""
        print("\n3. 📊 Testing Missing Data Handling...")
        
        if self.progression_model:
            # Test unknown emotion
            self._run_test("Unknown emotion handling",
                          lambda: self._test_unknown_emotion())
            
            # Test unknown genre
            self._run_test("Unknown genre handling",
                          lambda: self._test_unknown_genre())
            
            # Test None values
            self._run_test("None value handling",
                          lambda: self._test_none_values())
            
    def _test_unknown_emotion(self):
        """Test unknown emotion handling"""
        try:
            result = self.progression_model.generate_from_prompt("I want something xyz123unknown", "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "unknown" in str(e).lower() or "not found" in str(e).lower()
            
    def _test_unknown_genre(self):
        """Test unknown genre handling"""
        try:
            result = self.progression_model.generate_from_prompt("happy", "UnknownGenre123", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "genre" in str(e).lower() or "unknown" in str(e).lower()
            
    def _test_none_values(self):
        """Test None value handling"""
        try:
            result = self.progression_model.generate_from_prompt(None, "Pop", 1)
            return False  # Should not succeed
        except Exception as e:
            return "none" in str(e).lower() or "null" in str(e).lower() or "invalid" in str(e).lower()
            
    def _test_malformed_inputs(self):
        """Test malformed inputs"""
        print("\n4. 🔧 Testing Malformed Inputs...")
        
        if self.progression_model:
            # Test wrong data types
            self._run_test("Wrong data type for prompt",
                          lambda: self._test_wrong_data_types())
            
            # Test malformed JSON-like strings
            self._run_test("Malformed JSON-like strings",
                          lambda: self._test_malformed_json())
            
    def _test_wrong_data_types(self):
        """Test wrong data types"""
        try:
            result = self.progression_model.generate_from_prompt(123, "Pop", 1)
            return False  # Should not succeed
        except Exception as e:
            return "type" in str(e).lower() or "string" in str(e).lower()
            
    def _test_malformed_json(self):
        """Test malformed JSON-like strings"""
        malformed = '{"emotion": "happy", "genre": "Pop",}'
        try:
            result = self.progression_model.generate_from_prompt(malformed, "Pop", 1)
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            return "json" in str(e).lower() or "format" in str(e).lower()
            
    def _test_resource_exhaustion(self):
        """Test resource exhaustion scenarios"""
        print("\n5. 💻 Testing Resource Exhaustion...")
        
        if self.progression_model:
            # Test rapid successive calls
            self._run_test("Rapid successive calls",
                          lambda: self._test_rapid_calls())
            
    def _test_rapid_calls(self):
        """Test rapid successive calls"""
        try:
            results = []
            for i in range(10):
                result = self.progression_model.generate_from_prompt("happy", "Pop", 1)
                results.append(result)
            return len(results) == 10 and all(isinstance(r, list) for r in results)
        except Exception as e:
            return "rate" in str(e).lower() or "limit" in str(e).lower()
            
    def _test_transcendence_edge_cases(self):
        """Test Transcendence-specific edge cases"""
        print("\n6. 🌀 Testing Transcendence Edge Cases...")
        
        if self.progression_model:
            # Test Transcendence emotion recognition
            self._run_test("Transcendence emotion recognition",
                          lambda: self._test_transcendence_recognition())
            
            # Test sub-emotions
            self._run_test("Transcendence sub-emotions",
                          lambda: self._test_transcendence_sub_emotions())
            
            # Test transcendence keywords
            self._run_test("Transcendence keywords",
                          lambda: self._test_transcendence_keywords())
            
    def _test_transcendence_recognition(self):
        """Test Transcendence emotion recognition"""
        try:
            result = self.progression_model.generate_from_prompt("I want something transcendent", "Ambient", 1)
            if isinstance(result, list) and len(result) > 0:
                # Check if result contains transcendence-related content
                result_str = str(result).lower()
                return "transcend" in result_str or "mystical" in result_str
            return False
        except Exception as e:
            return False
            
    def _test_transcendence_sub_emotions(self):
        """Test Transcendence sub-emotions"""
        sub_emotions = ["lucid wonder", "ego death", "dreamflight", "sacred dissonance"]
        try:
            for sub_emotion in sub_emotions:
                result = self.progression_model.generate_from_prompt(f"I want something {sub_emotion}", "Ambient", 1)
                if not (isinstance(result, list) and len(result) > 0):
                    return False
            return True
        except Exception as e:
            return False
            
    def _test_transcendence_keywords(self):
        """Test various Transcendence keywords"""
        keywords = ["mystical", "ethereal", "cosmic", "divine", "enlightened", "dreamlike", "lucid"]
        try:
            for keyword in keywords:
                result = self.progression_model.generate_from_prompt(f"I want something {keyword}", "Ambient", 1)
                if not (isinstance(result, list) and len(result) > 0):
                    return False
            return True
        except Exception as e:
            return False
            
    def _generate_edge_case_report(self):
        """Generate comprehensive edge case report"""
        print("\n" + "=" * 70)
        print("📊 EDGE CASE AUDIT SUMMARY")
        print("=" * 70)
        
        total_tests = self.results['tests_run']
        passed_tests = self.results['tests_passed']
        failed_tests = self.results['tests_failed']
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {failed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Critical Errors: {len(self.results['critical_errors'])}")
        
        # Determine system health
        if pass_rate >= 90:
            health = "🟢 EXCELLENT"
        elif pass_rate >= 75:
            health = "🟡 GOOD"
        elif pass_rate >= 50:
            health = "🟠 NEEDS IMPROVEMENT"
        else:
            health = "🔴 CRITICAL"
            
        print(f"System Health: {health}")
        
        # Display critical errors
        if self.results['critical_errors']:
            print("\n🚨 CRITICAL ERRORS:")
            for error in self.results['critical_errors']:
                print(f"   • {error['test']}: {error['error']}")
                
        # Generate recommendations
        if pass_rate < 80:
            print("\n💡 RECOMMENDATIONS:")
            print("   • Implement more robust input validation")
            print("   • Add comprehensive error handling for edge cases")
            print("   • Improve boundary condition handling")
            print("   • Add rate limiting for resource protection")
            
        # Save detailed report
        report_file = f"edge_case_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    auditor = EdgeCaseErrorAudit()
    auditor.run_edge_case_audit()
