#!/usr/bin/env python3
"""
Comprehensive Test Suite for Consonant/Dissonant Interpolation Engine Integration
Phase 2: Interpolation Engine Enhancement

This test validates:
1. Enhanced EmotionState creation with CD values
2. Tension curve generation for all curve types
3. Enhanced interpolation with CD support
4. Complete progression generation with CD trajectories
5. Tension curve analysis functionality

Version: 1.0
Created: January 2, 2025
"""

import sys
import os
import json
import traceback
from typing import Dict, List, Optional

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emotion_interpolation_engine import (
    EmotionInterpolationEngine, 
    EmotionState, 
    InterpolationMethod,
    TensionCurveType,
    InterpolatedProgression
)

class InterpolationCDTester:
    """Comprehensive tester for CD-enhanced interpolation engine"""
    
    def __init__(self):
        self.engine = EmotionInterpolationEngine()
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details
        }
        self.test_results.append(result)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if details:
            print(f"   📋 {details}")
        print()
    
    def test_emotion_state_cd_creation(self):
        """Test enhanced EmotionState creation with CD values"""
        print("🧪 Testing Enhanced EmotionState Creation with CD Values")
        
        try:
            # Test 1: Consonant emotion state
            consonant_emotions = {"Joy": 0.7, "Gratitude": 0.3}
            consonant_state = self.engine.create_emotion_state(
                consonant_emotions, 
                style_context="Classical"
            )
            
            consonant_cd = consonant_state.consonant_dissonant_value
            is_consonant = consonant_cd is not None and consonant_cd < 0.4
            
            self.log_test(
                "Consonant EmotionState Creation",
                is_consonant,
                f"CD value: {consonant_cd:.3f}, Trajectory: {consonant_state.consonant_dissonant_trajectory}"
            )
            
            # Test 2: Dissonant emotion state
            dissonant_emotions = {"Malice": 0.6, "Anger": 0.4}
            dissonant_state = self.engine.create_emotion_state(
                dissonant_emotions,
                style_context="Classical"
            )
            
            dissonant_cd = dissonant_state.consonant_dissonant_value
            is_dissonant = dissonant_cd is not None and dissonant_cd > 0.6
            
            self.log_test(
                "Dissonant EmotionState Creation",
                is_dissonant,
                f"CD value: {dissonant_cd:.3f}, Trajectory: {dissonant_state.consonant_dissonant_trajectory}"
            )
            
            # Test 3: CD trajectory detection
            mixed_emotions = {"Joy": 0.4, "Malice": 0.6}
            mixed_state = self.engine.create_emotion_state(mixed_emotions)
            
            has_trajectory = mixed_state.consonant_dissonant_trajectory is not None
            
            self.log_test(
                "CD Trajectory Detection",
                has_trajectory,
                f"Trajectory: {mixed_state.consonant_dissonant_trajectory}"
            )
            
        except Exception as e:
            self.log_test("EmotionState CD Creation", False, f"Error: {str(e)}")
    
    def test_tension_curve_generation(self):
        """Test tension curve generation for all curve types"""
        print("🧪 Testing Tension Curve Generation")
        
        curve_types = [
            TensionCurveType.LINEAR,
            TensionCurveType.BUILD,
            TensionCurveType.RELEASE,
            TensionCurveType.PEAK,
            TensionCurveType.VALLEY,
            TensionCurveType.WAVE,
            TensionCurveType.ARCH,
            TensionCurveType.INVERTED_ARCH
        ]
        
        start_cd = 0.2
        end_cd = 0.8
        steps = 8
        
        for curve_type in curve_types:
            try:
                curve = self.engine.create_tension_curve(
                    start_cd, end_cd, steps, curve_type
                )
                
                # Validate curve properties
                correct_length = len(curve) == steps
                all_valid_range = all(0.0 <= val <= 1.0 for val in curve)
                
                # Different curve types have different behaviors
                if curve_type == TensionCurveType.WAVE:
                    # Wave curves oscillate around center - check that it oscillates properly
                    base_value = (start_cd + end_cd) / 2
                    has_oscillation = max(curve) > base_value and min(curve) < base_value
                    curve_valid = correct_length and all_valid_range and has_oscillation
                else:
                    # Other curves should start correctly
                    starts_correctly = abs(curve[0] - start_cd) < 0.1
                    
                    # Different end point behaviors
                    if curve_type in [TensionCurveType.BUILD, TensionCurveType.PEAK, TensionCurveType.ARCH]:
                        ends_correctly = abs(curve[-1] - end_cd) < 0.15  # Some curves may overshoot
                    else:
                        ends_correctly = abs(curve[-1] - end_cd) < 0.1
                    
                    curve_valid = correct_length and starts_correctly and ends_correctly and all_valid_range
                
                self.log_test(
                    f"Tension Curve: {curve_type.value}",
                    curve_valid,
                    f"Length: {len(curve)}, Start: {curve[0]:.3f}, End: {curve[-1]:.3f}, Range: [{min(curve):.3f}, {max(curve):.3f}]"
                )
                
            except Exception as e:
                self.log_test(f"Tension Curve: {curve_type.value}", False, f"Error: {str(e)}")
    
    def test_enhanced_interpolation(self):
        """Test enhanced interpolation with CD support"""
        print("🧪 Testing Enhanced Interpolation with CD Support")
        
        try:
            # Create test states
            consonant_emotions = {"Joy": 0.8, "Love": 0.2}
            dissonant_emotions = {"Anger": 0.6, "Malice": 0.4}
            
            start_state = self.engine.create_emotion_state(consonant_emotions)
            end_state = self.engine.create_emotion_state(dissonant_emotions)
            
            # Test 1: Standard interpolation
            mid_state = self.engine.interpolate_emotions(start_state, end_state, 0.5)
            
            has_cd_value = mid_state.consonant_dissonant_value is not None
            cd_in_range = start_state.consonant_dissonant_value < mid_state.consonant_dissonant_value < end_state.consonant_dissonant_value
            
            self.log_test(
                "Standard Interpolation with CD",
                has_cd_value and cd_in_range,
                f"Start CD: {start_state.consonant_dissonant_value:.3f}, Mid CD: {mid_state.consonant_dissonant_value:.3f}, End CD: {end_state.consonant_dissonant_value:.3f}"
            )
            
            # Test 2: Enhanced interpolation with tension curve
            enhanced_state = self.engine.interpolate_emotions_with_cd(
                start_state, end_state, 0.5, 
                InterpolationMethod.COSINE, 
                TensionCurveType.PEAK
            )
            
            has_enhanced_cd = enhanced_state.consonant_dissonant_value is not None
            has_trajectory = enhanced_state.consonant_dissonant_trajectory is not None
            
            self.log_test(
                "Enhanced Interpolation with Tension Curve",
                has_enhanced_cd and has_trajectory,
                f"Enhanced CD: {enhanced_state.consonant_dissonant_value:.3f}, Trajectory: {enhanced_state.consonant_dissonant_trajectory}"
            )
            
        except Exception as e:
            self.log_test("Enhanced Interpolation", False, f"Error: {str(e)}")
    
    def test_progression_generation_with_cd(self):
        """Test complete progression generation with CD trajectories"""
        print("🧪 Testing Progression Generation with CD Trajectories")
        
        try:
            # Test different tension curve types
            test_cases = [
                {
                    "name": "Linear CD Progression",
                    "start_emotion": {"Joy": 0.7, "Trust": 0.3},
                    "end_emotion": {"Sadness": 0.5, "Fear": 0.5},
                    "curve_type": TensionCurveType.LINEAR
                },
                {
                    "name": "Build Tension Progression",
                    "start_emotion": {"Love": 0.8, "Gratitude": 0.2},
                    "end_emotion": {"Anger": 0.6, "Malice": 0.4},
                    "curve_type": TensionCurveType.BUILD
                },
                {
                    "name": "Peak Tension Progression",
                    "start_emotion": {"Wonder": 0.6, "Aesthetic Awe": 0.4},
                    "end_emotion": {"Reverence": 0.7, "Gratitude": 0.3},
                    "curve_type": TensionCurveType.PEAK
                }
            ]
            
            for test_case in test_cases:
                try:
                    progression = self.engine.generate_interpolated_progression(
                        start_emotion=test_case["start_emotion"],
                        end_emotion=test_case["end_emotion"],
                        progression_length=8,
                        method=InterpolationMethod.COSINE,
                        tension_curve_type=test_case["curve_type"],
                        style_context="Classical"
                    )
                    
                    # Validate progression structure
                    has_chords = len(progression.chords) == 8
                    has_trajectory = len(progression.emotion_trajectory) == 8
                    has_cd_trajectory = len(progression.consonant_dissonant_trajectory) == 8
                    has_analysis = "curve_type" in progression.tension_curve_analysis
                    
                    progression_valid = has_chords and has_trajectory and has_cd_trajectory and has_analysis
                    
                    cd_start = progression.consonant_dissonant_trajectory[0]
                    cd_end = progression.consonant_dissonant_trajectory[-1]
                    cd_range = max(progression.consonant_dissonant_trajectory) - min(progression.consonant_dissonant_trajectory)
                    
                    self.log_test(
                        test_case["name"],
                        progression_valid,
                        f"Chords: {len(progression.chords)}, CD Range: {cd_range:.3f}, Start: {cd_start:.3f}, End: {cd_end:.3f}"
                    )
                    
                except Exception as e:
                    self.log_test(test_case["name"], False, f"Error: {str(e)}")
                    
        except Exception as e:
            self.log_test("Progression Generation with CD", False, f"Error: {str(e)}")
    
    def test_tension_curve_analysis(self):
        """Test tension curve analysis functionality"""
        print("🧪 Testing Tension Curve Analysis")
        
        try:
            # Create test CD trajectory
            cd_trajectory = [0.2, 0.4, 0.7, 0.9, 0.6, 0.3, 0.1, 0.2]
            
            # Test analysis
            analysis = self.engine._analyze_tension_curve(cd_trajectory, TensionCurveType.PEAK)
            
            # Validate analysis components
            has_basic_stats = all(key in analysis for key in [
                "start_tension", "end_tension", "max_tension", "min_tension", 
                "average_tension", "tension_range", "tension_direction"
            ])
            
            has_peaks_valleys = "peaks" in analysis and "valleys" in analysis
            has_stability = "tension_stability" in analysis
            has_character = "musical_character" in analysis
            has_resolution = "resolution_needed" in analysis
            
            analysis_complete = has_basic_stats and has_peaks_valleys and has_stability and has_character and has_resolution
            
            self.log_test(
                "Tension Curve Analysis",
                analysis_complete,
                f"Character: {analysis.get('musical_character', 'unknown')}, Peaks: {len(analysis.get('peaks', []))}, Valleys: {len(analysis.get('valleys', []))}, Stability: {analysis.get('tension_stability', 0):.3f}"
            )
            
            # Test empty trajectory handling
            empty_analysis = self.engine._analyze_tension_curve([])
            has_error_handling = "error" in empty_analysis
            
            self.log_test(
                "Empty Trajectory Handling",
                has_error_handling,
                f"Error message: {empty_analysis.get('error', 'none')}"
            )
            
        except Exception as e:
            self.log_test("Tension Curve Analysis", False, f"Error: {str(e)}")
    
    def test_integration_with_individual_chords(self):
        """Test integration with individual chord consonant/dissonant system"""
        print("🧪 Testing Integration with Individual Chord CD System")
        
        try:
            # Test that emotion-to-CD mapping is consistent
            consonant_emotions = {"Joy": 1.0}
            dissonant_emotions = {"Malice": 1.0}
            
            consonant_state = self.engine.create_emotion_state(consonant_emotions)
            dissonant_state = self.engine.create_emotion_state(dissonant_emotions)
            
            # Check CD values are in expected ranges
            consonant_cd = consonant_state.consonant_dissonant_value
            dissonant_cd = dissonant_state.consonant_dissonant_value
            
            consonant_correct = consonant_cd is not None and consonant_cd < 0.4
            dissonant_correct = dissonant_cd is not None and dissonant_cd > 0.7
            
            self.log_test(
                "Emotion-to-CD Mapping Consistency",
                consonant_correct and dissonant_correct,
                f"Joy CD: {consonant_cd:.3f} (should be <0.4), Malice CD: {dissonant_cd:.3f} (should be >0.7)"
            )
            
            # Test CD trajectory interpolation
            interpolated = self.engine.interpolate_emotions_with_cd(
                consonant_state, dissonant_state, 0.5, 
                tension_curve_type=TensionCurveType.LINEAR
            )
            
            interpolated_cd = interpolated.consonant_dissonant_value
            cd_in_middle = consonant_cd < interpolated_cd < dissonant_cd
            
            self.log_test(
                "CD Trajectory Interpolation",
                cd_in_middle,
                f"Interpolated CD: {interpolated_cd:.3f} (between {consonant_cd:.3f} and {dissonant_cd:.3f})"
            )
            
        except Exception as e:
            self.log_test("Integration with Individual Chords", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Phase 2 Consonant/Dissonant Interpolation Engine Tests")
        print("=" * 80)
        
        # Run test suites
        self.test_emotion_state_cd_creation()
        self.test_tension_curve_generation()
        self.test_enhanced_interpolation()
        self.test_progression_generation_with_cd()
        self.test_tension_curve_analysis()
        self.test_integration_with_individual_chords()
        
        # Print summary
        print("=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed_count = sum(1 for result in self.test_results if result["passed"])
        total_count = len(self.test_results)
        
        print(f"✅ Passed: {passed_count}/{total_count}")
        print(f"❌ Failed: {total_count - passed_count}/{total_count}")
        
        if passed_count == total_count:
            print("\n🎉 ALL TESTS PASSED! Phase 2 implementation is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the implementation.")
            
        # Print detailed results
        print("\n📋 Detailed Results:")
        for result in self.test_results:
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {result['test']}")
            if result["details"]:
                print(f"   {result['details']}")
        
        return passed_count == total_count


def main():
    """Main test execution function"""
    tester = InterpolationCDTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🔥 Phase 2: Interpolation Engine CD Integration - COMPLETE!")
        print("🎯 Ready to proceed with Phase 3: Advanced Features")
    else:
        print("\n🚨 Phase 2 tests failed. Please fix issues before continuing.")
    
    return success


if __name__ == "__main__":
    main() 