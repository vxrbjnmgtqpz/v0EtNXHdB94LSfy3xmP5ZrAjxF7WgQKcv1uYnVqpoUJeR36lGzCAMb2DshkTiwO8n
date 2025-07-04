#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT REPORT FOR TRANSCENDENCE INTEGRATION
AND EDGE CASE ERROR HANDLING

This script performs a thorough audit of the entire VirtualAssistance system
to identify:
1. Missing "Transcendence" emotion integration
2. Hardcoded emotion lists that need updating
3. Edge case error handling gaps
4. Completeness issues
"""

import os
import json
import re
from typing import Dict, List, Any, Set
from datetime import datetime

class ComprehensiveAuditReport:
    def __init__(self):
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'transcendence_integration_issues': [],
            'hardcoded_emotion_lists': [],
            'error_handling_gaps': [],
            'completeness_issues': [],
            'recommendations': []
        }
        
    def run_full_audit(self):
        """Run the comprehensive audit"""
        print("🔍 Starting Comprehensive Audit...")
        print("=" * 60)
        
        # 1. Check Transcendence integration
        self._audit_transcendence_integration()
        
        # 2. Check for hardcoded emotion lists
        self._audit_hardcoded_emotions()
        
        # 3. Check error handling
        self._audit_error_handling()
        
        # 4. Check for completeness
        self._audit_completeness()
        
        # 5. Generate recommendations
        self._generate_recommendations()
        
        # 6. Save and display results
        self._save_and_display_results()
        
    def _audit_transcendence_integration(self):
        """Audit Transcendence integration across all files"""
        print("\n1. 🎯 Auditing Transcendence Integration...")
        
        # Check database first
        try:
            with open('emotion_progression_database.json', 'r') as f:
                db = json.load(f)
            
            if 'Transcendence' in db.get('emotions', {}):
                print("   ✅ Transcendence found in main database")
            else:
                self.audit_results['transcendence_integration_issues'].append(
                    "Transcendence emotion missing from emotion_progression_database.json"
                )
                print("   ❌ Transcendence NOT found in main database")
        except Exception as e:
            self.audit_results['transcendence_integration_issues'].append(
                f"Could not load emotion_progression_database.json: {e}"
            )
            
        # Check key Python files
        files_to_check = [
            'chord_progression_model.py',
            'individual_chord_model.py',
            'neural_progression_analyzer.py',
            'emotion_interpolation_engine.py',
            'integrated_chat_server.py',
            'enhanced_demo.py',
            'comprehensive_chord_demo.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                self._check_file_for_transcendence(file_path)
            else:
                self.audit_results['transcendence_integration_issues'].append(
                    f"File not found: {file_path}"
                )
                
    def _check_file_for_transcendence(self, file_path: str):
        """Check if a file includes Transcendence emotion"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for Transcendence references
            if 'Transcendence' in content or 'transcendence' in content:
                print(f"   ✅ {file_path} includes Transcendence")
            else:
                # Check if file has emotion lists that should include Transcendence
                if ('emotion_labels' in content or 'emotion_keywords' in content or 
                    'Joy' in content and 'Sadness' in content and 'Fear' in content):
                    self.audit_results['transcendence_integration_issues'].append(
                        f"{file_path} has emotion lists but missing Transcendence"
                    )
                    print(f"   ❌ {file_path} has emotion lists but missing Transcendence")
                else:
                    print(f"   ⚪ {file_path} does not appear to handle emotions directly")
        except Exception as e:
            self.audit_results['transcendence_integration_issues'].append(
                f"Error checking {file_path}: {e}"
            )
            
    def _audit_hardcoded_emotions(self):
        """Find hardcoded emotion lists that need updating"""
        print("\n2. 🔍 Auditing Hardcoded Emotion Lists...")
        
        # Files with known hardcoded emotion lists
        files_with_hardcoded = [
            'integrated_chat_server.py',
            'emotion_interpolation_engine.py', 
            'individual_chord_model.py'
        ]
        
        for file_path in files_with_hardcoded:
            if os.path.exists(file_path):
                self._find_hardcoded_emotions_in_file(file_path)
                
    def _find_hardcoded_emotions_in_file(self, file_path: str):
        """Find hardcoded emotion lists in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for emotion keyword patterns
            emotion_list_patterns = [
                r'emotion_keywords\s*=\s*\{',
                r'emotion_labels\s*=\s*\[',
                r'\["Joy",\s*"Sadness"',
                r'\["happy",\s*"joy"',
                r'self\.emotion_keywords\s*=\s*\['
            ]
            
            found_hardcoded = False
            for pattern in emotion_list_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_hardcoded = True
                    break
            
            if found_hardcoded:
                # Check if Transcendence is included
                if 'Transcendence' not in content:
                    self.audit_results['hardcoded_emotion_lists'].append(
                        f"{file_path} has hardcoded emotion lists missing Transcendence"
                    )
                    print(f"   ❌ {file_path} has hardcoded emotions missing Transcendence")
                else:
                    print(f"   ✅ {file_path} has hardcoded emotions but includes Transcendence")
            else:
                print(f"   ⚪ {file_path} does not appear to have hardcoded emotion lists")
                
        except Exception as e:
            self.audit_results['hardcoded_emotion_lists'].append(
                f"Error checking {file_path}: {e}"
            )
            
    def _audit_error_handling(self):
        """Audit error handling across the system"""
        print("\n3. 🛡️ Auditing Error Handling...")
        
        # Key files that should have robust error handling
        critical_files = [
            'integrated_chat_server.py',
            'chord_progression_model.py',
            'individual_chord_model.py',
            'neural_progression_analyzer.py',
            'emotion_interpolation_engine.py'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                self._audit_error_handling_in_file(file_path)
                
    def _audit_error_handling_in_file(self, file_path: str):
        """Audit error handling in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count try/except blocks
            try_count = len(re.findall(r'\btry\s*:', content))
            except_count = len(re.findall(r'\bexcept\s+', content))
            
            # Look for specific error handling patterns
            has_input_validation = bool(re.search(r'if\s+not\s+\w+.*:|if\s+len\s*\(.*\)\s*[<>=]', content))
            has_type_checking = bool(re.search(r'isinstance\s*\(|type\s*\(', content))
            has_key_error_handling = bool(re.search(r'except\s+KeyError|\.get\s*\(.*,.*\)', content))
            has_value_error_handling = bool(re.search(r'except\s+ValueError', content))
            
            error_handling_score = sum([
                try_count > 0,
                except_count > 0,
                has_input_validation,
                has_type_checking,
                has_key_error_handling,
                has_value_error_handling
            ])
            
            if error_handling_score < 3:
                self.audit_results['error_handling_gaps'].append(
                    f"{file_path} has weak error handling (score: {error_handling_score}/6)"
                )
                print(f"   ⚠️ {file_path} has weak error handling (score: {error_handling_score}/6)")
            else:
                print(f"   ✅ {file_path} has good error handling (score: {error_handling_score}/6)")
                
        except Exception as e:
            self.audit_results['error_handling_gaps'].append(
                f"Error auditing {file_path}: {e}"
            )
            
    def _audit_completeness(self):
        """Audit system completeness"""
        print("\n4. 📋 Auditing System Completeness...")
        
        # Check for required files
        required_files = [
            'emotion_progression_database.json',
            'individual_chord_database.json',
            'chord_progression_model.py',
            'individual_chord_model.py',
            'neural_progression_analyzer.py',
            'emotion_interpolation_engine.py',
            'integrated_chat_server.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path} exists")
            else:
                self.audit_results['completeness_issues'].append(
                    f"Required file missing: {file_path}"
                )
                print(f"   ❌ {file_path} missing")
                
        # Check test coverage
        test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
        if len(test_files) < 5:
            self.audit_results['completeness_issues'].append(
                f"Limited test coverage: only {len(test_files)} test files found"
            )
            print(f"   ⚠️ Limited test coverage: only {len(test_files)} test files found")
        else:
            print(f"   ✅ Good test coverage: {len(test_files)} test files found")
            
    def _generate_recommendations(self):
        """Generate specific recommendations based on audit findings"""
        print("\n5. 💡 Generating Recommendations...")
        
        # Transcendence integration recommendations
        if self.audit_results['transcendence_integration_issues']:
            self.audit_results['recommendations'].append({
                'category': 'Transcendence Integration',
                'priority': 'HIGH',
                'action': 'Update all hardcoded emotion lists to include Transcendence and its sub-emotions',
                'files': ['integrated_chat_server.py', 'emotion_interpolation_engine.py', 'individual_chord_model.py']
            })
            
        # Error handling recommendations
        if self.audit_results['error_handling_gaps']:
            self.audit_results['recommendations'].append({
                'category': 'Error Handling',
                'priority': 'MEDIUM',
                'action': 'Add comprehensive error handling for edge cases and invalid inputs',
                'files': [issue.split(' ')[0] for issue in self.audit_results['error_handling_gaps']]
            })
            
        # Completeness recommendations
        if self.audit_results['completeness_issues']:
            self.audit_results['recommendations'].append({
                'category': 'System Completeness',
                'priority': 'MEDIUM',
                'action': 'Address missing files and improve test coverage',
                'files': []
            })
            
        for rec in self.audit_results['recommendations']:
            print(f"   🎯 {rec['category']} ({rec['priority']}): {rec['action']}")
            
    def _save_and_display_results(self):
        """Save audit results and display summary"""
        # Save to file
        report_file = f"comprehensive_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 AUDIT SUMMARY")
        print("=" * 60)
        print(f"Transcendence Integration Issues: {len(self.audit_results['transcendence_integration_issues'])}")
        print(f"Hardcoded Emotion Lists: {len(self.audit_results['hardcoded_emotion_lists'])}")
        print(f"Error Handling Gaps: {len(self.audit_results['error_handling_gaps'])}")
        print(f"Completeness Issues: {len(self.audit_results['completeness_issues'])}")
        print(f"Total Recommendations: {len(self.audit_results['recommendations'])}")
        print(f"\nDetailed report saved to: {report_file}")
        
        # Display critical issues
        if self.audit_results['transcendence_integration_issues']:
            print("\n🚨 CRITICAL TRANSCENDENCE INTEGRATION ISSUES:")
            for issue in self.audit_results['transcendence_integration_issues']:
                print(f"   • {issue}")
                
        if self.audit_results['hardcoded_emotion_lists']:
            print("\n⚠️ HARDCODED EMOTION LISTS NEEDING UPDATE:")
            for issue in self.audit_results['hardcoded_emotion_lists']:
                print(f"   • {issue}")

if __name__ == "__main__":
    auditor = ComprehensiveAuditReport()
    auditor.run_full_audit()
