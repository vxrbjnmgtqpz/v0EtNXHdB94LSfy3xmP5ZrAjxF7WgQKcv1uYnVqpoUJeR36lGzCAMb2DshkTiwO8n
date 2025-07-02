#!/usr/bin/env python3
"""
Fix Chord Progression Database Based on Emotional Alignment Audit
Systematically corrects all flagged progressions according to audit recommendations
"""

import json
import os
from typing import Dict, List, Any

class ChordProgressionAuditor:
    """Apply audit fixes to chord progression database"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.fixes_applied = []
        
        # Load the current database
        with open(database_path, 'r') as f:
            self.database = json.load(f)
    
    def apply_audit_fixes(self):
        """Apply all fixes identified in the emotional alignment audit"""
        
        print("=== Applying Chord Progression Audit Fixes ===\n")
        
        # JOY FIXES
        self.fix_joy_progressions()
        
        # SADNESS FIXES
        self.fix_sadness_progressions()
        
        # TRUST FIXES
        self.fix_trust_progressions()
        
        # LOVE FIXES
        self.fix_love_progressions()
        
        # ANGER FIXES
        self.fix_anger_progressions()
        
        # FEAR FIXES
        self.fix_fear_progressions()
        
        # DISGUST FIXES
        self.fix_disgust_progressions()
        
        # ANTICIPATION FIXES
        self.fix_anticipation_progressions()
        
        # SHAME FIXES
        self.fix_shame_progressions()
        
        # ENVY FIXES
        self.fix_envy_progressions()
        
        # Note: Surprise and Aesthetic Awe were noted as having no issues
        
        self.save_fixed_database()
        self.print_summary()
    
    def fix_joy_progressions(self):
        """Fix Joy progressions identified in audit"""
        
        # joy_008: I–iii–IV–V → Replace iii with ii for less somber tone
        self.fix_progression("Joy", "joy_008", ["I", "ii", "IV", "V"], 
                           "Replaced iii with ii to reduce somber minor chord impact")
        
        # joy_009: I–V–vi–iii–IV–I–ii–V → Replace iii with ii to lighten tone
        self.fix_progression("Joy", "joy_009", ["I", "V", "vi", "ii", "IV", "I", "ii", "V"],
                           "Replaced iii with ii to reduce melancholy and maintain joy")
        
        # joy_011: I–vi–ii–V → Replace vi with IV for warmer, more hopeful sound
        self.fix_progression("Joy", "joy_011", ["I", "IV", "ii", "V"],
                           "Replaced vi with IV for warmer, more hopeful progression")
    
    def fix_sadness_progressions(self):
        """Fix Sadness progressions identified in audit"""
        
        # sad_003: i–♭VII–♭VI–♭VII → End on i instead of ♭VII for proper melancholy resolution
        self.fix_progression("Sadness", "sad_003", ["i", "♭VII", "♭VI", "i"],
                           "Changed ending from ♭VII to i for proper melancholy resolution")
        
        # sad_007: i–iv–i–♭VII → End on i instead of ♭VII to maintain sorrowful mood
        self.fix_progression("Sadness", "sad_007", ["i", "iv", "i", "i"],
                           "Changed ending from ♭VII to i to maintain unresolved, mournful feeling")
    
    def fix_trust_progressions(self):
        """Fix Trust progressions identified in audit"""
        
        # trust_009: i–IV–vi°–ii → Replace vi° with V for less dissonance/fear
        self.fix_progression("Trust", "trust_009", ["i", "IV", "V", "ii"],
                           "Replaced vi° with V to remove fear-laden dissonance")
        
        # trust_004: i–IV–ii–♭VII → Add resolution to i at end
        self.fix_progression("Trust", "trust_004", ["i", "IV", "ii", "♭VII", "i"],
                           "Added resolution to i for more grounded, supportive feel")
    
    def fix_love_progressions(self):
        """Fix Love progressions identified in audit"""
        
        # love_003: I–♭VII–v–I → Use V (major) instead of v (minor) for brighter resolution
        self.fix_progression("Love", "love_003", ["I", "♭VII", "V", "I"],
                           "Changed v to V for brighter, more soulful resolution")
        
        # love_005: I–♭VII–IV–v → Add resolution to I at end
        self.fix_progression("Love", "love_005", ["I", "♭VII", "IV", "v", "I"],
                           "Added resolution to I for warmer, more resolved feeling")
    
    def fix_anger_progressions(self):
        """Fix Anger progressions identified in audit"""
        
        # anger_006: I–♭III–♭II–I → Use ♭iii (minor) instead of ♭III for more menacing tone
        self.fix_progression("Anger", "anger_006", ["I", "♭iii", "♭II", "I"],
                           "Changed ♭III to ♭iii (minor) for more menacing, less stable sound")
        
        # anger_003: I–V–♭II–I → Use v (minor) instead of V for maintained tension
        self.fix_progression("Anger", "anger_003", ["I", "v", "♭II", "I"],
                           "Changed V to v (minor) to maintain instability and aggression")
    
    def fix_fear_progressions(self):
        """Fix Fear progressions identified in audit"""
        
        # fear_010: i–♭VI–♭VII–i → Use ♭vi (minor) instead of ♭VI for more ominous tone
        self.fix_progression("Fear", "fear_010", ["i", "♭vi", "♭VII", "i"],
                           "Changed ♭VI to ♭vi (minor) to remove warm consonance")
        
        # fear_005: i–♭II–♭VI–♭VII → Change ♭VI to ♭vi and ensure proper resolution
        self.fix_progression("Fear", "fear_005", ["i", "♭II", "♭vi", "i"],
                           "Changed ♭VI to ♭vi and ended on i for maximum tension resolution")
    
    def fix_disgust_progressions(self):
        """Fix Disgust progressions identified in audit"""
        
        # disgust_004: ♭v–i°–♭VI–♭II → Change ♭VI to ♭vi and end on i° for unresolved feeling
        self.fix_progression("Disgust", "disgust_004", ["♭v", "i°", "♭vi", "i°"],
                           "Changed ♭VI to ♭vi and ended on i° for consistently unstable sound")
        
        # disgust_007: ♭II–♭v–♭VI–i° → Change ♭VI to ♭vi for consistent sourness
        self.fix_progression("Disgust", "disgust_007", ["♭II", "♭v", "♭vi", "i°"],
                           "Changed ♭VI to ♭vi to maintain consistently sour harmony")
    
    def fix_anticipation_progressions(self):
        """Fix Anticipation progressions identified in audit"""
        
        # anticipation_002: i–IV–V–i → Replace IV with ii° for more ambiguous tension
        self.fix_progression("Anticipation", "anticipation_002", ["i", "ii°", "V", "i"],
                           "Replaced IV with ii° to maintain suspense and unresolved hope")
    
    def fix_shame_progressions(self):
        """Fix Shame progressions identified in audit"""
        
        # shame_010: i–♭III+–iv–V → Use regular ♭III instead of augmented for focused tragedy
        self.fix_progression("Shame", "shame_010", ["i", "♭III", "iv", "V"],
                           "Changed ♭III+ to ♭III to emphasize tragedy over cosmic drama")
    
    def fix_envy_progressions(self):
        """Fix Envy progressions identified in audit"""
        
        # Note: Audit suggested V7 instead of V, but we'll use V7 notation
        # envy_009: i–V–♭II–♯iv° → Use V7 for more complex, less triumphant sound
        self.fix_progression("Envy", "envy_009", ["i", "V7", "♭II", "♯iv°"],
                           "Changed V to V7 for more exotic, less straightforward dominant")
    
    def fix_progression(self, emotion: str, progression_id: str, new_chords: List[str], reason: str):
        """Fix a specific progression in the database"""
        
        emotion_data = self.database["emotions"][emotion]
        progressions = emotion_data["progression_pool"]
        
        # Find the progression to fix
        for i, prog in enumerate(progressions):
            if prog["progression_id"] == progression_id:
                old_chords = prog["chords"].copy()
                prog["chords"] = new_chords
                
                fix_record = {
                    "emotion": emotion,
                    "progression_id": progression_id,
                    "old_chords": old_chords,
                    "new_chords": new_chords,
                    "reason": reason
                }
                
                self.fixes_applied.append(fix_record)
                
                print(f"✓ Fixed {progression_id}: {old_chords} → {new_chords}")
                print(f"  Reason: {reason}\n")
                return
        
        print(f"✗ Could not find progression {progression_id} in {emotion}")
    
    def save_fixed_database(self):
        """Save the fixed database"""
        
        # Create backup of original
        backup_path = self.database_path.replace('.json', '_backup.json')
        if not os.path.exists(backup_path):
            with open(self.database_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)
            print(f"✓ Created backup: {backup_path}")
        
        # Save fixed version
        with open(self.database_path, 'w') as f:
            json.dump(self.database, f, indent=2)
        
        print(f"✓ Saved fixed database: {self.database_path}")
    
    def print_summary(self):
        """Print summary of all fixes applied"""
        
        print(f"\n=== AUDIT FIXES SUMMARY ===")
        print(f"Total fixes applied: {len(self.fixes_applied)}")
        
        by_emotion = {}
        for fix in self.fixes_applied:
            emotion = fix["emotion"]
            if emotion not in by_emotion:
                by_emotion[emotion] = 0
            by_emotion[emotion] += 1
        
        print("\nFixes by emotion:")
        for emotion, count in sorted(by_emotion.items()):
            print(f"  {emotion}: {count} fixes")
        
        print("\nAll fixes have been applied to improve emotional alignment!")

def main():
    """Main function to apply audit fixes"""
    
    database_path = "/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance/emotion_progression_database.json"
    
    if not os.path.exists(database_path):
        print(f"Error: Database file not found: {database_path}")
        return
    
    auditor = ChordProgressionAuditor(database_path)
    auditor.apply_audit_fixes()

if __name__ == "__main__":
    main()
