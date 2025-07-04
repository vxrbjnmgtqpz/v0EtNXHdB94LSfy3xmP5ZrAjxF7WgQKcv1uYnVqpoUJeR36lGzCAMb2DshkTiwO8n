"""
Fix Chord Progression Alignment Issues

This script addresses the specific chord progression problems identified in the audit:
- Joy progressions with too many minor chords
- Sadness progressions ending on major chords
- Trust progressions with dissonant diminished chords
- Love progressions ending unresolved on minor chords
- Anger progressions with too-resolved major chords
- Fear progressions with consonant ♭VI major chords
- Disgust progressions with stabilizing major chords
- Anticipation progressions with overly stable resolutions
- Shame progressions with grandiose augmented chords
- Envy progressions with triumphant dominants
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ProgressionFix:
    """Represents a fix for a chord progression"""
    progression_id: str
    original_chords: List[str]
    fixed_chords: List[str]
    reason: str
    category: str

class ChordProgressionAlignmentFixer:
    """Fixes chord progression alignment issues based on audit findings"""
    
    def __init__(self, database_path: str = "emotion_progression_database.json"):
        self.database_path = database_path
        self.fixes_applied = []
        
    def load_database(self) -> Dict:
        """Load the emotion progression database"""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Database file {self.database_path} not found")
            return {}
    
    def save_database(self, database: Dict) -> None:
        """Save the updated database"""
        with open(self.database_path, 'w') as f:
            json.dump(database, f, indent=2)
    
    def fix_all_alignment_issues(self) -> List[ProgressionFix]:
        """Apply all chord progression fixes identified in the audit"""
        database = self.load_database()
        
        if not database:
            print("No database to fix")
            return []
        
        # Apply category-specific fixes
        self._fix_joy_progressions(database)
        self._fix_sadness_progressions(database)
        self._fix_trust_progressions(database)
        self._fix_love_progressions(database)
        self._fix_anger_progressions(database)
        self._fix_fear_progressions(database)
        self._fix_disgust_progressions(database)
        self._fix_anticipation_progressions(database)
        self._fix_shame_progressions(database)
        self._fix_envy_progressions(database)
        
        # Save updated database
        self.save_database(database)
        
        return self.fixes_applied
    
    def _fix_joy_progressions(self, database: Dict) -> None:
        """Fix Joy progressions with too many minor chords"""
        joy_emotion = database.get('emotions', {}).get('Joy', {})
        sub_emotions = joy_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix specific problematic progressions identified in audit
                if prog_id == 'joy_008' or chords == ["I", "iii", "IV", "V"]:
                    # Replace iii with ii for more upbeat feel
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "ii", "IV", "V"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced minor iii chord with ii to maintain joyful mood", "Joy"
                    ))
                
                elif prog_id == 'joy_009' or chords == ["I", "V", "vi", "iii", "IV", "I", "ii", "V"]:
                    # Swap iii for ii to reduce sadness
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "V", "vi", "ii", "IV", "I", "ii", "V"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced iii with ii to lighten emotional tone", "Joy"
                    ))
                
                elif prog_id == 'joy_011' or chords == ["I", "vi", "ii", "V"]:
                    # Replace vi with IV for warmer progression
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "IV", "ii", "V"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced vi with IV for warmer, more hopeful progression", "Joy"
                    ))
    
    def _fix_sadness_progressions(self, database: Dict) -> None:
        """Fix Sadness progressions ending on major chords"""
        sadness_emotion = database.get('emotions', {}).get('Sadness', {})
        sub_emotions = sadness_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix progressions ending on ♭VII major
                if prog_id == 'sad_003' or chords == ["i", "♭VII", "♭VI", "♭VII"]:
                    # End on i instead of ♭VII
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "♭VII", "♭VI", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ending from ♭VII to i to preserve melancholy mood", "Sadness"
                    ))
                
                elif prog_id == 'sad_007' or chords == ["i", "iv", "i", "♭VII"]:
                    # End on i instead of ♭VII
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "iv", "i", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ending from ♭VII to i to maintain sorrowful feel", "Sadness"
                    ))
    
    def _fix_trust_progressions(self, database: Dict) -> None:
        """Fix Trust progressions with dissonant chords"""
        trust_emotion = database.get('emotions', {}).get('Trust', {})
        sub_emotions = trust_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix progression with diminished vi°
                if prog_id == 'trust_009' or chords == ["i", "IV", "vi°", "ii"]:
                    # Replace vi° with V for consonance
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "IV", "V", "ii"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced dissonant vi° with V to support trust feeling", "Trust"
                    ))
                
                elif prog_id == 'trust_004' or chords == ["i", "IV", "ii", "♭VII"]:
                    # Add resolution to i
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "IV", "ii", "♭VII", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Added resolution to i for more supportive ending", "Trust"
                    ))
    
    def _fix_love_progressions(self, database: Dict) -> None:
        """Fix Love progressions with unresolved minor endings"""
        love_emotion = database.get('emotions', {}).get('Love', {})
        sub_emotions = love_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix progressions with minor v creating sadness
                if prog_id == 'love_003' or chords == ["I", "♭VII", "v", "I"]:
                    # Use major V instead of minor v
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "♭VII", "V", "I"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed minor v to major V to reduce sadness", "Love"
                    ))
                
                elif prog_id == 'love_005' or chords == ["I", "♭VII", "IV", "v"]:
                    # Add resolution to I
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "♭VII", "IV", "V", "I"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Added resolution to I for warmer, more resolved ending", "Love"
                    ))
    
    def _fix_anger_progressions(self, database: Dict) -> None:
        """Fix Anger progressions with too-resolved major chords"""
        anger_emotion = database.get('emotions', {}).get('Anger', {})
        sub_emotions = anger_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix ♭III major being too stable
                if prog_id == 'anger_006' or chords == ["I", "♭III", "♭II", "I"]:
                    # Use minor ♭iii to maintain tension
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "♭iii", "♭II", "I"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ♭III to ♭iii to maintain aggressive tension", "Anger"
                    ))
                
                elif prog_id == 'anger_003' or chords == ["I", "V", "♭II", "I"]:
                    # Use minor v instead of major V
                    original_chords = chords.copy()
                    progression['chords'] = ["I", "v", "♭II", "I"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed V to v to reduce tonal resolution and maintain instability", "Anger"
                    ))
    
    def _fix_fear_progressions(self, database: Dict) -> None:
        """Fix Fear progressions with consonant ♭VI major chords"""
        fear_emotion = database.get('emotions', {}).get('Fear', {})
        sub_emotions = fear_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix ♭VI major providing too much relief
                if prog_id == 'fear_010' or chords == ["i", "♭VI", "♭VII", "i"]:
                    # Use ♭vi minor instead of ♭VI major
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "♭vi", "♭VII", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ♭VI to ♭vi to maintain anxiety without relief", "Fear"
                    ))
                
                elif prog_id == 'fear_005' or chords == ["i", "♭II", "♭VI", "♭VII"]:
                    # Use ♭vi and end on i for better resolution
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "♭II", "♭vi", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ♭VI to ♭vi and ended on i for consistent fear", "Fear"
                    ))
    
    def _fix_disgust_progressions(self, database: Dict) -> None:
        """Fix Disgust progressions with stabilizing major chords"""
        disgust_emotion = database.get('emotions', {}).get('Disgust', {})
        sub_emotions = disgust_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix ♭VI major providing stability
                if prog_id == 'disgust_004' or chords == ["♭v", "i°", "♭VI", "♭II"]:
                    # Use ♭vi minor and end on i° for instability
                    original_chords = chords.copy()
                    progression['chords'] = ["♭v", "i°", "♭vi", "i°"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ♭VI to ♭vi and ended on i° for consistent dissonance", "Disgust"
                    ))
                
                elif prog_id == 'disgust_007' or chords == ["♭II", "♭v", "♭VI", "i°"]:
                    # Use ♭vi minor for consistency
                    original_chords = chords.copy()
                    progression['chords'] = ["♭II", "♭v", "♭vi", "i°"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed ♭VI to ♭vi to maintain hollow, disoriented atmosphere", "Disgust"
                    ))
    
    def _fix_anticipation_progressions(self, database: Dict) -> None:
        """Fix Anticipation progressions with overly stable resolutions"""
        anticipation_emotion = database.get('emotions', {}).get('Anticipation', {})
        sub_emotions = anticipation_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix stable IV chord reducing tension
                if prog_id == 'anticipation_002' or chords == ["i", "IV", "V", "i"]:
                    # Use ii° instead of IV for more tension
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "ii°", "V", "i"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced stable IV with ii° to maintain suspense", "Anticipation"
                    ))
    
    def _fix_shame_progressions(self, database: Dict) -> None:
        """Fix Shame progressions with grandiose augmented chords"""
        shame_emotion = database.get('emotions', {}).get('Shame', {})
        sub_emotions = shame_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix augmented chord being too grandiose
                if prog_id == 'shame_010' or chords == ["i", "♭III+", "iv", "V"]:
                    # Use regular ♭III for more personal sorrow
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "♭III", "iv", "V"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Replaced ♭III+ with ♭III to focus on personal sorrow over drama", "Shame"
                    ))
    
    def _fix_envy_progressions(self, database: Dict) -> None:
        """Fix Envy progressions with triumphant dominants"""
        envy_emotion = database.get('emotions', {}).get('Envy', {})
        sub_emotions = envy_emotion.get('sub_emotions', {})
        
        for sub_emotion_name, sub_emotion_data in sub_emotions.items():
            if 'progression_pool' not in sub_emotion_data:
                continue
                
            for progression in sub_emotion_data['progression_pool']:
                prog_id = progression.get('progression_id', '')
                chords = progression.get('chords', [])
                
                # Fix V being too triumphant
                if prog_id == 'envy_009' or chords == ["i", "V", "♭II", "♯iv°"]:
                    # Use V7 for more complex, less triumphant sound
                    original_chords = chords.copy()
                    progression['chords'] = ["i", "V7", "♭II", "♯iv°"]
                    self.fixes_applied.append(ProgressionFix(
                        prog_id, original_chords, progression['chords'],
                        "Changed V to V7 to add complexity and reduce triumphant feel", "Envy"
                    ))
    
    def print_fix_summary(self) -> None:
        """Print a summary of all fixes applied"""
        if not self.fixes_applied:
            print("No fixes were applied.")
            return
        
        print(f"\n=== CHORD PROGRESSION ALIGNMENT FIXES SUMMARY ===")
        print(f"Total fixes applied: {len(self.fixes_applied)}\n")
        
        # Group by category
        by_category = {}
        for fix in self.fixes_applied:
            if fix.category not in by_category:
                by_category[fix.category] = []
            by_category[fix.category].append(fix)
        
        for category, fixes in by_category.items():
            print(f"{category.upper()} ({len(fixes)} fixes):")
            for fix in fixes:
                print(f"  • {fix.progression_id}: {' - '.join(fix.original_chords)} → {' - '.join(fix.fixed_chords)}")
                print(f"    Reason: {fix.reason}")
            print()

def main():
    """Run the chord progression alignment fixes"""
    print("🎼 Fixing Chord Progression Alignment Issues...")
    print("=" * 50)
    
    fixer = ChordProgressionAlignmentFixer()
    fixes = fixer.fix_all_alignment_issues()
    
    fixer.print_fix_summary()
    
    if fixes:
        print("✅ All alignment issues have been fixed!")
        print("The emotion progression database has been updated.")
    else:
        print("ℹ️  No progressions were found that matched the audit issues.")
        print("This might mean the database structure is different than expected.")

if __name__ == "__main__":
    main() 