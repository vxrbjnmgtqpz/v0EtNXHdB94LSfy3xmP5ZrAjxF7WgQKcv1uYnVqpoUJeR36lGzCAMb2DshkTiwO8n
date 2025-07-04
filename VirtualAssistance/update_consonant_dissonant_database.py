#!/usr/bin/env python3
"""
Update Individual Chord Database with Consonant/Dissonant Profiles
Upgrades from 12-emotion to 22-emotion system and adds consonant/dissonant metadata
"""

import json
import copy
from datetime import datetime
from typing import Dict, List, Any

class ConsonantDissonantDatabaseUpdater:
    def __init__(self):
        # Complete 22-emotion system
        self.emotion_labels = [
            "Joy", "Sadness", "Fear", "Anger", "Disgust", "Surprise", 
            "Trust", "Anticipation", "Shame", "Love", "Envy", "Aesthetic Awe", "Malice",
            "Arousal", "Guilt", "Reverence", "Wonder", "Dissociation", 
            "Empowerment", "Belonging", "Ideology", "Gratitude"
        ]
        
        # Consonant/Dissonant mappings based on framework
        self.chord_consonance_map = {
            # Major triads (consonant)
            "I": {"base": 0.2, "description": "Perfect consonance, foundational stability"},
            "IV": {"base": 0.2, "description": "Stable consonance, subdominant resolution"},
            "V": {"base": 0.25, "description": "Stable with anticipation, dominant function"},
            "♭VII": {"base": 0.3, "description": "Modal consonance, stable in context"},
            "♭VI": {"base": 0.3, "description": "Modal consonance, warm and stable"},
            "♭III": {"base": 0.3, "description": "Modal consonance, warm modulation"},
            
            # Minor triads (consonant)
            "i": {"base": 0.3, "description": "Minor consonance, darker stability"},
            "ii": {"base": 0.3, "description": "Minor consonance, subdominant minor"},
            "iii": {"base": 0.35, "description": "Minor consonance, mediant function"},
            "iv": {"base": 0.3, "description": "Minor consonance, subdominant function"},
            "v": {"base": 0.35, "description": "Minor consonance, modal dominant"},
            "vi": {"base": 0.3, "description": "Minor consonance, relative minor"},
            "♭ii": {"base": 0.45, "description": "Neapolitan, mild dissonance"},
            
            # Diminished chords (highly dissonant)
            "vii°": {"base": 0.75, "description": "Diminished, dramatic tension"},
            "ii°": {"base": 0.75, "description": "Diminished, requires resolution"},
            "♯iv°": {"base": 0.8, "description": "Augmented fourth, tritone tension"},
            "i°": {"base": 0.85, "description": "Diminished tonic, extreme instability"},
            "dim7": {"base": 0.75, "description": "Diminished seventh, classic tension"},
            "♯ii°": {"base": 0.8, "description": "Chromatic diminished, sharp tension"},
            
            # Augmented chords (highly dissonant)
            "♯III+": {"base": 0.7, "description": "Augmented mediant, unstable expansion"},
            "aug": {"base": 0.7, "description": "Augmented triad, mysterious instability"},
            "I+": {"base": 0.65, "description": "Augmented tonic, unstable resolution"},
            
            # Seventh chords (moderately dissonant)
            "maj7": {"base": 0.45, "description": "Major seventh, sophisticated harmony"},
            "7": {"base": 0.55, "description": "Dominant seventh, classic tension-resolution"},
            "m7": {"base": 0.4, "description": "Minor seventh, smooth jazz harmony"},
            "min7": {"base": 0.4, "description": "Minor seventh, versatile harmony"},
            "mM7": {"base": 0.65, "description": "Minor-major seventh, complex emotion"},
            "I7": {"base": 0.5, "description": "Major seventh on tonic, blues character"},
            "♭VII7": {"base": 0.45, "description": "Modal seventh, rock/blues harmony"},
            
            # Extended chords (moderately dissonant)
            "add9": {"base": 0.5, "description": "Added ninth, contemporary openness"},
            "9": {"base": 0.6, "description": "Ninth chord, extended harmony"},
            "sus2": {"base": 0.35, "description": "Suspended second, stable suspension"},
            "sus4": {"base": 0.35, "description": "Suspended fourth, stable suspension"},
            
            # Altered chords (highly dissonant)
            "7alt": {"base": 0.8, "description": "Altered dominant, jazz sophistication"},
            "7♯5": {"base": 0.75, "description": "Dominant sharp five, augmented tension"},
            "7♭5": {"base": 0.7, "description": "Dominant flat five, diminished tension"},
            "7♯9": {"base": 0.8, "description": "Dominant sharp nine, Hendrix chord"},
            "7♭9": {"base": 0.75, "description": "Dominant flat nine, dark tension"}
        }
        
        # Context modifiers for different styles
        self.context_modifiers = {
            "Classical": 1.0,
            "Jazz": 0.8,
            "Blues": 0.7,
            "Rock": 0.9,
            "Pop": 0.9,
            "Folk": 0.95,
            "R&B": 0.8,
            "Cinematic": 0.85,
            "Experimental": 0.5
        }
        
        # Emotional resonance mapping
        self.emotion_consonance_resonance = {
            "Joy": {"consonant": 0.9, "dissonant": 0.2},
            "Sadness": {"consonant": 0.3, "dissonant": 0.7},
            "Fear": {"consonant": 0.2, "dissonant": 0.9},
            "Anger": {"consonant": 0.1, "dissonant": 0.95},
            "Disgust": {"consonant": 0.1, "dissonant": 0.9},
            "Surprise": {"consonant": 0.4, "dissonant": 0.6},
            "Trust": {"consonant": 0.95, "dissonant": 0.1},
            "Anticipation": {"consonant": 0.4, "dissonant": 0.6},
            "Shame": {"consonant": 0.3, "dissonant": 0.7},
            "Love": {"consonant": 0.8, "dissonant": 0.3},
            "Envy": {"consonant": 0.2, "dissonant": 0.8},
            "Aesthetic Awe": {"consonant": 0.6, "dissonant": 0.4},
            "Malice": {"consonant": 0.1, "dissonant": 0.95},
            "Arousal": {"consonant": 0.4, "dissonant": 0.6},
            "Guilt": {"consonant": 0.2, "dissonant": 0.8},
            "Reverence": {"consonant": 0.8, "dissonant": 0.2},
            "Wonder": {"consonant": 0.6, "dissonant": 0.4},
            "Dissociation": {"consonant": 0.1, "dissonant": 0.9},
            "Empowerment": {"consonant": 0.7, "dissonant": 0.3},
            "Belonging": {"consonant": 0.8, "dissonant": 0.2},
            "Ideology": {"consonant": 0.6, "dissonant": 0.4},
            "Gratitude": {"consonant": 0.9, "dissonant": 0.1}
        }
    
    def load_database(self, filepath: str) -> Dict[str, Any]:
        """Load the existing chord database"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    
    def update_emotion_weights_to_22(self, old_weights: Dict[str, float]) -> Dict[str, float]:
        """Update emotion weights from 12-emotion to 22-emotion system"""
        new_weights = {}
        
        # Copy existing emotions
        for emotion in self.emotion_labels:
            if emotion in old_weights:
                new_weights[emotion] = old_weights[emotion]
            else:
                # Add new emotions with default values based on existing patterns
                if emotion == "Malice":
                    new_weights[emotion] = (old_weights.get("Anger", 0) + old_weights.get("Disgust", 0)) * 0.5
                elif emotion == "Arousal":
                    new_weights[emotion] = (old_weights.get("Anticipation", 0) + old_weights.get("Surprise", 0)) * 0.5
                elif emotion == "Guilt":
                    new_weights[emotion] = (old_weights.get("Shame", 0) + old_weights.get("Sadness", 0)) * 0.5
                elif emotion == "Reverence":
                    new_weights[emotion] = (old_weights.get("Trust", 0) + old_weights.get("Aesthetic Awe", 0)) * 0.5
                elif emotion == "Wonder":
                    new_weights[emotion] = (old_weights.get("Surprise", 0) + old_weights.get("Aesthetic Awe", 0)) * 0.5
                elif emotion == "Dissociation":
                    new_weights[emotion] = (old_weights.get("Fear", 0) + old_weights.get("Sadness", 0)) * 0.4
                elif emotion == "Empowerment":
                    new_weights[emotion] = (old_weights.get("Joy", 0) + old_weights.get("Trust", 0)) * 0.5
                elif emotion == "Belonging":
                    new_weights[emotion] = (old_weights.get("Love", 0) + old_weights.get("Trust", 0)) * 0.5
                elif emotion == "Ideology":
                    new_weights[emotion] = (old_weights.get("Trust", 0) + old_weights.get("Anticipation", 0)) * 0.4
                elif emotion == "Gratitude":
                    new_weights[emotion] = (old_weights.get("Joy", 0) + old_weights.get("Love", 0)) * 0.5
                else:
                    new_weights[emotion] = 0.0
        
        return new_weights
    
    def get_consonant_dissonant_profile(self, chord_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consonant/dissonant profile for a chord"""
        chord_symbol = chord_info.get("chord", "")
        emotion_weights = chord_info.get("emotion_weights", {})
        
        # Get base consonance value
        base_consonance = 0.4  # Default moderate consonance
        description = "Moderate consonance"
        
        # Try to match chord symbol with known consonance values
        for chord_pattern, consonance_info in self.chord_consonance_map.items():
            if chord_pattern in chord_symbol or chord_symbol.startswith(chord_pattern):
                base_consonance = consonance_info["base"]
                description = consonance_info["description"]
                break
        
        # Calculate emotional resonance
        emotional_resonance = {}
        for emotion, weight in emotion_weights.items():
            if emotion in self.emotion_consonance_resonance:
                if base_consonance <= 0.4:  # Consonant chord
                    resonance = self.emotion_consonance_resonance[emotion]["consonant"]
                else:  # Dissonant chord
                    resonance = self.emotion_consonance_resonance[emotion]["dissonant"]
                emotional_resonance[emotion] = resonance
        
        return {
            "base_value": base_consonance,
            "context_modifiers": self.context_modifiers.copy(),
            "emotional_resonance": emotional_resonance,
            "description": description
        }
    
    def update_database(self, input_file: str, output_file: str) -> bool:
        """Update the entire database with consonant/dissonant profiles"""
        try:
            # Load existing database
            database = self.load_database(input_file)
            if not database:
                print("Failed to load database")
                return False
            
            # Update database metadata
            database["database_info"]["version"] = "2.0"
            database["database_info"]["description"] = "Individual chord-to-emotion mapping database with consonant/dissonant profiles and 22-emotion system"
            database["database_info"]["total_emotions"] = 22
            database["database_info"]["updated"] = datetime.now().strftime("%Y-%m-%d")
            database["database_info"]["architecture_notes"].extend([
                "Updated to 22-emotion system for full compatibility",
                "Added consonant/dissonant profiles for each chord",
                "Includes context-aware consonance modifiers",
                "Supports emotional resonance with consonant/dissonant qualities"
            ])
            
            # Update each chord
            updated_chords = []
            for chord_info in database["chord_to_emotion_map"]:
                # Create updated chord copy
                updated_chord = copy.deepcopy(chord_info)
                
                # Update emotion weights to 22-emotion system
                updated_chord["emotion_weights"] = self.update_emotion_weights_to_22(chord_info["emotion_weights"])
                
                # Add style_context if missing
                if "style_context" not in updated_chord:
                    updated_chord["style_context"] = "Classical"
                
                # Add consonant/dissonant profile
                updated_chord["consonant_dissonant_profile"] = self.get_consonant_dissonant_profile(updated_chord)
                
                updated_chords.append(updated_chord)
            
            # Update database
            database["chord_to_emotion_map"] = updated_chords
            
            # Save updated database
            with open(output_file, 'w') as f:
                json.dump(database, f, indent=2)
            
            print(f"✅ Database updated successfully!")
            print(f"📊 Updated {len(updated_chords)} chords")
            print(f"🎵 Added consonant/dissonant profiles")
            print(f"💫 Upgraded to 22-emotion system")
            print(f"📝 Saved to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating database: {e}")
            return False

def main():
    print("🎵 Individual Chord Database Consonant/Dissonant Updater")
    print("=" * 60)
    
    updater = ConsonantDissonantDatabaseUpdater()
    
    # Update database
    input_file = "individual_chord_database.json"
    output_file = "individual_chord_database_updated.json"
    
    if updater.update_database(input_file, output_file):
        print("\n🎉 Database update completed successfully!")
        print("🔄 You can now test the updated database.")
    else:
        print("\n❌ Database update failed")

if __name__ == "__main__":
    main()
