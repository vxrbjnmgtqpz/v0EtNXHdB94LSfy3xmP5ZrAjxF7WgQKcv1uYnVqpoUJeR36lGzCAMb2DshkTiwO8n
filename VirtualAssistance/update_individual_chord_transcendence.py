#!/usr/bin/env python3
"""
Update individual chord database to include Transcendence as 23rd emotion
"""

import json
import sys

def update_individual_chord_db():
    """Add Transcendence emotion to all chord entries in individual database"""
    
    try:
        # Load the current database
        with open('individual_chord_database_updated.json', 'r') as f:
            db = json.load(f)
        
        print(f"Current database has {db['database_info']['total_emotions']} emotions")
        print(f"Total chord entries: {len(db['chord_to_emotion_map'])}")
        
        # Update database info
        db['database_info']['total_emotions'] = 23
        db['database_info']['description'] = "Individual chord-to-emotion mapping database with consonant/dissonant profiles and 23-emotion system including Transcendence"
        db['database_info']['updated'] = "2025-07-03"
        
        # Add note about transcendence
        if "Added Transcendence as 23rd emotion" not in db['database_info']['architecture_notes']:
            db['database_info']['architecture_notes'].append("Added Transcendence as 23rd emotion for mystical/otherworldly chord mappings")
        
        # Update each chord entry to include Transcendence
        updated_count = 0
        for chord_entry in db['chord_to_emotion_map']:
            if 'Transcendence' not in chord_entry['emotion_weights']:
                # Determine appropriate transcendence weight based on chord characteristics
                transcendence_weight = calculate_transcendence_weight(chord_entry)
                chord_entry['emotion_weights']['Transcendence'] = transcendence_weight
                updated_count += 1
        
        print(f"Updated {updated_count} chord entries with Transcendence weights")
        
        # Save the updated database
        with open('individual_chord_database_updated.json', 'w') as f:
            json.dump(db, f, indent=2)
        
        print("✅ Successfully updated individual chord database with Transcendence emotion")
        return True
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_transcendence_weight(chord_entry):
    """Calculate appropriate transcendence weight for a chord based on its characteristics"""
    
    chord = chord_entry['chord']
    symbol = chord_entry['symbol']
    mode = chord_entry.get('mode_context', 'Ionian')
    weights = chord_entry['emotion_weights']
    
    # Base transcendence weight
    transcendence = 0.0
    
    # Lydian mode chords get higher transcendence (ethereal quality)
    if mode == 'Lydian':
        transcendence += 0.3
    
    # Whole tone, diminished, and augmented chords are more transcendent
    if any(x in symbol for x in ['°', '+', 'aug', 'dim']):
        transcendence += 0.4
    
    # Extended chords (7ths, 9ths, 11ths, 13ths) suggest transcendence
    if any(x in symbol for x in ['7', '9', '11', '13', 'maj7']):
        transcendence += 0.2
    
    # Suspended chords are ethereal
    if 'sus' in symbol:
        transcendence += 0.25
    
    # Add9, add11 chords are dreamy
    if 'add' in symbol:
        transcendence += 0.15
    
    # High aesthetic awe + wonder + reverence suggests transcendence
    aesthetic_component = weights.get('Aesthetic Awe', 0) * 0.3
    wonder_component = weights.get('Wonder', 0) * 0.4
    reverence_component = weights.get('Reverence', 0) * 0.2
    
    transcendence += aesthetic_component + wonder_component + reverence_component
    
    # Specific chord types that are transcendent:
    transcendent_chords = {
        'Isus4': 0.6,
        'IVsus2': 0.5,
        'Vsus4': 0.4,
        'vi7': 0.3,
        'iiø7': 0.7,  # Half-diminished very mystical
        'VIImaj7': 0.6,  # Lydian-type harmony
        'Iadd9': 0.4,
        'IVadd9': 0.5
    }
    
    if chord in transcendent_chords:
        transcendence = max(transcendence, transcendent_chords[chord])
    
    # Cap at 1.0
    return min(transcendence, 1.0)

if __name__ == "__main__":
    success = update_individual_chord_db()
    sys.exit(0 if success else 1)
