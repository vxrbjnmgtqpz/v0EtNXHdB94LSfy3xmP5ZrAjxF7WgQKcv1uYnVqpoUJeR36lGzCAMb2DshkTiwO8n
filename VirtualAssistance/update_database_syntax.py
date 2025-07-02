#!/usr/bin/env python3
"""
Update individual chord database to use 'harmonic_syntax' instead of 'mode_context'
"""

import json

def update_database():
    print("Updating individual chord database...")
    
    # Read the database
    with open('individual_chord_database.json', 'r') as f:
        data = json.load(f)
    
    # Update mode_context to harmonic_syntax
    updated_count = 0
    for chord in data['chord_to_emotion_map']:
        if 'mode_context' in chord:
            chord['harmonic_syntax'] = chord.pop('mode_context')
            updated_count += 1
    
    # Write back
    with open('individual_chord_database.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated {updated_count} chord entries: mode_context -> harmonic_syntax")

if __name__ == "__main__":
    update_database()
