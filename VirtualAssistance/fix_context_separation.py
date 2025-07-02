#!/usr/bin/env python3
"""
Fix context separation in individual chord database
Separate modal contexts from style contexts properly
"""

import json

def separate_contexts():
    # Load the database
    with open('individual_chord_database.json', 'r') as f:
        data = json.load(f)
    
    # Define modal vs style contexts
    modal_contexts = {
        'Ionian', 'Aeolian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Locrian',
        'Harmonic Minor', 'Melodic Minor', 'Hungarian Minor'
    }
    
    style_contexts = {
        'Jazz', 'Blues', 'Classical', 'Pop', 'Rock', 'Folk', 'RnB', 'Cinematic'
    }
    
    updated_count = 0
    
    # Process each chord
    for chord in data['chord_to_emotion_map']:
        current_context = chord['harmonic_syntax']
        
        if current_context in modal_contexts:
            # This is a modal context
            chord['mode_context'] = current_context
            chord['style_context'] = 'Classical'  # Default style for pure modal contexts
        elif current_context in style_contexts:
            # This is a style context
            chord['style_context'] = current_context
            # Infer mode from chord type for styles
            if current_context == 'Jazz':
                # Jazz typically uses more complex harmonies, often Ionian or Dorian
                chord['mode_context'] = 'Ionian'  # Default, could be refined
            elif current_context == 'Blues':
                # Blues often uses Mixolydian or minor pentatonic contexts
                chord['mode_context'] = 'Mixolydian' if chord['chord'].endswith('7') else 'Aeolian'
            else:
                chord['mode_context'] = 'Ionian'  # Safe default
        else:
            # Unknown context - use defaults
            chord['mode_context'] = 'Ionian'
            chord['style_context'] = 'Classical'
            print(f"Warning: Unknown context '{current_context}' for chord {chord['chord']}")
        
        # Remove the old combined field
        del chord['harmonic_syntax']
        
        # Update chord_id to reflect new structure
        chord['chord_id'] = f"{chord['style_context'].lower()}_{chord['mode_context'].lower()}_{chord['chord']}"
        
        updated_count += 1
    
    # Update database info
    data['database_info']['version'] = '1.3'
    data['database_info']['updated'] = '2025-07-02'
    data['database_info']['architecture_notes'].append(
        "Version 1.3: Separated mode_context (modal frameworks) from style_context (genre frameworks)"
    )
    
    # Save the updated database
    with open('individual_chord_database.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated {updated_count} chord entries to separate mode_context and style_context")
    print("✓ Database structure now properly separates modal and stylistic contexts")

if __name__ == "__main__":
    separate_contexts()
