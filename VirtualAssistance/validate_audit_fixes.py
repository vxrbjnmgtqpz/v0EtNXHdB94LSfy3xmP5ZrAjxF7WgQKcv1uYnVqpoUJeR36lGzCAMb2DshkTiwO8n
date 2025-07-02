#!/usr/bin/env python3
"""
Validation script for audit fixes in the individual chord database.
Tests specific chords mentioned in the audit to ensure corrections were applied.
"""

import sys
import json
sys.path.append('/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance')

def validate_audit_fixes():
    """Validate that all audit fixes have been properly applied."""
    print("=== VALIDATING AUDIT FIXES ===")
    
    # Load the database directly
    with open('/Users/timothydowler/Projects/MIDIp2p/VirtualAssistance/individual_chord_database.json', 'r') as f:
        data = json.load(f)
    
    chord_map = data['chord_to_emotion_map']
    
    # Helper function to find chord by context and roman numeral
    def find_chord(mode_context, style_context, roman_numeral):
        for chord in chord_map:
            if (chord['mode_context'] == mode_context and 
                chord['style_context'] == style_context and 
                chord['chord'] == roman_numeral):
                return chord
        return None
    
    print("🔍 Checking Ionian Mode Fixes:")
    
    # 1. Ionian iii (Em) - should have reduced Surprise, increased Sadness/Trust
    iii_chord = find_chord('Ionian', 'Classical', 'iii')
    if iii_chord:
        print(f"  iii chord (Em): Surprise={iii_chord['emotion_weights']['Surprise']:.1f}, Sadness={iii_chord['emotion_weights']['Sadness']:.1f}, Trust={iii_chord['emotion_weights']['Trust']:.1f}")
        if iii_chord['emotion_weights']['Surprise'] <= 0.3 and iii_chord['emotion_weights']['Sadness'] >= 0.4:
            print("  ✅ Ionian iii fix applied correctly")
        else:
            print("  ❌ Ionian iii fix not applied correctly")
    
    # 2. Ionian ii (Dm) - should have reduced Trust, increased Sadness
    ii_chord = find_chord('Ionian', 'Classical', 'ii')
    if ii_chord:
        print(f"  ii chord (Dm): Trust={ii_chord['emotion_weights']['Trust']:.1f}, Sadness={ii_chord['emotion_weights']['Sadness']:.1f}")
        if ii_chord['emotion_weights']['Trust'] <= 0.4 and ii_chord['emotion_weights']['Sadness'] >= 0.3:
            print("  ✅ Ionian ii fix applied correctly")
        else:
            print("  ❌ Ionian ii fix not applied correctly")
    
    print("\\n🔍 Checking Aeolian Mode Fixes:")
    
    # 3. Aeolian ♭VII (G) - should have added Joy
    bvii_chord = find_chord('Aeolian', 'Classical', '♭VII')
    if bvii_chord:
        print(f"  ♭VII chord (G): Joy={bvii_chord['emotion_weights']['Joy']:.1f}")
        if bvii_chord['emotion_weights']['Joy'] > 0.1:
            print("  ✅ Aeolian ♭VII fix applied correctly")
        else:
            print("  ❌ Aeolian ♭VII fix not applied correctly")
    
    # 4. Aeolian ♭III (C) - should have added Joy
    biii_chord = find_chord('Aeolian', 'Classical', '♭III')
    if biii_chord:
        print(f"  ♭III chord (C): Joy={biii_chord['emotion_weights']['Joy']:.1f}")
        if biii_chord['emotion_weights']['Joy'] > 0.1:
            print("  ✅ Aeolian ♭III fix applied correctly")
        else:
            print("  ❌ Aeolian ♭III fix not applied correctly")
    
    print("\\n🔍 Checking Augmented Chord Fixes:")
    
    # 5. Check all ♭III+ chords for reduced Envy, increased Surprise/Anticipation
    aug_contexts = [('Harmonic Minor', 'Classical'), ('Melodic Minor', 'Classical'), ('Hungarian Minor', 'Classical')]
    for mode_context, style_context in aug_contexts:
        aug_chord = find_chord(mode_context, style_context, '♭III+')
        if aug_chord:
            print(f"  {mode_context} ♭III+: Envy={aug_chord['emotion_weights']['Envy']:.1f}, Surprise={aug_chord['emotion_weights']['Surprise']:.1f}")
            if aug_chord['emotion_weights']['Envy'] <= 0.4 and aug_chord['emotion_weights']['Surprise'] >= 0.6:
                print(f"  ✅ {mode_context} ♭III+ fix applied correctly")
            else:
                print(f"  ❌ {mode_context} ♭III+ fix not applied correctly")
    
    print("\\n🔍 Checking V7 Chord Fixes:")
    
    # 6. Check V7 chords in minor contexts for added Joy/Trust
    v7_contexts = [('Harmonic Minor', 'Classical'), ('Melodic Minor', 'Classical'), ('Hungarian Minor', 'Classical')]
    for mode_context, style_context in v7_contexts:
        v7_chord = find_chord(mode_context, style_context, 'V7')
        if v7_chord:
            print(f"  {mode_context} V7: Joy={v7_chord['emotion_weights']['Joy']:.1f}, Trust={v7_chord['emotion_weights']['Trust']:.1f}")
            if v7_chord['emotion_weights']['Joy'] > 0.1 or v7_chord['emotion_weights']['Trust'] > 0.05:
                print(f"  ✅ {mode_context} V7 fix applied correctly")
            else:
                print(f"  ❌ {mode_context} V7 fix not applied correctly")
    
    print("\\n🔍 Checking Blues Mode Fixes:")
    
    # 7. Check Blues major chords for added Joy
    blues_majors = ['♭VII', '♭III', '♭VI']
    for roman in blues_majors:
        blues_chord = find_chord('Aeolian', 'Blues', roman)  # Blues chords are typically in Aeolian mode
        if blues_chord:
            print(f"  Blues {roman}: Joy={blues_chord['emotion_weights']['Joy']:.1f}")
            if blues_chord['emotion_weights']['Joy'] > 0.1:
                print(f"  ✅ Blues {roman} fix applied correctly")
            else:
                print(f"  ❌ Blues {roman} fix not applied correctly")
    
    print("\\n🔍 Checking Locrian Mode Fixes:")
    
    # 8. Check Locrian major chords for moderated negative emotions and added positive ones
    locrian_majors = ['♭II', '♭V', '♭VII']
    for roman in locrian_majors:
        loc_chord = find_chord('Locrian', 'Classical', roman)
        if loc_chord:
            anger = loc_chord['emotion_weights']['Anger']
            fear = loc_chord['emotion_weights']['Fear'] 
            joy = loc_chord['emotion_weights']['Joy']
            trust = loc_chord['emotion_weights']['Trust']
            print(f"  Locrian {roman}: Anger={anger:.1f}, Fear={fear:.1f}, Joy={joy:.1f}, Trust={trust:.1f}")
            if (anger < 0.8 or fear < 0.8) and (joy > 0.1 or trust > 0.1):
                print(f"  ✅ Locrian {roman} fix applied correctly")
            else:
                print(f"  ❌ Locrian {roman} fix not applied correctly")
    
    print("\\n✅ AUDIT VALIDATION COMPLETE")

if __name__ == "__main__":
    validate_audit_fixes()
