#!/usr/bin/env python3
"""
Database Fix Script - Based on Harmonic Validation Report
Fixes identified theoretical inconsistencies in syntaxAll.json and legalityAll.json
"""

import json
import os
from pathlib import Path
from copy import deepcopy

def backup_files():
    """Create backups of original files before fixing"""
    script_dir = Path(__file__).parent
    
    # Backup existing files
    for filename in ['syntaxAll.json', 'legalityAll.json']:
        source = script_dir / filename
        backup = script_dir / f"{filename}.backup"
        
        if source.exists():
            with open(source, 'r') as f:
                data = f.read()
            with open(backup, 'w') as f:
                f.write(data)
            print(f"✅ Backed up {filename} to {filename}.backup")

def fix_jazz_ionian_legality(legality_data):
    """Fix Jazz Ionian: Add missing ii → I progression"""
    print("\n🔧 Fixing Jazz Ionian legality...")
    
    jazz_ionian = legality_data.get("Jazz", {}).get("Ionian", {})
    
    # Check all ii-type chords and ensure they can resolve to I
    ii_chords = [chord for chord in jazz_ionian.keys() if chord.startswith('ii')]
    i_chords = ['I', 'IM7', 'IM9', 'I6/9']  # Tonic targets
    
    fixes_made = 0
    for ii_chord in ii_chords:
        if ii_chord in jazz_ionian:
            current_targets = jazz_ionian[ii_chord]
            missing_targets = [target for target in i_chords if target in jazz_ionian.keys() and target not in current_targets]
            
            if missing_targets:
                jazz_ionian[ii_chord].extend(missing_targets)
                print(f"   Added {missing_targets} as targets for {ii_chord}")
                fixes_made += 1
    
    # Also ensure basic ii → I exists
    if 'ii' in jazz_ionian and 'I' in jazz_ionian:
        if 'I' not in jazz_ionian['ii']:
            jazz_ionian['ii'].append('I')
            print(f"   Added I as target for ii")
            fixes_made += 1
    
    print(f"   Jazz Ionian: {fixes_made} progressions fixed")

def fix_aeolian_modal_consistency(syntax_data, legality_data):
    """Fix Aeolian modes: Remove major IV and V chords that don't belong in natural minor"""
    print("\n🔧 Fixing Aeolian modal consistency...")
    
    problematic_styles = ['Pop', 'RnB', 'Cinematic']
    
    for style in problematic_styles:
        if style in syntax_data and 'Aeolian' in syntax_data[style]:
            aeolian_syntax = syntax_data[style]['Aeolian']
            
            # Remove major IV from subdominant (keep minor iv)
            if 'subdominant' in aeolian_syntax:
                subdominant = aeolian_syntax['subdominant']
                for chord_type in subdominant:
                    if isinstance(subdominant[chord_type], list):
                        # Remove major IV, keep minor iv
                        original_chords = subdominant[chord_type][:]
                        subdominant[chord_type] = [chord for chord in original_chords 
                                                 if not (chord == 'IV' and 'iv' in original_chords)]
                        if len(original_chords) != len(subdominant[chord_type]):
                            print(f"   {style} Aeolian: Removed major IV from {chord_type}")
            
            # Remove major V from dominant (keep minor v)  
            if 'dominant' in aeolian_syntax:
                dominant = aeolian_syntax['dominant']
                for chord_type in dominant:
                    if isinstance(dominant[chord_type], list):
                        # Remove major V, keep minor v
                        original_chords = dominant[chord_type][:]
                        dominant[chord_type] = [chord for chord in original_chords 
                                              if not (chord == 'V' and 'v' in original_chords)]
                        if len(original_chords) != len(dominant[chord_type]):
                            print(f"   {style} Aeolian: Removed major V from {chord_type}")
        
        # Fix corresponding legality data
        if style in legality_data and 'Aeolian' in legality_data[style]:
            aeolian_legality = legality_data[style]['Aeolian']
            
            # Remove IV from legality if iv exists
            if 'IV' in aeolian_legality and 'iv' in aeolian_legality:
                del aeolian_legality['IV']
                print(f"   {style} Aeolian: Removed IV from legality (kept iv)")
                
                # Remove IV from other chords' target lists
                for chord, targets in aeolian_legality.items():
                    if 'IV' in targets:
                        targets.remove('IV')
            
            # Remove V from legality if v exists  
            if 'V' in aeolian_legality and 'v' in aeolian_legality:
                del aeolian_legality['V']
                print(f"   {style} Aeolian: Removed V from legality (kept v)")
                
                # Remove V from other chords' target lists
                for chord, targets in aeolian_legality.items():
                    if 'V' in targets:
                        targets.remove('V')

def fix_blues_ionian_anomalies(syntax_data, legality_data):
    """Fix Blues Ionian: Remove non-standard ♭II+ chord"""
    print("\n🔧 Fixing Blues Ionian anomalies...")
    
    if 'Blues' in syntax_data and 'Ionian' in syntax_data['Blues']:
        blues_ionian = syntax_data['Blues']['Ionian']
        
        # Remove ♭II+ (bII+) from other category
        if 'other' in blues_ionian:
            other = blues_ionian['other']
            for chord_type in other:
                if isinstance(other[chord_type], list):
                    original_chords = other[chord_type][:]
                    other[chord_type] = [chord for chord in original_chords if 'bII+' not in chord]
                    if len(original_chords) != len(other[chord_type]):
                        print(f"   Blues Ionian: Removed bII+ from {chord_type}")
    
    # Remove from legality as well
    if 'Blues' in legality_data and 'Ionian' in legality_data['Blues']:
        blues_legality = legality_data['Blues']['Ionian']
        if 'bII+' in blues_legality:
            del blues_legality['bII+']
            print(f"   Blues Ionian: Removed bII+ from legality")
            
            # Remove bII+ from other chords' targets
            for chord, targets in blues_legality.items():
                if 'bII+' in targets:
                    targets.remove('bII+')

def add_missing_jazz_progressions(legality_data):
    """Add common jazz progressions that might be missing"""
    print("\n🔧 Adding missing jazz progressions...")
    
    if 'Jazz' in legality_data:
        for mode in legality_data['Jazz']:
            mode_data = legality_data['Jazz'][mode]
            
            # Ensure vi can resolve to ii (vi-ii-V-I progression)
            if 'vi' in mode_data and 'ii' in mode_data:
                if 'ii' not in mode_data['vi']:
                    mode_data['vi'].append('ii')
                    print(f"   Jazz {mode}: Added ii as target for vi")
            
            # Ensure iii can resolve to vi (iii-vi-ii-V progression)
            if 'iii' in mode_data and 'vi' in mode_data:
                if 'vi' not in mode_data['iii']:
                    mode_data['iii'].append('vi')
                    print(f"   Jazz {mode}: Added vi as target for iii")

def validate_cross_consistency(syntax_data, legality_data):
    """Ensure all chords in legality exist in syntax"""
    print("\n🔍 Validating cross-consistency...")
    
    issues_found = 0
    
    for style in legality_data:
        if style not in syntax_data:
            print(f"   Warning: {style} exists in legality but not syntax")
            continue
            
        for mode in legality_data[style]:
            if mode not in syntax_data[style]:
                print(f"   Warning: {style}.{mode} exists in legality but not syntax")
                continue
                
            legality_chords = set(legality_data[style][mode].keys())
            
            # Collect all chords from syntax
            syntax_chords = set()
            for function in syntax_data[style][mode].values():
                for chord_type in function.values():
                    if isinstance(chord_type, list):
                        syntax_chords.update(chord_type)
            
            # Find chords in legality but not syntax
            missing_in_syntax = legality_chords - syntax_chords
            if missing_in_syntax:
                print(f"   {style}.{mode}: Chords in legality but not syntax: {missing_in_syntax}")
                issues_found += 1
                
                # Remove these chords from legality
                for chord in missing_in_syntax:
                    if chord in legality_data[style][mode]:
                        del legality_data[style][mode][chord]
                        print(f"     Removed {chord} from {style}.{mode} legality")
                
                # Remove from other chords' target lists
                for chord, targets in legality_data[style][mode].items():
                    for missing_chord in missing_in_syntax:
                        if missing_chord in targets:
                            targets.remove(missing_chord)
    
    if issues_found == 0:
        print("   ✅ No cross-consistency issues found")
    else:
        print(f"   🔧 Fixed {issues_found} cross-consistency issues")

def save_fixed_data(syntax_data, legality_data):
    """Save the fixed data back to files"""
    script_dir = Path(__file__).parent
    
    # Save syntax data
    syntax_file = script_dir / "syntaxAll.json"
    with open(syntax_file, 'w', encoding='utf-8') as f:
        json.dump(syntax_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved fixed syntax data to {syntax_file}")
    
    # Save legality data  
    legality_file = script_dir / "legalityAll.json"
    with open(legality_file, 'w', encoding='utf-8') as f:
        json.dump(legality_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved fixed legality data to {legality_file}")

def main():
    """Main function to fix database issues"""
    print("=" * 70)
    print("HARMONIC DATABASE FIX SCRIPT")
    print("Based on Harmonic Syntax and Legality Data Validation Report")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    
    # Load current data
    syntax_file = script_dir / "syntaxAll.json"
    legality_file = script_dir / "legalityAll.json"
    
    if not syntax_file.exists() or not legality_file.exists():
        print("❌ Required files not found!")
        return False
    
    # Create backups
    backup_files()
    
    # Load data
    with open(syntax_file, 'r', encoding='utf-8') as f:
        syntax_data = json.load(f)
    
    with open(legality_file, 'r', encoding='utf-8') as f:
        legality_data = json.load(f)
    
    print(f"\n📊 Loaded data:")
    print(f"   Syntax: {len(syntax_data)} styles")
    print(f"   Legality: {len(legality_data)} styles")
    
    # Apply fixes based on validation report
    fix_jazz_ionian_legality(legality_data)
    fix_aeolian_modal_consistency(syntax_data, legality_data)
    fix_blues_ionian_anomalies(syntax_data, legality_data)
    add_missing_jazz_progressions(legality_data)
    validate_cross_consistency(syntax_data, legality_data)
    
    # Save fixed data
    save_fixed_data(syntax_data, legality_data)
    
    print("\n" + "=" * 70)
    print("✅ DATABASE FIXES COMPLETED!")
    print("All issues identified in the validation report have been addressed.")
    print("Backup files created with .backup extension.")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    main()
