#!/usr/bin/env python3
"""
Consolidate all individual syntax JSON files into a single syntaxAll.json file
Similar to how legalityAll.json was created from individual legality rules
"""

import json
import os
from pathlib import Path

def load_syntax_file(filename):
    """Load a syntax JSON file and return its data"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def consolidate_syntax_files():
    """Consolidate all syntax files into syntaxAll.json"""
    
    script_dir = Path(__file__).parent
    
    # Define the styles and their corresponding files
    styles = {
        "Blues": "syntaxBlues.json",
        "Jazz": "syntaxJazz.json", 
        "Classical": "syntaxClassical.json",
        "Pop": "syntaxPop.json",
        "Rock": "syntaxRock.json",
        "Folk": "syntaxFolk.json",
        "RnB": "syntaxRnB.json",
        "Cinematic": "syntaxCinematic.json"
    }
    
    # Initialize the consolidated structure
    consolidated_syntax = {}
    
    print("Consolidating syntax files...")
    print("=" * 50)
    
    # Process each style
    for style_name, filename in styles.items():
        filepath = script_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping {style_name}")
            continue
            
        print(f"Processing {style_name} from {filename}...")
        
        # Load the syntax data
        syntax_data = load_syntax_file(filepath)
        
        if syntax_data is None:
            print(f"Failed to load {filename}, skipping {style_name}")
            continue
            
        # Extract the modeChordData
        if "modeChordData" in syntax_data:
            consolidated_syntax[style_name] = syntax_data["modeChordData"]
            
            # Count modes and chords for this style
            modes = list(syntax_data["modeChordData"].keys())
            total_chords = 0
            for mode_data in syntax_data["modeChordData"].values():
                for function_data in mode_data.values():
                    for chord_type_data in function_data.values():
                        if isinstance(chord_type_data, list):
                            total_chords += len(chord_type_data)
            
            print(f"  └─ Loaded {len(modes)} modes with ~{total_chords} chord definitions")
        else:
            print(f"  └─ Warning: No modeChordData found in {filename}")
    
    print()
    print("Consolidation Summary:")
    print("=" * 50)
    
    # Generate summary statistics
    total_styles = len(consolidated_syntax)
    all_modes = set()
    total_chord_definitions = 0
    
    for style_name, style_data in consolidated_syntax.items():
        modes_in_style = set(style_data.keys())
        all_modes.update(modes_in_style)
        
        style_chord_count = 0
        for mode_data in style_data.values():
            for function_data in mode_data.values():
                for chord_type_data in function_data.values():
                    if isinstance(chord_type_data, list):
                        style_chord_count += len(chord_type_data)
        
        total_chord_definitions += style_chord_count
        print(f"{style_name:12}: {len(modes_in_style)} modes, ~{style_chord_count} chords")
    
    print(f"\nTotal: {total_styles} styles, {len(all_modes)} unique modes, ~{total_chord_definitions} chord definitions")
    print(f"Modes covered: {sorted(all_modes)}")
    
    # Save the consolidated file
    output_file = script_dir / "syntaxAll.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_syntax, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Successfully created {output_file}")
        print(f"📁 File size: {output_file.stat().st_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving consolidated file: {e}")
        return False

if __name__ == "__main__":
    print("Syntax Files Consolidation Tool")
    print("=" * 40)
    print()
    
    success = consolidate_syntax_files()
    
    if success:
        print("\n🎵 All syntax files have been successfully consolidated!")
        print("The new syntaxAll.json file contains all style-specific chord data.")
    else:
        print("\n❌ Consolidation failed. Please check the error messages above.")
