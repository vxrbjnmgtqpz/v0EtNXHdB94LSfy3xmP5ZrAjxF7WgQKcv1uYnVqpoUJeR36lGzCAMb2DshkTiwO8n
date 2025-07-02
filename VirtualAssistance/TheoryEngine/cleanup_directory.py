#!/usr/bin/env python3
"""
Directory Cleanup Script
Remove redundant individual files now that we have consolidated "All" files
Keeps the consolidated files and removes the bloat
"""

import os
import shutil
from pathlib import Path

def cleanup_directory():
    """Clean up redundant files while preserving consolidated data"""
    
    script_dir = Path(__file__).parent
    
    print("=" * 60)
    print("THEORY ENGINE DIRECTORY CLEANUP")
    print("=" * 60)
    print()
    
    # Files to remove (individual syntax files)
    syntax_files_to_remove = [
        "syntaxBlues.json",
        "syntaxJazz.json", 
        "syntaxClassical.json",
        "syntaxPop.json",
        "syntaxRock.json",
        "syntaxFolk.json",
        "syntaxRnB.json",
        "syntaxCinematic.json"
    ]
    
    # Individual legality files to remove (keeping only legalityAll.json)
    legality_files_to_remove = [
        "legalityClassical.json"  # Individual legality file
    ]
    
    # Old conversion scripts that are no longer needed
    conversion_scripts_to_remove = [
        "convert_all_js_to_json.py",
        "convert_all_js_to_json_fixed.py", 
        "convert_js_to_json.py",
        "convert_modulation.py",
        "convert_syntax_simple.py"
    ]
    
    # Old theory engine versions that are superseded
    old_engine_files_to_remove = [
        "solfege_theory_engine.py",
        "json_theory_engine.py",
        "SolfegeTheoryEngine.wl",
        "SolfegeTheoryEngine_Enhanced.wl"
    ]
    
    # Files to keep (critical files)
    files_to_keep = [
        "syntaxAll.json",           # Consolidated syntax data
        "legalityAll.json",         # Consolidated legality rules
        "EnhancedSolfegeTheoryEngine.wl",  # Main Wolfram engine
        "enhanced_solfege_theory_engine.py",  # Main Python interface
        "solfegeChords.json",       # Core chord definitions
        "modulation.json",          # Modulation data
        "generate_style_legalities.py",  # Legality generator (useful for regeneration)
        "consolidate_syntax_files.py",   # Syntax consolidator (useful for regeneration)
        "STYLE_LEGALITY_DEMO.py",
        "CONSOLIDATED_SYNTAX_DEMO.py",
        "STYLE_LEGALITY_IMPLEMENTATION_SUMMARY.md",
        "SYNTAX_CONSOLIDATION_SUMMARY.md",
        "SolfegeTheoryEngine.nb"    # Notebook version
    ]
    
    all_files_to_remove = (
        syntax_files_to_remove + 
        legality_files_to_remove + 
        conversion_scripts_to_remove + 
        old_engine_files_to_remove
    )
    
    print("📁 Files marked for removal:")
    print("-" * 30)
    
    # Group files by category for better output
    categories = {
        "Individual Syntax Files": syntax_files_to_remove,
        "Individual Legality Files": legality_files_to_remove, 
        "Old Conversion Scripts": conversion_scripts_to_remove,
        "Superseded Engine Files": old_engine_files_to_remove
    }
    
    total_size_removed = 0
    files_removed = 0
    
    for category, file_list in categories.items():
        if file_list:
            print(f"\n{category}:")
            for filename in file_list:
                filepath = script_dir / filename
                if filepath.exists():
                    size_kb = filepath.stat().st_size / 1024
                    total_size_removed += size_kb
                    print(f"  ❌ {filename} ({size_kb:.1f} KB)")
                else:
                    print(f"  ⚠️  {filename} (not found)")
    
    print(f"\nEstimated space to be freed: {total_size_removed:.1f} KB")
    print()
    
    # Confirm before deletion
    response = input("🗑️  Proceed with cleanup? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return False
    
    print("\n🧹 Starting cleanup...")
    print("-" * 30)
    
    # Remove files
    for filename in all_files_to_remove:
        filepath = script_dir / filename
        if filepath.exists():
            try:
                os.remove(filepath)
                files_removed += 1
                print(f"✅ Removed: {filename}")
            except Exception as e:
                print(f"❌ Error removing {filename}: {e}")
        else:
            print(f"⚠️  Not found: {filename}")
    
    print()
    print("=" * 60)
    print("🎉 CLEANUP COMPLETED!")
    print("=" * 60)
    print(f"📊 Files removed: {files_removed}")
    print(f"💾 Space freed: ~{total_size_removed:.1f} KB")
    print()
    print("✅ Preserved essential files:")
    print("   • syntaxAll.json (consolidated syntax data)")
    print("   • legalityAll.json (consolidated legality rules)")
    print("   • EnhancedSolfegeTheoryEngine.wl (main engine)")
    print("   • enhanced_solfege_theory_engine.py (Python interface)")
    print("   • Core support files and documentation")
    print()
    print("🚀 Theory engine is ready with clean, consolidated architecture!")
    
    return True

def list_remaining_files():
    """Show what files remain after cleanup"""
    script_dir = Path(__file__).parent
    remaining_files = [f for f in os.listdir(script_dir) 
                      if f.endswith(('.py', '.wl', '.json', '.md', '.nb')) 
                      and not f.startswith('.')]
    
    print("\n📁 Remaining files in TheoryEngine directory:")
    print("-" * 50)
    
    # Group by type
    file_types = {
        'Core Engine': [],
        'Data Files': [],
        'Demo/Test Scripts': [],
        'Documentation': [],
        'Notebooks': []
    }
    
    for filename in sorted(remaining_files):
        if filename in ['EnhancedSolfegeTheoryEngine.wl', 'enhanced_solfege_theory_engine.py']:
            file_types['Core Engine'].append(filename)
        elif filename.endswith('.json'):
            file_types['Data Files'].append(filename)
        elif 'DEMO' in filename or 'test' in filename.lower():
            file_types['Demo/Test Scripts'].append(filename)
        elif filename.endswith('.md'):
            file_types['Documentation'].append(filename)
        elif filename.endswith('.nb'):
            file_types['Notebooks'].append(filename)
        else:
            file_types['Core Engine'].append(filename)
    
    for category, files in file_types.items():
        if files:
            print(f"\n{category}:")
            for filename in files:
                filepath = script_dir / filename
                if filepath.exists():
                    size_kb = filepath.stat().st_size / 1024
                    print(f"  📄 {filename} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    print("Enhanced Solfege Theory Engine - Directory Cleanup")
    print("This script will remove redundant individual files and keep consolidated data")
    print()
    
    success = cleanup_directory()
    
    if success:
        list_remaining_files()
        print("\n🎵 The theory engine directory is now clean and optimized!")
    else:
        print("\n🔄 No changes made to the directory.")
