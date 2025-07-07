#!/bin/bash

# MIDIp2p Documentation Organization - Phase 2
# Complete organization of remaining files

echo "🗂️  MIDIp2p Documentation Organization - Phase 2"
echo "================================================="

PROJECT_ROOT="/Users/timothydowler/Projects/MIDIp2p"
DOC_ROOT="$PROJECT_ROOT/Documentation"

cd "$PROJECT_ROOT"

echo "📦 Moving probable chronology files..."

# Early Development (GPU/Architecture)
echo "Moving early development files..."
find . -maxdepth 1 -name "*GPU_NATIVE*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;
find . -maxdepth 1 -name "*ARCHITECTURE*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;
find . -maxdepth 1 -name "*JAM_Framework*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;

# Mid Development (Network/Framework Evolution)
echo "Moving mid development files..."
find . -maxdepth 1 -name "*NETWORK*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;
find . -maxdepth 1 -name "*UDP*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;
find . -maxdepth 1 -name "*PNBTR*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;
find . -maxdepth 1 -name "*WIFI*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;
find . -maxdepth 1 -name "*THUNDERBOLT*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;

# Recent Development (TOASTer/Build)
echo "Moving recent development files..."
find . -maxdepth 1 -name "*TOAST*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;
find . -maxdepth 1 -name "*BUILD*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;
find . -maxdepth 1 -name "*TRANSPORT*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;

echo "📁 Moving unknown chronology files..."

# General Documentation
find . -maxdepth 1 -name "README*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" \;
find . -maxdepth 1 -name "Roadmap*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" \;
find . -maxdepth 1 -name "*GUIDELINES*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" \;
find . -maxdepth 1 -name "*DOCUMENTATION*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" \;

# Technical Specs
find . -maxdepth 1 -name "*TECHNICAL*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Technical_Specs/" \;
find . -maxdepth 1 -name "*PROTOCOL*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Technical_Specs/" \;
find . -maxdepth 1 -name "*SPEC*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Technical_Specs/" \;
find . -maxdepth 1 -name "protocolflow.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Technical_Specs/" \;

# Legal/Patent
find . -maxdepth 1 -name "*Patent*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Legal_Patent/" \;
find . -maxdepth 1 -name "*JAMNET*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/Legal_Patent/" \;

# Move remaining miscellaneous files to appropriate categories
echo "Categorizing remaining files..."

# Backup/Status files (probable chronology - recent)
find . -maxdepth 1 -name "*BACKUP*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;
find . -maxdepth 1 -name "*STATUS*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;
find . -maxdepth 1 -name "*COMPLETE*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" \;

# Framework specific docs
find . -maxdepth 1 -name "*JAM*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;
find . -maxdepth 1 -name "*JDAT*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;
find . -maxdepth 1 -name "*JVID*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Early_Development/" \;

# Technical implementation docs
find . -maxdepth 1 -name "*IMPLEMENTATION*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;
find . -maxdepth 1 -name "*INTEGRATION*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" \;

# Any remaining .md files go to unknown chronology
find . -maxdepth 1 -name "*.md" -type f -exec mv {} "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" \;

echo "✅ Phase 2 organization complete!"

# Generate final counts
echo ""
echo "📊 Final Organization Summary:"
echo "=============================="
echo "Known Chronology:"
echo "  Phase 1: $(find "$DOC_ROOT/01_Known_Chronology/Phase_1/" -name "*.md" | wc -l) files"
echo "  Phase 2: $(find "$DOC_ROOT/01_Known_Chronology/Phase_2/" -name "*.md" | wc -l) files"
echo "  Phase 3: $(find "$DOC_ROOT/01_Known_Chronology/Phase_3/" -name "*.md" | wc -l) files"
echo "  Phase 4: $(find "$DOC_ROOT/01_Known_Chronology/Phase_4/" -name "*.md" | wc -l) files"
echo "  Dated: $(find "$DOC_ROOT/01_Known_Chronology/Dated_Documents/" -name "*.md" | wc -l) files"
echo ""
echo "Probable Chronology:"
echo "  Early Dev: $(find "$DOC_ROOT/02_Probable_Chronology/Early_Development/" -name "*.md" | wc -l) files"
echo "  Mid Dev: $(find "$DOC_ROOT/02_Probable_Chronology/Mid_Development/" -name "*.md" | wc -l) files"
echo "  Recent Dev: $(find "$DOC_ROOT/02_Probable_Chronology/Recent_Development/" -name "*.md" | wc -l) files"
echo ""
echo "Unknown Chronology:"
echo "  General: $(find "$DOC_ROOT/03_Unknown_Chronology/General_Documentation/" -name "*.md" | wc -l) files"
echo "  Technical: $(find "$DOC_ROOT/03_Unknown_Chronology/Technical_Specs/" -name "*.md" | wc -l) files"
echo "  Legal: $(find "$DOC_ROOT/03_Unknown_Chronology/Legal_Patent/" -name "*.md" | wc -l) files"

# Total count
TOTAL_ORGANIZED=$(find "$DOC_ROOT" -name "*.md" | wc -l)
echo ""
echo "📁 Total files organized: $TOTAL_ORGANIZED"
echo "🗂️  All markdown files have been categorized and moved!"
echo ""
echo "📋 Next: Review Documentation/MASTER_INDEX.md for navigation"
