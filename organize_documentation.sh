#!/bin/bash

# MIDIp2p Documentation Organization Script
# July 6, 2025

echo "🗂️  MIDIp2p Documentation Organization Tool"
echo "============================================="

PROJECT_ROOT="/Users/timothydowler/Projects/MIDIp2p"
DOC_ROOT="$PROJECT_ROOT/Documentation"

cd "$PROJECT_ROOT"

# Phase 1: Inventory all markdown files
echo "📋 Phase 1: Creating file inventory..."

echo "# MIDIp2p Markdown File Inventory" > "$DOC_ROOT/FILE_INVENTORY.md"
echo "**Generated:** $(date)" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "**Location:** $PROJECT_ROOT" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

# Count total files
TOTAL_FILES=$(find . -name "*.md" -type f | wc -l)
echo "**Total Files Found:** $TOTAL_FILES" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

# Categorize by filename patterns
echo "## 📁 CATEGORY 1: KNOWN CHRONOLOGY" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Phase Documents:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*PHASE*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Dated Documents:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*JULY*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*2025*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Sequential Documents:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*pre*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "## 📁 CATEGORY 2: PROBABLE CHRONOLOGY" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Technology Evolution:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*GPU*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*TOAST*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*BUILD*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Network Development:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*NETWORK*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*WIFI*" -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "## 📁 CATEGORY 3: UNKNOWN CHRONOLOGY" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### General Documentation:" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "README.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "Roadmap*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "### Complete File List (Alphabetical):" >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "\`\`\`" >> "$DOC_ROOT/FILE_INVENTORY.md"
find . -name "*.md" -type f | sort >> "$DOC_ROOT/FILE_INVENTORY.md"
echo "\`\`\`" >> "$DOC_ROOT/FILE_INVENTORY.md"

echo "✅ Inventory complete! See Documentation/FILE_INVENTORY.md"

# Phase 2: Begin file migration (Known Chronology first)
echo ""
echo "📦 Phase 2: Beginning file migration..."

# Move Phase documents
echo "Moving Phase documents..."
find . -maxdepth 1 -name "*PHASE_1*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Phase_1/" \;
find . -maxdepth 1 -name "*PHASE_2*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Phase_2/" \;
find . -maxdepth 1 -name "*PHASE_3*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Phase_3/" \;
find . -maxdepth 1 -name "*PHASE_4*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Phase_4/" \;

# Move dated documents
echo "Moving dated documents..."
find . -maxdepth 1 -name "*JULY*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Dated_Documents/" \;
find . -maxdepth 1 -name "*2025*" -name "*.md" -type f -exec mv {} "$DOC_ROOT/01_Known_Chronology/Dated_Documents/" \;

echo "✅ Phase 2 complete! Known chronology files moved."

# Generate final summary
echo ""
echo "📊 Organization Summary:"
echo "========================"
echo "Known Chronology:"
echo "  Phase 1: $(ls -1 "$DOC_ROOT/01_Known_Chronology/Phase_1/" | wc -l) files"
echo "  Phase 2: $(ls -1 "$DOC_ROOT/01_Known_Chronology/Phase_2/" | wc -l) files"
echo "  Phase 3: $(ls -1 "$DOC_ROOT/01_Known_Chronology/Phase_3/" | wc -l) files"
echo "  Phase 4: $(ls -1 "$DOC_ROOT/01_Known_Chronology/Phase_4/" | wc -l) files"
echo "  Dated: $(ls -1 "$DOC_ROOT/01_Known_Chronology/Dated_Documents/" | wc -l) files"
echo ""
echo "📁 Documentation structure created at: $DOC_ROOT"
echo "📋 See MASTER_INDEX.md for navigation"
echo "🗂️  Organization Phase 1 & 2 complete!"

echo ""
echo "🎯 Next Steps:"
echo "1. Review migrated files in Documentation folders"
echo "2. Run organization script for probable chronology files"
echo "3. Manually categorize remaining unknown chronology files"
echo "4. Update MASTER_INDEX.md with final counts and links"
