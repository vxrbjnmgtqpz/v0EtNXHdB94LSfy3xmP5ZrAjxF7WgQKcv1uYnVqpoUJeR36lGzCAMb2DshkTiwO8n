#!/bin/bash

# JAMNet Terminology Migration Script
# Migrates JDAT→JDAT, JMID→JMID, JVID→JVID throughout the project

set -e  # Exit on any error
export LC_CTYPE=C  # Handle encoding issues with sed

echo "🔄 JAMNet Terminology Migration: JDAT→JDAT, JMID→JMID, JVID→JVID"
echo "=============================================================================="

# Create backup before proceeding
BACKUP_TAG="v0.8.1-pre-terminology-update"
echo "📦 Creating git backup tag: $BACKUP_TAG"
git tag $BACKUP_TAG || echo "Tag already exists, continuing..."
echo "✅ Backup created. To rollback: git reset --hard $BACKUP_TAG"

echo ""
echo "🔍 Phase 1: Text Content Updates (Documentation, Comments, Strings)"
echo "=================================================================="

# Function to perform safe text replacements
safe_replace() {
    local pattern="$1"
    local replacement="$2"
    local description="$3"
    
    echo "  🔄 $description"
    
    # Find and replace in text files, excluding binary files, build directories, and git
    find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cc" -o -name "*.cxx" -o -name "*.json" -o -name "*.cmake" -o -name "*.txt" -o -name "CMakeLists.txt" -o -name "*.sh" -o -name "*.py" \) \
        -not -path "./.git/*" \
        -not -path "./*/build*/*" \
        -not -path "./*/.cache/*" \
        -not -path "./cmake-build-*/*" \
        -exec grep -l "$pattern" {} \; 2>/dev/null | while read file; do
        if [ -f "$file" ]; then
            sed -i.bak "s/$pattern/$replacement/g" "$file" 2>/dev/null || echo "    ⚠️  Skipping $file (encoding issue)"
        fi
    done
    
    # Clean up backup files
    find . -name "*.bak" -delete 2>/dev/null || true
}

# Perform text replacements in order (most specific first)
safe_replace "JDAT" "JDAT" "JDAT → JDAT in all text"
safe_replace "JMID" "JMID" "JMID → JMID in all text"  
safe_replace "JVID" "JVID" "JVID → JVID in all text"
safe_replace "jdat" "jdat" "jdat → jdat in all text"
safe_replace "jmid" "jmid" "jmid → jmid in all text"
safe_replace "jvid" "jvid" "jvid → jvid in all text"

# Handle ADAT references contextually
echo "  🔄 Contextual ADAT replacements..."
safe_replace "4-channel interleaving protocol" "4-channel interleaving protocol" "4-channel interleaving protocol → 4-channel interleaving protocol"
safe_replace "4-channel interleaving format" "4-channel interleaving format" "4-channel interleaving format → 4-channel interleaving format"
safe_replace "4-channel interleaving style" "4-channel interleaving style" "4-channel interleaving style → 4-channel interleaving style"
safe_replace "\\bADAT\\b" "4-channel interleaving" "Standalone ADAT → 4-channel interleaving"

echo ""
echo "🏗️ Phase 2: Schema and Configuration Files"
echo "============================================="

# Update schema files
if [ -f "JMID_Framework/schemas/jmid-message.schema.json" ]; then
    echo "  🔄 Updating JMID schema file"
    mv "JMID_Framework/schemas/jmid-message.schema.json" "JMID_Framework/schemas/jmid-message.schema.json"
fi

if [ -f "JDAT_Framework/schemas/jdat-message.schema.json" ]; then
    echo "  🔄 Updating JDAT schema file"
    mv "JDAT_Framework/schemas/jdat-message.schema.json" "JDAT_Framework/schemas/jdat-message.schema.json"
fi

echo ""
echo "📁 Phase 3: File Renames (Headers, Sources, Examples)"
echo "====================================================="

# Function to rename files with progress
rename_files() {
    local pattern="$1"
    local replacement="$2"
    local description="$3"
    
    echo "  🔄 $description"
    
    find . -type f -name "*$pattern*" \
        -not -path "./.git/*" \
        -not -path "./*/build*/*" \
        -not -path "./*/.cache/*" \
        -not -path "./cmake-build-*/*" | while read file; do
        
        newfile=$(echo "$file" | sed "s/$pattern/$replacement/g")
        if [ "$file" != "$newfile" ]; then
            echo "    📄 $file → $newfile"
            mv "$file" "$newfile"
        fi
    done
}

# Rename files (most specific first)
rename_files "JMID" "JMID" "JMID → JMID file names"
rename_files "JMIDMessage" "JMIDMessage" "JMIDMessage → JMIDMessage files"
rename_files "JMIDParser" "JMIDParser" "JMIDParser → JMIDParser files"
rename_files "JDAT" "JDAT" "JDAT → JDAT file names"
rename_files "JDATMessage" "JDATMessage" "JDATMessage → JDATMessage files"
rename_files "JVID" "JVID" "JVID → JVID file names"

echo ""
echo "📂 Phase 4: Directory Renames"
echo "=============================="

# Rename directories
if [ -d "JMID_Framework" ]; then
    echo "  📂 JMID_Framework → JMID_Framework"
    mv "JMID_Framework" "JMID_Framework"
fi

if [ -d "JDAT_Framework" ]; then
    echo "  📂 JDAT_Framework → JDAT_Framework"
    mv "JDAT_Framework" "JDAT_Framework"
fi

if [ -d "JVID_Framework" ]; then
    echo "  📂 JVID_Framework → JVID_Framework"
    mv "JVID_Framework" "JVID_Framework"
fi

echo ""
echo "🔧 Phase 5: CMake and Build System Updates"
echo "==========================================="

# Update CMake files to reflect new directory and target names
safe_replace "JMID_Framework" "JMID_Framework" "Directory references in CMake"
safe_replace "JDAT_Framework" "JDAT_Framework" "Directory references in CMake"
safe_replace "JVID_Framework" "JVID_Framework" "Directory references in CMake"
safe_replace "jmid_framework" "jmid_framework" "CMake target names"
safe_replace "jdat_framework" "jdat_framework" "CMake target names"
safe_replace "jvid_framework" "jvid_framework" "CMake target names"

echo ""
echo "🧹 Phase 6: Update Git Tracking"
echo "================================"

echo "  🔄 Adding renamed files to git"
git add -A

echo ""
echo "✅ Phase 7: Validation"
echo "======================"

echo "  🔍 Checking for any remaining old terminology..."

# Check for remaining instances
remaining_jdat=$(grep -r "JDAT" . --exclude-dir=.git --exclude-dir=build\* --exclude-dir=.cache 2>/dev/null || true)
remaining_jmid=$(grep -r "JMID" . --exclude-dir=.git --exclude-dir=build\* --exclude-dir=.cache 2>/dev/null || true)
remaining_jvid=$(grep -r "JVID" . --exclude-dir=.git --exclude-dir=build\* --exclude-dir=.cache 2>/dev/null || true)

if [ -n "$remaining_jdat" ] || [ -n "$remaining_jmid" ] || [ -n "$remaining_jvid" ]; then
    echo "  ⚠️  Some old terminology remains:"
    [ -n "$remaining_jdat" ] && echo "    JDAT instances found"
    [ -n "$remaining_jmid" ] && echo "    JMID instances found"
    [ -n "$remaining_jvid" ] && echo "    JVID instances found"
    echo "  💡 Run 'grep -r \"JDAT\\|JMID\\|JVID\" . --exclude-dir=.git' to see details"
else
    echo "  ✅ No old terminology detected!"
fi

echo ""
echo "📋 Phase 8: Build Test"
echo "======================"

echo "  🔨 Testing JMID Framework build..."
if [ -d "JMID_Framework" ]; then
    cd JMID_Framework
    mkdir -p build_test
    cd build_test
    if cmake .. && make -j$(nproc 2>/dev/null || echo 4); then
        echo "  ✅ JMID Framework builds successfully"
    else
        echo "  ❌ JMID Framework build failed"
    fi
    cd ../..
fi

echo "  🔨 Testing TOASTer build..."
if [ -d "TOASTer" ]; then
    cd TOASTer
    mkdir -p build_test
    cd build_test
    if cmake .. && make -j$(nproc 2>/dev/null || echo 4); then
        echo "  ✅ TOASTer builds successfully"
    else
        echo "  ❌ TOASTer build failed"
    fi
    cd ../..
fi

echo ""
echo "🎉 Migration Complete!"
echo "======================"
echo ""
echo "📝 Summary of changes:"
echo "  • JDAT → JDAT (all instances)"
echo "  • JMID → JMID (all instances)"
echo "  • JVID → JVID (all instances)"
echo "  • ADAT → 4-channel interleaving (contextual)"
echo "  • Directory renames: JSON*_Framework → J*_Framework"
echo "  • File renames: JSON* files → J* files"
echo "  • CMake targets updated"
echo "  • Schema files renamed"
echo ""
echo "🚀 Next steps:"
echo "  1. Commit the changes: git commit -m 'Complete terminology migration: JDAT→JDAT, JMID→JMID, JVID→JVID'"
echo "  2. Test all functionality with new naming"
echo "  3. Update any external documentation or dependencies"
echo ""
echo "🔙 To rollback if needed: git reset --hard $BACKUP_TAG"
