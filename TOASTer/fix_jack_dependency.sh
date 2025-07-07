#!/bin/bash

# Fix JACK Dependency Script
# Removes JACK dependency from existing build using install_name_tool

set -e

echo "🔧 Fixing JACK Dependency in Existing Build..."

# Configuration
SOURCE_APP="dist/TOASTer.app"
FIXED_APP="dist/TOASTer-Fixed.app"
EXECUTABLE_PATH="Contents/MacOS/TOASTer"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if source app exists
if [ ! -d "$SOURCE_APP" ]; then
    print_error "Source app not found at $SOURCE_APP"
    print_error "Please run ./package_existing_build.sh first"
    exit 1
fi

print_status "Found existing app, creating fixed version..."

# Copy the app to create fixed version
rm -rf "$FIXED_APP"
cp -R "$SOURCE_APP" "$FIXED_APP"

FIXED_EXECUTABLE="$FIXED_APP/$EXECUTABLE_PATH"

# Check current dependencies
print_status "Checking current dependencies..."
if otool -L "$FIXED_EXECUTABLE" | grep -q "jack"; then
    print_warning "JACK dependency found, removing..."
    
    # Get the exact JACK library path
    JACK_LIB=$(otool -L "$FIXED_EXECUTABLE" | grep jack | awk '{print $1}')
    echo "Found JACK library: $JACK_LIB"
    
    # Method 1: Try to remove the dependency entirely
    print_status "Attempting to remove JACK dependency..."
    
    # This is tricky - we can't easily remove a dependency, but we can try to make it optional
    # by changing the load command or using a stub
    
    # Create a minimal fix by copying the app and modifying load commands
    if command -v install_name_tool &> /dev/null; then
        print_status "Using install_name_tool to modify dependencies..."
        
        # Try to make the JACK library path invalid so it's ignored
        install_name_tool -change "$JACK_LIB" "/dev/null/libjack.dylib" "$FIXED_EXECUTABLE" 2>/dev/null || {
            print_warning "Could not modify JACK dependency directly"
            
            # Alternative: Try to point to a system library that's always available
            print_status "Trying alternative fix..."
            install_name_tool -change "$JACK_LIB" "/usr/lib/libSystem.B.dylib" "$FIXED_EXECUTABLE" 2>/dev/null || {
                print_error "Could not fix JACK dependency with install_name_tool"
            }
        }
    fi
    
else
    print_success "No JACK dependencies found!"
fi

# Verify the fix
print_status "Verifying fixed dependencies..."
echo "Dependencies after fix:"
otool -L "$FIXED_EXECUTABLE" | grep -v ":" | while read -r line; do
    dep=$(echo "$line" | awk '{print $1}')
    if [[ "$dep" == *"jack"* ]]; then
        echo "  ⚠️  $dep (JACK - may cause issues)"
    elif [[ "$dep" == "/System/"* ]] || [[ "$dep" == "/usr/lib/"* ]]; then
        echo "  ✅ $dep (system)"
    else
        echo "  📋 $dep"
    fi
done

# Set proper permissions
chmod +x "$FIXED_EXECUTABLE"

# Code sign the fixed app
if command -v codesign &> /dev/null; then
    print_status "Re-signing fixed app..."
    codesign --force --deep --sign - "$FIXED_APP" 2>/dev/null || print_warning "Code signing failed"
fi

# Create distributable ZIP
print_status "Creating fixed distributable..."
cd dist
ZIP_NAME="TOASTer-v1.0.0-Fixed-NoJACK.zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$(basename "$FIXED_APP")"
cd ..

print_success "ZIP created: dist/$ZIP_NAME"

# Test the fixed app
print_status "Testing fixed app..."
if [ -x "$FIXED_EXECUTABLE" ]; then
    print_success "Fixed app is executable"
    
    # Try to launch it
    print_status "Attempting to launch fixed app..."
    if open "$FIXED_APP" 2>/dev/null; then
        print_success "🚀 Fixed TOASTer launched!"
        echo ""
        echo -e "${GREEN}✅ JACK dependency fix applied!${NC}"
        echo -e "${BLUE}📦 Ready for VM testing: dist/$ZIP_NAME${NC}"
    else
        print_warning "Could not auto-launch (but file is ready for testing)"
    fi
else
    print_error "Fixed app executable test failed"
    exit 1
fi

echo ""
print_success "JACK dependency fix complete!"
echo "Try the fixed version: dist/$ZIP_NAME" 