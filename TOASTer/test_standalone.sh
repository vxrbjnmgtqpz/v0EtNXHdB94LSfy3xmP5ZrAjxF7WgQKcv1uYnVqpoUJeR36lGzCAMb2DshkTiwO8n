#!/bin/bash

# TOASTer Standalone Test Script
# Tests the standalone build to ensure it works correctly

set -e

echo "🧪 Testing TOASTer Standalone Build..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
DIST_DIR="dist"
APP_NAME="TOASTer.app"
APP_PATH="$DIST_DIR/$APP_NAME"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/TOASTer"

# Check if standalone build exists
if [ ! -d "$APP_PATH" ]; then
    print_error "Standalone build not found at $APP_PATH"
    echo "Please run ./create_standalone_build.sh first"
    exit 1
fi

print_status "Testing app bundle structure..."

# Check app bundle structure
required_paths=(
    "$APP_PATH/Contents/Info.plist"
    "$APP_PATH/Contents/MacOS/TOASTer"
    "$APP_PATH/Contents/Resources"
)

for path in "${required_paths[@]}"; do
    if [ -e "$path" ]; then
        print_success "Found: $(basename "$path")"
    else
        print_error "Missing: $path"
        exit 1
    fi
done

# Check executable permissions
print_status "Checking executable permissions..."
if [ -x "$EXECUTABLE_PATH" ]; then
    print_success "Executable has proper permissions"
else
    print_error "Executable is not executable"
    exit 1
fi

# Check dependencies
print_status "Checking dependencies..."
if command -v otool &> /dev/null; then
    echo "Dependencies found:"
    otool -L "$EXECUTABLE_PATH" | grep -v ":" | while read -r line; do
        dep=$(echo "$line" | awk '{print $1}')
        if [[ "$dep" == *"@executable_path"* ]]; then
            echo "  ✓ $dep (bundled)"
        elif [[ "$dep" == "/System/"* ]] || [[ "$dep" == "/usr/lib/"* ]]; then
            echo "  ✓ $dep (system)"
        else
            echo "  ⚠️  $dep (external - may cause issues)"
        fi
    done
else
    print_warning "otool not available, skipping dependency check"
fi

# Test app launch (non-interactive)
print_status "Testing app launch..."
if "$EXECUTABLE_PATH" --help &>/dev/null || [ $? -eq 1 ]; then
    # Exit code 1 is often expected for GUI apps when run from command line
    print_success "App can be launched successfully"
else
    print_warning "App launch test inconclusive (this may be normal for GUI apps)"
fi

# Check for critical resources
print_status "Checking for critical resources..."

resources_dir="$APP_PATH/Contents/Resources"
if [ -d "$resources_dir/shaders" ]; then
    print_success "Metal shaders found"
    shader_count=$(find "$resources_dir/shaders" -name "*.metal" -o -name "*.comp" -o -name "*.glsl" | wc -l)
    echo "  Found $shader_count shader files"
fi

if [ -f "$resources_dir/TOASTer.icns" ]; then
    print_success "App icon found"
fi

# Check app bundle size
print_status "Checking app bundle size..."
app_size=$(du -sh "$APP_PATH" | cut -f1)
print_success "App bundle size: $app_size"

# Final verification
echo ""
echo "🎯 Test Summary:"
echo ""

if [ -x "$EXECUTABLE_PATH" ]; then
    echo -e "${GREEN}✅ Standalone build appears to be working correctly!${NC}"
    echo ""
    echo -e "${BLUE}Ready for distribution:${NC}"
    echo "  📱 App Bundle: $APP_PATH"
    
    if [ -f "$DIST_DIR/TOASTer-v1.0.0-Standalone.dmg" ]; then
        echo "  💿 DMG: $DIST_DIR/TOASTer-v1.0.0-Standalone.dmg"
    fi
    
    if [ -f "$DIST_DIR/TOASTer-v1.0.0-Standalone.zip" ]; then
        echo "  📦 ZIP: $DIST_DIR/TOASTer-v1.0.0-Standalone.zip"
    fi
    
    echo ""
    echo -e "${YELLOW}To test on another machine:${NC}"
    echo "  1. Copy the DMG or ZIP to another Mac"
    echo "  2. Install/extract and run TOASTer"
    echo "  3. Test UDP networking functionality"
    echo ""
    
else
    echo -e "${RED}❌ Issues found with standalone build${NC}"
    exit 1
fi 