#!/bin/bash

# Quick macOS Build Script - Uses macOS-specific CMakeLists.txt without JACK

set -e

echo "🍎 Quick macOS Build (CoreAudio, No JACK)..."

# Configuration  
BUILD_DIR="build_quick_macos"
DIST_DIR="dist"
APP_NAME="TOASTer.app"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Clean build
print_status "Setting up quick macOS build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# Use macOS-specific CMakeLists.txt
cd "$BUILD_DIR"
print_status "Using macOS-specific configuration (no JACK)..."

cp ../CMakeLists_macOS.txt ../CMakeLists.txt.backup
cp ../CMakeLists_macOS.txt ../CMakeLists.txt

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DJAM_BUILD_EXAMPLES=OFF \
    -DJAM_BUILD_TESTS=OFF

print_success "Configuration complete"

# Build just the main target
print_status "Building TOASTer..."
make -j$(sysctl -n hw.ncpu) TOASTer

cd ..

# Restore original CMakeLists.txt
if [ -f "CMakeLists.txt.backup" ]; then
    mv CMakeLists.txt.backup CMakeLists.txt
fi

# Quick package
if [ -d "$BUILD_DIR/TOASTer_artefacts/Release/$APP_NAME" ]; then
    print_status "Packaging..."
    rm -rf "$DIST_DIR/$APP_NAME"
    cp -R "$BUILD_DIR/TOASTer_artefacts/Release/$APP_NAME" "$DIST_DIR/"
    
    # Quick dependency check
    EXECUTABLE_PATH="$DIST_DIR/$APP_NAME/Contents/MacOS/TOASTer"
    if otool -L "$EXECUTABLE_PATH" | grep -q "jack"; then
        echo "❌ JACK dependency still found"
        exit 1
    else
        print_success "✅ No JACK dependencies - using CoreAudio"
    fi
    
    # Quick ZIP
    cd "$DIST_DIR"
    ZIP_NAME="TOASTer-v1.0.0-macOS-NoJACK.zip"
    rm -f "$ZIP_NAME"
    zip -r "$ZIP_NAME" "$APP_NAME"
    cd ..
    
    print_success "macOS build complete: $DIST_DIR/$ZIP_NAME"
    
    # Test launch
    if open "$DIST_DIR/$APP_NAME"; then
        print_success "🚀 TOASTer launched with CoreAudio!"
    fi
else
    echo "❌ Build failed"
    exit 1
fi 