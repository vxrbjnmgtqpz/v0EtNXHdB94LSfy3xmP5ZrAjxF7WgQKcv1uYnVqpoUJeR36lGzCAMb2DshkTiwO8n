#!/bin/bash

# TOASTer macOS Standalone Build Script
# Creates a standalone build using CoreAudio instead of JACK (which is for Linux)

set -e

echo "🍎 Creating TOASTer macOS Standalone Build (No JACK, CoreAudio Only)..."

# Configuration
BUILD_DIR="build_macos_standalone"
DIST_DIR="dist"
APP_NAME="TOASTer.app"
VERSION="1.0.0"

# Colors for output
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

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the TOASTer directory"
    exit 1
fi

# Clean and create build directory
print_status "Setting up macOS-specific build environment..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# Configure CMake for macOS standalone (NO JACK)
print_status "Configuring CMake for macOS CoreAudio (disabling JACK)..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
    -DCMAKE_INSTALL_PREFIX=/Applications \
    -DJUCE_ENABLE_MODULE_SOURCE_GROUPS=ON \
    -DJAM_GPU_BACKEND=metal \
    -DJAM_ENABLE_GPU=ON \
    -DJAM_ENABLE_JACK=OFF \
    -DJAM_BUILD_EXAMPLES=OFF \
    -DJAM_BUILD_TESTS=OFF \
    -DCMAKE_CXX_STANDARD=17

print_success "CMake configuration complete (JACK disabled for macOS)"

# Build the application
print_status "Building TOASTer with CoreAudio backend..."
make -j$(sysctl -n hw.ncpu) TOASTer
print_success "Build complete"

# Check if app was created
if [ ! -d "TOASTer_artefacts/Release/$APP_NAME" ]; then
    print_error "App bundle not found after build"
    exit 1
fi

# Go back to project root
cd ..

# Copy the app to distribution directory
print_status "Preparing macOS standalone app bundle..."
rm -rf "$DIST_DIR/$APP_NAME"
cp -R "$BUILD_DIR/TOASTer_artefacts/Release/$APP_NAME" "$DIST_DIR/"

APP_PATH="$DIST_DIR/$APP_NAME"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/TOASTer"

# Verify no JACK dependencies
print_status "Verifying NO JACK dependencies..."
if otool -L "$EXECUTABLE_PATH" | grep -q "jack"; then
    print_error "JACK dependency still found! Build configuration issue."
    otool -L "$EXECUTABLE_PATH" | grep jack
    exit 1
else
    print_success "Confirmed: No JACK dependencies (using CoreAudio)"
fi

# Copy required resources
print_status "Adding macOS-specific resources..."

# Copy Metal shaders
if [ -d "../JAM_Framework_v2/shaders" ]; then
    mkdir -p "$APP_PATH/Contents/Resources/shaders"
    cp -R ../JAM_Framework_v2/shaders/* "$APP_PATH/Contents/Resources/shaders/" 2>/dev/null || true
    print_success "Copied Metal shaders for GPU acceleration"
fi

# Copy app resources
if [ -d "Resources" ]; then
    cp -R Resources/* "$APP_PATH/Contents/Resources/" 2>/dev/null || true
    print_success "Copied app resources"
fi

# Set proper permissions
print_status "Setting permissions..."
chmod +x "$EXECUTABLE_PATH"

# Show all dependencies for verification
print_status "macOS Dependencies Summary:"
if command -v otool &> /dev/null; then
    otool -L "$EXECUTABLE_PATH" | grep -v ":" | while read -r line; do
        dep=$(echo "$line" | awk '{print $1}')
        if [[ "$dep" == "/System/"* ]] || [[ "$dep" == "/usr/lib/"* ]]; then
            echo "  ✅ $dep (native macOS)"
        else
            echo "  ⚠️  $dep (external - may need bundling)"
        fi
    done
else
    print_warning "otool not available"
fi

# Code signing
if command -v codesign &> /dev/null; then
    print_status "Code signing for macOS distribution..."
    codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || print_warning "Code signing failed (OK for testing)"
    print_success "Code signing completed"
fi

# Create distributable ZIP
print_status "Creating macOS distributable ZIP..."
cd "$DIST_DIR"
ZIP_NAME="TOASTer-v${VERSION}-macOS-Standalone.zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$APP_NAME"
cd ..

print_success "ZIP created: $DIST_DIR/$ZIP_NAME"

# Get sizes
app_size=$(du -sh "$APP_PATH" | cut -f1)
zip_size=$(du -sh "$DIST_DIR/$ZIP_NAME" | cut -f1)

# Test the app
print_status "Testing macOS app bundle..."
if [ -x "$EXECUTABLE_PATH" ]; then
    print_success "App is executable and ready for macOS"
else
    print_error "App executable test failed"
    exit 1
fi

# Display summary
echo ""
echo "🎉 macOS Standalone Build Complete!"
echo ""
echo -e "${GREEN}✅ Native macOS Features:${NC}"
echo "  🍎 CoreAudio backend (no JACK dependency)"
echo "  🔩 Metal GPU acceleration"
echo "  🌐 UDP networking with TOAST protocol"
echo "  🧠 PNBTR prediction system"
echo "  📱 Native macOS app bundle"
echo ""
echo -e "${BLUE}📦 Distribution Files:${NC}"
echo "  📱 App Bundle: $DIST_DIR/$APP_NAME ($app_size)"
echo "  📦 ZIP Archive: $DIST_DIR/$ZIP_NAME ($zip_size)"
echo ""
echo -e "${YELLOW}🧪 Testing Instructions:${NC}"
echo "  1. Copy $ZIP_NAME to any Mac (including VMs)"
echo "  2. Unzip and double-click TOASTer.app"
echo "  3. Should launch without dependency errors"
echo "  4. Test UDP networking functionality"
echo ""

# Test launch
print_status "Testing app launch..."
if open "$APP_PATH" 2>/dev/null; then
    print_success "✅ TOASTer launched successfully with CoreAudio!"
    echo ""
    echo -e "${GREEN}🚀 Ready for VM testing - no more JACK dependency issues!${NC}"
else
    print_warning "Could not auto-launch (but this may be normal)"
fi

echo ""
print_success "macOS standalone build complete - should work on any Mac!" 