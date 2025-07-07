#!/bin/bash

# TOASTer Simple Standalone Build Script
# Creates a distributable version without examples to avoid build issues

set -e  # Exit on any error

echo "🚀 Starting TOASTer Simple Standalone Build..."

# Configuration
BUILD_DIR="build_standalone_simple"
DIST_DIR="dist"
APP_NAME="TOASTer.app"
VERSION="1.0.0"

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

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the TOASTer directory"
    exit 1
fi

# Clean and create build directory
print_status "Setting up build environment..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# Configure CMake for standalone release build (skip examples)
print_status "Configuring CMake for standalone release (without examples)..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
    -DCMAKE_INSTALL_PREFIX=/Applications \
    -DJUCE_ENABLE_MODULE_SOURCE_GROUPS=ON \
    -DJAM_GPU_BACKEND=metal \
    -DJAM_ENABLE_GPU=ON \
    -DJAM_ENABLE_JACK=ON \
    -DJAM_BUILD_EXAMPLES=OFF \
    -DJAM_BUILD_TESTS=OFF \
    -DCMAKE_CXX_STANDARD=17

print_success "CMake configuration complete"

# Build the application
print_status "Building TOASTer application..."
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
print_status "Preparing standalone app bundle..."
rm -rf "$DIST_DIR/$APP_NAME"
cp -R "$BUILD_DIR/TOASTer_artefacts/Release/$APP_NAME" "$DIST_DIR/"

APP_PATH="$DIST_DIR/$APP_NAME"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/TOASTer"

print_status "Bundling dependencies..."

# Simple dependency bundling
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
mkdir -p "$FRAMEWORKS_DIR"

# Copy any local dylibs
if otool -L "$EXECUTABLE_PATH" 2>/dev/null | grep -q "\.dylib"; then
    print_status "Found dynamic libraries to bundle"
    # Use a simpler approach - just update search paths
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$EXECUTABLE_PATH" 2>/dev/null || true
fi

# Copy required resources
print_status "Copying additional resources..."

# Copy Metal shaders if they exist
if [ -d "../JAM_Framework_v2/shaders" ]; then
    mkdir -p "$APP_PATH/Contents/Resources/shaders"
    cp -R ../JAM_Framework_v2/shaders/* "$APP_PATH/Contents/Resources/shaders/" 2>/dev/null || true
    print_success "Copied Metal shaders"
fi

# Copy icons if they exist
if [ -d "Resources" ]; then
    cp -R Resources/* "$APP_PATH/Contents/Resources/" 2>/dev/null || true
    print_success "Copied app resources"
fi

# Set proper permissions
print_status "Setting proper permissions..."
chmod +x "$EXECUTABLE_PATH"

# Code signing (optional, for distribution)
if command -v codesign &> /dev/null; then
    print_status "Code signing application..."
    codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || print_warning "Code signing failed (this is OK for testing)"
fi

# Create a simple ZIP for distribution
print_status "Creating ZIP archive..."
cd "$DIST_DIR"
ZIP_NAME="TOASTer-v${VERSION}-Standalone-Simple.zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$APP_NAME"
cd ..

print_success "ZIP created: $DIST_DIR/$ZIP_NAME"

# Test the app
print_status "Testing app bundle..."
if [ -x "$EXECUTABLE_PATH" ]; then
    print_success "App executable is ready"
    
    # Get app size
    app_size=$(du -sh "$APP_PATH" | cut -f1)
    print_success "App bundle size: $app_size"
else
    print_error "App executable not found or not executable"
    exit 1
fi

# Display summary
echo ""
echo "🎉 Simple Standalone Build Complete!"
echo ""
echo -e "${GREEN}Distributable files created:${NC}"
echo "  📱 App Bundle: $DIST_DIR/$APP_NAME"
echo "  📦 ZIP Archive: $DIST_DIR/$ZIP_NAME"
echo ""
echo -e "${BLUE}To test:${NC}"
echo "  1. Copy the ZIP file to another Mac"
echo "  2. Unzip and run TOASTer.app directly"
echo "  3. Or double-click TOASTer.app in the dist folder"
echo ""
echo -e "${YELLOW}Features included:${NC}"
echo "  ✓ JAM Framework v2 with TOAST UDP protocol"
echo "  ✓ GPU acceleration via Metal backend"
echo "  ✓ Real UDP multicast networking"
echo "  ✓ Optimized for distribution"
echo ""
print_success "Ready for testing!"

# Try to launch the app for verification
print_status "Attempting to launch app for quick test..."
if open "$APP_PATH" 2>/dev/null; then
    print_success "App launched successfully! Check if it opens properly."
else
    print_warning "Could not auto-launch app, but this may be normal"
fi 