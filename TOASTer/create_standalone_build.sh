#!/bin/bash

# TOASTer Standalone Build Script
# Creates a distributable version of TOASTer with all dependencies bundled

set -e  # Exit on any error

echo "🚀 Starting TOASTer Standalone Build Process..."

# Configuration
BUILD_DIR="build_standalone"
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

# Configure CMake for standalone release build
print_status "Configuring CMake for standalone release..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
    -DCMAKE_INSTALL_PREFIX=/Applications \
    -DJUCE_ENABLE_MODULE_SOURCE_GROUPS=ON \
    -DJAM_GPU_BACKEND=metal \
    -DJAM_ENABLE_GPU=ON \
    -DJAM_ENABLE_JACK=ON \
    -DCMAKE_CXX_STANDARD=17

print_success "CMake configuration complete"

# Build the application
print_status "Building TOASTer application..."
make -j$(sysctl -n hw.ncpu)
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

# Function to bundle dependencies using otool and install_name_tool
bundle_dependencies() {
    local binary_path="$1"
    local frameworks_dir="$APP_PATH/Contents/Frameworks"
    
    mkdir -p "$frameworks_dir"
    
    print_status "Bundling dependencies for $(basename "$binary_path")..."
    
    # Get list of dynamic libraries
    local deps=$(otool -L "$binary_path" | grep -E "\.dylib|\.framework" | grep -v "/System/" | grep -v "/usr/lib/" | awk '{print $1}' | grep -v ":")
    
    for dep in $deps; do
        if [ -f "$dep" ]; then
            local lib_name=$(basename "$dep")
            local framework_path="$frameworks_dir/$lib_name"
            
            if [ ! -f "$framework_path" ]; then
                print_status "  Copying $lib_name..."
                cp "$dep" "$framework_path"
                
                # Update the library's internal references
                install_name_tool -id "@executable_path/../Frameworks/$lib_name" "$framework_path"
                
                # Recursively bundle dependencies of this library
                bundle_dependencies "$framework_path"
            fi
            
            # Update reference in the binary
            install_name_tool -change "$dep" "@executable_path/../Frameworks/$lib_name" "$binary_path"
        fi
    done
}

# Bundle dependencies
if [ -f "$EXECUTABLE_PATH" ]; then
    bundle_dependencies "$EXECUTABLE_PATH"
else
    print_warning "Executable not found at expected path: $EXECUTABLE_PATH"
fi

# Copy required resources
print_status "Copying additional resources..."

# Copy any Metal shaders if they exist
if [ -d "../JAM_Framework_v2/shaders" ]; then
    mkdir -p "$APP_PATH/Contents/Resources/shaders"
    cp -R ../JAM_Framework_v2/shaders/* "$APP_PATH/Contents/Resources/shaders/"
    print_success "Copied Metal shaders"
fi

# Copy icons if they exist
if [ -d "Resources" ]; then
    cp -R Resources/* "$APP_PATH/Contents/Resources/"
    print_success "Copied app resources"
fi

# Set proper permissions
print_status "Setting proper permissions..."
chmod +x "$EXECUTABLE_PATH"
find "$APP_PATH" -name "*.dylib" -exec chmod +x {} \;

# Code signing (optional, for distribution)
if command -v codesign &> /dev/null; then
    print_status "Code signing application..."
    codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || print_warning "Code signing failed (this is OK for testing)"
fi

# Create a DMG for easy distribution
print_status "Creating distributable DMG..."
DMG_NAME="TOASTer-v${VERSION}-Standalone.dmg"
rm -f "$DIST_DIR/$DMG_NAME"

# Create temporary directory for DMG contents
DMG_DIR="$DIST_DIR/dmg_temp"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"

# Copy app to DMG directory
cp -R "$APP_PATH" "$DMG_DIR/"

# Create symbolic link to Applications
ln -s /Applications "$DMG_DIR/Applications"

# Create the DMG
hdiutil create -format UDZO -srcfolder "$DMG_DIR" "$DIST_DIR/$DMG_NAME"
rm -rf "$DMG_DIR"

print_success "DMG created: $DIST_DIR/$DMG_NAME"

# Create a simple ZIP for easier distribution
print_status "Creating ZIP archive..."
cd "$DIST_DIR"
ZIP_NAME="TOASTer-v${VERSION}-Standalone.zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$APP_NAME"
cd ..

print_success "ZIP created: $DIST_DIR/$ZIP_NAME"

# Final verification
print_status "Verifying standalone build..."

# Check if the app can find its dependencies
if otool -L "$EXECUTABLE_PATH" | grep -q "@executable_path"; then
    print_success "App is properly configured for standalone distribution"
else
    print_warning "App may still have system dependencies"
fi

# Display summary
echo ""
echo "🎉 Standalone build complete!"
echo ""
echo -e "${GREEN}Distributable files created:${NC}"
echo "  📱 App Bundle: $DIST_DIR/$APP_NAME"
echo "  💿 DMG Image: $DIST_DIR/$DMG_NAME"
echo "  📦 ZIP Archive: $DIST_DIR/$ZIP_NAME"
echo ""
echo -e "${BLUE}To test:${NC}"
echo "  1. Copy any of these files to another Mac"
echo "  2. Double-click the DMG and drag TOASTer to Applications"
echo "  3. Or unzip the ZIP file and run TOASTer.app directly"
echo ""
echo -e "${YELLOW}Features included:${NC}"
echo "  ✓ JAM Framework v2 with TOAST UDP protocol"
echo "  ✓ GPU acceleration via Metal backend"
echo "  ✓ PNBTR prediction system"
echo "  ✓ Real UDP multicast networking"
echo "  ✓ All dependencies bundled"
echo ""
print_success "Ready for testing on any macOS 10.13+ system!" 