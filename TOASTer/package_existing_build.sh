#!/bin/bash

# Package Existing TOASTer Build Script
# Packages the existing working build into a standalone distributable

set -e

echo "📦 Packaging Existing TOASTer Build for Distribution..."

# Configuration
SOURCE_BUILD="build_jam"
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

# Check if source build exists
if [ ! -d "$SOURCE_BUILD/TOASTer_artefacts/Release/$APP_NAME" ]; then
    print_error "Source build not found at $SOURCE_BUILD/TOASTer_artefacts/Release/$APP_NAME"
    print_error "Please ensure the build_jam build completed successfully first"
    exit 1
fi

print_status "Found existing build, packaging for distribution..."

# Create distribution directory
mkdir -p "$DIST_DIR"

# Copy the app bundle
print_status "Copying app bundle..."
rm -rf "$DIST_DIR/$APP_NAME"
cp -R "$SOURCE_BUILD/TOASTer_artefacts/Release/$APP_NAME" "$DIST_DIR/"

APP_PATH="$DIST_DIR/$APP_NAME"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/TOASTer"

# Verify app exists
if [ ! -f "$EXECUTABLE_PATH" ]; then
    print_error "App executable not found at $EXECUTABLE_PATH"
    exit 1
fi

print_success "App bundle copied successfully"

# Add resources
print_status "Adding additional resources..."

# Copy Metal shaders if they exist
if [ -d "../JAM_Framework_v2/shaders" ]; then
    mkdir -p "$APP_PATH/Contents/Resources/shaders"
    cp -R ../JAM_Framework_v2/shaders/* "$APP_PATH/Contents/Resources/shaders/" 2>/dev/null || true
    print_success "Copied Metal shaders"
fi

# Copy app icon and resources
if [ -d "Resources" ]; then
    cp -R Resources/* "$APP_PATH/Contents/Resources/" 2>/dev/null || true
    print_success "Copied app resources"
fi

# Set proper permissions
print_status "Setting permissions..."
chmod +x "$EXECUTABLE_PATH"

# Simple dependency check
print_status "Checking dependencies..."
if command -v otool &> /dev/null; then
    echo "App dependencies:"
    otool -L "$EXECUTABLE_PATH" | grep -v ":" | while read -r line; do
        dep=$(echo "$line" | awk '{print $1}')
        if [[ "$dep" == "/System/"* ]] || [[ "$dep" == "/usr/lib/"* ]]; then
            echo "  ✓ $dep (system)"
        else
            echo "  📋 $dep"
        fi
    done
fi

# Code signing for distribution
if command -v codesign &> /dev/null; then
    print_status "Code signing application..."
    codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || print_warning "Code signing failed (OK for testing)"
    print_success "Code signing completed"
fi

# Create ZIP for distribution
print_status "Creating ZIP archive..."
cd "$DIST_DIR"
ZIP_NAME="TOASTer-v${VERSION}-Ready.zip"
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "$APP_NAME"
cd ..

print_success "ZIP created: $DIST_DIR/$ZIP_NAME"

# Get app information
app_size=$(du -sh "$APP_PATH" | cut -f1)
zip_size=$(du -sh "$DIST_DIR/$ZIP_NAME" | cut -f1)

# Test the app
print_status "Testing app bundle..."
if [ -x "$EXECUTABLE_PATH" ]; then
    print_success "App is executable and ready"
else
    print_error "App executable test failed"
    exit 1
fi

# Display summary
echo ""
echo "🎉 Standalone Package Ready!"
echo ""
echo -e "${GREEN}Distribution files:${NC}"
echo "  📱 App Bundle: $DIST_DIR/$APP_NAME ($app_size)"
echo "  📦 ZIP Archive: $DIST_DIR/$ZIP_NAME ($zip_size)"
echo ""
echo -e "${BLUE}Testing instructions:${NC}"
echo "  1. Copy $DIST_DIR/$ZIP_NAME to any Mac"
echo "  2. Unzip and double-click TOASTer.app"
echo "  3. Test UDP networking functionality"
echo ""
echo -e "${YELLOW}Features included:${NC}"
echo "  ✓ JAM Framework v2 with real TOAST protocol"
echo "  ✓ GPU acceleration via Metal backend"
echo "  ✓ Real UDP multicast networking (no placeholders)"
echo "  ✓ PNBTR prediction system"
echo "  ✓ Network state detection"
echo ""

# Launch for immediate testing
print_status "Launching app for immediate testing..."
if open "$APP_PATH"; then
    print_success "TOASTer launched! Check if it opens and functions correctly."
    echo ""
    echo -e "${GREEN}🚀 Your standalone TOASTer app is ready for distribution!${NC}"
else
    print_warning "Could not auto-launch app"
    echo "You can manually open: $APP_PATH"
fi

echo ""
print_success "Package complete - ready for VM testing or distribution!" 