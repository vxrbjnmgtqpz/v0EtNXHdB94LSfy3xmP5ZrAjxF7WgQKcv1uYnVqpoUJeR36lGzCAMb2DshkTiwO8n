#!/bin/bash

# TOASTer Icon Generation Script
# Converts source icon to all required macOS app icon sizes

set -e

SOURCE_IMAGE="$1"
OUTPUT_DIR="Resources/TOASTer.iconset"

if [ -z "$SOURCE_IMAGE" ]; then
    echo "Usage: $0 <source_image_path>"
    echo "Example: $0 toaster_icon.png"
    exit 1
fi

if [ ! -f "$SOURCE_IMAGE" ]; then
    echo "Error: Source image '$SOURCE_IMAGE' not found"
    exit 1
fi

# Check if ImageMagick or sips is available
if command -v sips &> /dev/null; then
    CONVERT_CMD="sips"
    echo "Using macOS sips for image conversion"
elif command -v convert &> /dev/null; then
    CONVERT_CMD="convert"
    echo "Using ImageMagick for image conversion"
else
    echo "Error: Neither sips nor ImageMagick convert found"
    echo "Install ImageMagick with: brew install imagemagick"
    exit 1
fi

# Create iconset directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Converting '$SOURCE_IMAGE' to macOS app icon sizes..."

# Generate all required icon sizes for macOS
declare -a sizes=(
    "16x16:icon_16x16.png"
    "32x32:icon_16x16@2x.png"
    "32x32:icon_32x32.png"
    "64x64:icon_32x32@2x.png"
    "128x128:icon_128x128.png"
    "256x256:icon_128x128@2x.png"
    "256x256:icon_256x256.png"
    "512x512:icon_256x256@2x.png"
    "512x512:icon_512x512.png"
    "1024x1024:icon_512x512@2x.png"
)

for size_info in "${sizes[@]}"; do
    IFS=':' read -r size filename <<< "$size_info"
    
    if [ "$CONVERT_CMD" = "sips" ]; then
        sips -z ${size/x/ } "$SOURCE_IMAGE" --out "$OUTPUT_DIR/$filename"
    else
        convert "$SOURCE_IMAGE" -resize "$size" "$OUTPUT_DIR/$filename"
    fi
    
    echo "Generated: $filename ($size)"
done

# Generate .icns file using iconutil (macOS built-in)
ICNS_FILE="Resources/TOASTer.icns"
echo "Generating $ICNS_FILE..."
iconutil -c icns "$OUTPUT_DIR" -o "$ICNS_FILE"

# Cleanup iconset directory (optional)
# rm -rf "$OUTPUT_DIR"

echo "✅ Icon generation complete!"
echo "Generated files:"
echo "  - $ICNS_FILE (for macOS app bundle)"
echo "  - $OUTPUT_DIR/ (individual icon sizes)"
echo ""
echo "Next steps:"
echo "1. Add the icon to your CMakeLists.txt"
echo "2. Rebuild your project"
