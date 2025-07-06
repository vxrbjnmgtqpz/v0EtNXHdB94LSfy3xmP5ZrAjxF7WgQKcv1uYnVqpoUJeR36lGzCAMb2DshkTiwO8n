#!/bin/bash

# Create a simple TOASTer icon programmatically using macOS tools
# This creates a basic icon until the user's custom icon can be added

set -e

OUTPUT_DIR="Resources/TOASTer.iconset"
ICNS_FILE="Resources/TOASTer.icns"

echo "Creating TOASTer app icon..."

# Create Resources directory
mkdir -p Resources

# Create iconset directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create a simple icon using macOS built-in tools
# This creates a basic rounded square with "T" for TOASTer
create_simple_icon() {
    local size="$1"
    local output="$2"
    
    # Use Python to create a simple icon (available on all macOS systems)
    python3 -c "
import os
from PIL import Image, ImageDraw, ImageFont
import sys

try:
    size = int('$size'.split('x')[0])
    
    # Create image with rounded corners
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle background (toaster brown)
    margin = size // 10
    draw.rounded_rectangle([margin, margin, size-margin, size-margin], 
                          radius=size//8, fill=(139, 69, 19, 255))
    
    # Draw lighter top (like a toaster)
    draw.rounded_rectangle([margin, margin, size-margin, size//2], 
                          radius=size//8, fill=(160, 82, 45, 255))
    
    # Draw simple 'T' or toaster slots
    slot_width = size // 8
    slot_height = size // 4
    slot_y = size // 3
    
    # Two toaster slots
    slot1_x = size // 3
    slot2_x = 2 * size // 3 - slot_width
    
    draw.rectangle([slot1_x, slot_y, slot1_x + slot_width, slot_y + slot_height], 
                  fill=(0, 0, 0, 200))
    draw.rectangle([slot2_x, slot_y, slot2_x + slot_width, slot_y + slot_height], 
                  fill=(0, 0, 0, 200))
    
    # Save image
    img.save('$output', 'PNG')
    print(f'Generated: $output ({size}x{size})')
    
except ImportError:
    # Fallback: create simple colored square if PIL not available
    import subprocess
    cmd = [
        'sips', '--createIcon',
        '--setProperty', 'format', 'png',
        '--resampleHeightWidth', '$size', '$size',
        '/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns',
        '--out', '$output'
    ]
    subprocess.run(cmd, capture_output=True)
    print(f'Generated fallback: $output ({size}x{size})')
    
except Exception as e:
    print(f'Error creating icon: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Check if we can use Python with PIL, otherwise use sips fallback
if python3 -c "from PIL import Image" 2>/dev/null; then
    echo "Using Python PIL for icon generation"
    USE_PIL=true
else
    echo "PIL not available, using sips for basic icon"
    USE_PIL=false
fi

# Generate all required icon sizes
declare -a sizes=(
    "16:icon_16x16.png"
    "32:icon_16x16@2x.png"
    "32:icon_32x32.png"
    "64:icon_32x32@2x.png"
    "128:icon_128x128.png"
    "256:icon_128x128@2x.png"
    "256:icon_256x256.png"
    "512:icon_256x256@2x.png"
    "512:icon_512x512.png"
    "1024:icon_512x512@2x.png"
)

for size_info in "${sizes[@]}"; do
    IFS=':' read -r size filename <<< "$size_info"
    
    if [ "$USE_PIL" = true ]; then
        create_simple_icon "${size}x${size}" "$OUTPUT_DIR/$filename"
    else
        # Fallback using sips
        sips -s format png --resampleHeightWidth $size $size \
             /System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns \
             --out "$OUTPUT_DIR/$filename" >/dev/null 2>&1
        echo "Generated fallback: $filename (${size}x${size})"
    fi
done

# Generate .icns file
echo "Creating $ICNS_FILE..."
iconutil -c icns "$OUTPUT_DIR" -o "$ICNS_FILE"

echo "✅ TOASTer icon generation complete!"
echo "Generated: $ICNS_FILE"
echo ""
echo "To use your custom toaster icon:"
echo "1. Save your icon image as 'toaster_icon.png'"
echo "2. Run: ./generate_icons.sh toaster_icon.png"
