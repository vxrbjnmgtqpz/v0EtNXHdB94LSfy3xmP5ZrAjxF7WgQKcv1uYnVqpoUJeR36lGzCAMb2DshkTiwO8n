# TOASTer Icon Integration Guide

## Step 1: Save Your Icon Image

Save your TOASTer icon image as:
`/Users/timothydowler/Projects/MIDIp2p/TOASTer/toaster_icon.png`

**Recommended specifications:**
- Format: PNG with transparency
- Size: 1024x1024 pixels (minimum)
- Quality: High resolution, clean edges
- Background: Transparent or solid color

## Step 2: Generate Icon Files

Run the icon generation script:

```bash
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
./generate_icons.sh toaster_icon.png
```

This will create:
- `Resources/TOASTer.icns` - Main macOS icon file
- `Resources/TOASTer.iconset/` - Individual icon sizes

## Step 3: Add Icon to JUCE Project

The icon will be automatically added to your CMakeLists.txt configuration.

## Manual Steps if Needed

If you prefer to do it manually:

1. **Save your icon** as `toaster_icon.png` in the TOASTer directory
2. **Run:** `./generate_icons.sh toaster_icon.png`
3. **Rebuild:** Your project will automatically use the new icon

## Troubleshooting

**If sips command not found:**
- This is unusual on macOS, but you can install ImageMagick as backup:
```bash
brew install imagemagick
```

**If icon doesn't appear:**
- Clean build directory: `rm -rf build*`
- Rebuild project completely
- Check that `Resources/TOASTer.icns` exists

**Icon quality issues:**
- Use higher resolution source image (1024x1024 minimum)
- Ensure source has transparent background
- Avoid complex details that don't scale well

Your vintage toaster icon looks perfect for this! It should scale nicely to all required sizes.
