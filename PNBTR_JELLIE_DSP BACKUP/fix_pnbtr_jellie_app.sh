#!/bin/zsh

# PNBTR+JELLIE Training Testbed Fix Script
# This script rebuilds the application with real processing enabled

echo "🔧 PNBTR+JELLIE Training Testbed Fix"
echo "🧹 Cleaning old build artifacts..."

# Navigate to project directory
cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/vst3_plugin

# Remove old build artifacts
if [ -d "build" ]; then
  rm -rf build
  echo "✅ Removed old build directory"
else
  echo "ℹ️ No existing build directory found"
fi

# Create new build directory
mkdir -p build
cd build

# Configure with real processing enabled
echo "🔄 Configuring CMake with real processing enabled..."
cmake -DUSE_REAL_PROCESSING=ON -DDISABLE_PLACEHOLDER_DATA=ON ..

# Build the application
echo "🔨 Building application with real processing components..."
make -j$(sysctl -n hw.ncpu)

# Verify build was successful
if [ -d "pnbtr_jellie_gui_app.app" ]; then
  echo "✅ Build successful!"
  
  # Check file size to ensure we have real components
  size=$(du -k "pnbtr_jellie_gui_app.app/Contents/MacOS/pnbtr_jellie_gui_app" | cut -f1)
  echo "📊 Application binary size: ${size}KB"
  
  if [ $size -lt 100 ]; then
    echo "⚠️ Warning: Binary size seems small, might not include all components"
  else
    echo "✅ Binary size looks good, real components should be included"
  fi
  
  echo "🚀 Launching application..."
  open pnbtr_jellie_gui_app.app
else
  echo "❌ Build failed. Check error messages above."
  exit 1
fi

echo ""
echo "📝 Verification Steps:"
echo "1. Check that microphone input shows real waveforms (green oscilloscope)"
echo "2. Verify JELLIE encoding is processing real audio (middle displays)"
echo "3. Confirm metrics show real calculations, not random values"
echo "4. Test recording and export functionality"
echo ""
echo "If issues persist, please check the console logs for errors."
