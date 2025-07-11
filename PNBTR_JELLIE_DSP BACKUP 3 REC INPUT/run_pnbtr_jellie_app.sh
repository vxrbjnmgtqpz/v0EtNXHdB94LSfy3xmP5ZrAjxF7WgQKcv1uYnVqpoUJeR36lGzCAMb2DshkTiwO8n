#!/bin/zsh

# Run the PNBTR+JELLIE Training Testbed with real processing components

echo "🔍 Verifying application status..."

# Navigate to project directory
cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/vst3_plugin

# Check if the app exists
if [ -d "build/pnbtr_jellie_gui_app.app" ]; then
  echo "✅ Application found in build directory"
  APP_PATH="build/pnbtr_jellie_gui_app.app"
else
  # Try to find it elsewhere
  FOUND_APP=$(find . -name "pnbtr_jellie_gui_app.app" -type d | head -n 1)
  
  if [ -n "$FOUND_APP" ]; then
    echo "✅ Application found at: $FOUND_APP"
    APP_PATH="$FOUND_APP"
  else
    echo "❌ Application not found. Run fix_pnbtr_jellie_app.sh first to build it."
    echo "👉 Running fix script now..."
    cd ..
    ./fix_pnbtr_jellie_app.sh
    exit 0
  fi
fi

echo "🚀 Launching PNBTR+JELLIE Training Testbed with real components..."
open "$APP_PATH"

echo ""
echo "👉 Instructions:"
echo "1. Click ▶️ START to begin real audio processing"
echo "2. Speak into your microphone - you should see real waveforms"
echo "3. Check that all displays show real data, not placeholder patterns"
echo "4. If issues persist, run fix_pnbtr_jellie_app.sh to rebuild"
