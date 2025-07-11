#!/bin/bash

# PNBTR+JELLIE Training Testbed Launch Script
# Makes it easy to launch the app without typing the long path

echo "🚀 Launching PNBTR+JELLIE Training Testbed..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the build directory
cd "$SCRIPT_DIR/build"

# Check if the app exists
APP_PATH="PnbtrJellieTrainer_artefacts/Release/PNBTR+JELLIE Training Testbed.app/Contents/MacOS/PNBTR+JELLIE Training Testbed"

if [ -f "$APP_PATH" ]; then
    echo "✅ Found app at: $APP_PATH"
    echo "🎯 Launching..."
    
    # Launch the app
    "$APP_PATH" &
    
    echo "🎉 App launched! Check for the GUI window."
    echo "📋 Console output will appear in this terminal."
else
    echo "❌ App not found at: $APP_PATH"
    echo "🔧 Try running: cd build && make"
fi 