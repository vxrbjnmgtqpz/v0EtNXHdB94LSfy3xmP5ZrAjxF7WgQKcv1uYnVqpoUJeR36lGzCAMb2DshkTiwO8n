#!/bin/bash

echo "🔍 PNBTR+JELLIE System Diagnosis"
echo "================================"
echo

# Check current directory
echo "📍 Current Directory:"
pwd
echo

# Check if app exists
echo "🎯 Application File Check:"
APP_PATH="build/PnbtrJellieTrainer_artefacts/Release/PNBTR+JELLIE Training Testbed.app/Contents/MacOS/PNBTR+JELLIE Training Testbed"

if [ -f "$APP_PATH" ]; then
    echo "✅ App found at: $APP_PATH"
    echo "📊 App file size: $(ls -lh "$APP_PATH" | awk '{print $5}')"
    echo "🕐 Last modified: $(stat -f "%Sm" "$APP_PATH")"
else
    echo "❌ App NOT found at: $APP_PATH"
    echo "🔍 Looking for app in build directory..."
    find build -name "*.app" -type d 2>/dev/null | head -5
fi
echo

# Check Metal shaders
echo "🎨 Metal Shaders Check:"
if [ -d "shaders" ]; then
    echo "✅ Shaders directory exists"
    echo "📊 Shader count: $(ls shaders/*.metal 2>/dev/null | wc -l)"
    echo "📝 Shader files:"
    ls shaders/*.metal 2>/dev/null | sed 's/^/   /'
else
    echo "❌ Shaders directory not found"
fi
echo

# Check if app is currently running
echo "🚀 Process Check:"
if pgrep -f "PNBTR+JELLIE Training Testbed" > /dev/null; then
    echo "✅ App is currently running"
    echo "🔢 Process ID: $(pgrep -f "PNBTR+JELLIE Training Testbed")"
else
    echo "❌ App is not running"
fi
echo

# Check build status
echo "🔨 Build Status:"
if [ -f "build/Makefile" ]; then
    echo "✅ Build system configured"
    cd build
    if make --dry-run >/dev/null 2>&1; then
        echo "✅ Build system ready"
    else
        echo "⚠️  Build system may need reconfiguration"
    fi
    cd ..
else
    echo "❌ Build system not configured"
fi
echo

# Check Core Audio system
echo "🎵 Core Audio System Check:"
if command -v system_profiler >/dev/null; then
    echo "📊 Audio devices:"
    system_profiler SPAudioDataType | grep -E "(Input|Output)" | head -10 | sed 's/^/   /'
else
    echo "⚠️  Cannot query audio devices"
fi
echo

# Check permissions
echo "🔐 Permissions Check:"
if [ -x "$APP_PATH" ]; then
    echo "✅ App is executable"
else
    echo "❌ App is not executable or not found"
fi

if [ -x "launch_app.sh" ]; then
    echo "✅ Launch script is executable"
else
    echo "❌ Launch script is not executable"
fi
echo

# Test launch script
echo "🚀 Launch Script Test:"
if [ -f "launch_app.sh" ]; then
    echo "✅ Launch script exists"
    echo "📝 Script content preview:"
    head -10 launch_app.sh | sed 's/^/   /'
else
    echo "❌ Launch script not found"
fi
echo

echo "🎯 DIAGNOSIS COMPLETE"
echo "===================="

# Recommendations
echo
echo "📋 RECOMMENDATIONS:"
if [ ! -f "$APP_PATH" ]; then
    echo "1. ❗ BUILD THE APP: cd build && make"
fi

if ! pgrep -f "PNBTR+JELLIE Training Testbed" > /dev/null; then
    echo "2. 🚀 LAUNCH THE APP: ./launch_app.sh"
fi

echo "3. 🔍 CHECK CONSOLE: Look for debugging output when pressing buttons"
echo "4. 🎯 TEST DEBUGGING: Try the debugging buttons in the GUI"
echo 