#!/bin/bash

# Test Audio Pipeline Script
# This script tests the Core Audio → Metal pipeline using the debugging functions

echo "🔧 Testing PNBTR+JELLIE Audio Pipeline..."

# Check if app is running
if ! pgrep -f "PNBTR+JELLIE Training Testbed" > /dev/null; then
    echo "❌ App not running. Please launch the app first with:"
    echo "   ./launch_app.sh"
    exit 1
fi

echo "✅ App is running"

# The debugging functions can be called from within the app
# This script serves as a guide for testing steps

echo ""
echo "🎯 MANUAL TESTING STEPS:"
echo ""
echo "1. In the app, click 'Use Default Input Device' button"
echo "2. Click 'Enable Sine Test' to bypass microphone"
echo "3. Click 'Check MetalBridge Status' to verify GPU pipeline"
echo "4. Click 'Force Callback' to trigger audio processing"
echo "5. Watch the console for audio callback messages"
echo ""
echo "💡 If you see '[🔁 CoreAudio INPUT CALLBACK #X]' messages, the pipeline is working!"
echo ""
echo "📋 Expected Success Flow:"
echo "   ✅ AudioUnit initialized with default input device"
echo "   ✅ Record arms enabled"
echo "   ✅ Audio callbacks firing"
echo "   ✅ MetalBridge processing audio"
echo "   ✅ GPU shaders executing"
echo ""
echo "🔧 If audio still doesn't work, the issue is likely in:"
echo "   - MetalBridge initialization"
echo "   - Metal shader compilation"
echo "   - GPU pipeline configuration"
echo ""
echo "Ready to test! 🚀" 