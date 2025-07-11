#!/bin/bash
# build_pnbtr_jellie.sh
# PNBTR+JELLIE Training Testbed Build Script
# Based on lessons learned from comprehensive development guide

set -e  # Exit on any error

echo "🔧 Building PNBTR+JELLIE Training Testbed..."
echo "📖 Using lessons learned from comprehensive development guide"

# Verify prerequisites
echo "🔍 Checking prerequisites..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake 3.22+"
    exit 1
fi

# Check for Xcode tools
if ! command -v xcrun &> /dev/null; then
    echo "❌ Xcode command line tools not found. Please install Xcode"
    exit 1
fi

# Check for Metal compiler
if ! xcrun -find metal &> /dev/null; then
    echo "❌ Metal compiler not found. Please install Xcode with Metal support"
    exit 1
fi

echo "✅ Prerequisites verified"

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "📋 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# LESSON LEARNED: Check for Metal shader compilation first
echo "🔨 Compiling Metal shaders..."
if make CompileMetalShaders; then
    echo "✅ Metal shaders compiled successfully"
else
    echo "❌ Metal shader compilation failed"
    exit 1
fi

# Build main application
echo "🏗️ Building application..."
CPU_COUNT=$(sysctl -n hw.ncpu)
echo "Using ${CPU_COUNT} CPU cores for parallel build"

if make -j${CPU_COUNT}; then
    echo "✅ Application compiled successfully"
else
    echo "❌ Application compilation failed"
    exit 1
fi

# LESSON LEARNED: Verify all required components
echo "🔍 Verifying build outputs..."

# Check for executable
if [ -f "PnbtrJellieTrainer.app/Contents/MacOS/PnbtrJellieTrainer" ]; then
    echo "✅ Application executable found"
else
    echo "❌ Application executable missing"
    exit 1
fi

# Check for Metal shaders in bundle
if [ -d "PnbtrJellieTrainer.app/Contents/Resources/shaders" ]; then
    echo "✅ Metal shaders integrated successfully"
    SHADER_COUNT=$(ls PnbtrJellieTrainer.app/Contents/Resources/shaders/*.metallib 2>/dev/null | wc -l)
    echo "   Found ${SHADER_COUNT} compiled Metal shaders"
else
    echo "❌ Metal shaders missing from app bundle"
    exit 1
fi

# Check app bundle structure
if [ -f "PnbtrJellieTrainer.app/Contents/Info.plist" ]; then
    echo "✅ App bundle structure valid"
else
    echo "❌ Invalid app bundle structure"
    exit 1
fi

# Display build information
echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "📊 Build Summary:"
echo "   • Application: PnbtrJellieTrainer.app"
echo "   • Metal Shaders: ${SHADER_COUNT} compiled"
echo "   • Build Type: Release"
echo "   • Architecture: $(uname -m)"
echo ""
echo "🚀 To run the application:"
echo "   open PnbtrJellieTrainer.app"
echo ""
echo "🔧 To run with console output:"
echo "   ./PnbtrJellieTrainer.app/Contents/MacOS/PnbtrJellieTrainer"
echo ""
echo "📖 For development guidance, see:"
echo "   ../PBJ_DOCUMENTATION/COMPREHENSIVE_GPU_AUDIO_DEVELOPMENT_GUIDE.md" 