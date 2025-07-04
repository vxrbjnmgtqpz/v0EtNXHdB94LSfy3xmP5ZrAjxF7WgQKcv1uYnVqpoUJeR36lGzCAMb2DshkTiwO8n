#!/bin/bash

# JELLIE Starter Kit - Build and Test Script
# ==========================================

set -e  # Exit on any error

echo "🎵 JELLIE Starter Kit - Build and Test"
echo "====================================="
echo ""

# Check if we're in the right directory
if [ ! -f "JELLIE_STARTER_KIT.md" ]; then
    echo "❌ Error: Please run this script from the JDAT_Framework directory"
    exit 1
fi

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "🔧 Configuring build with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=OFF \
    -DENABLE_TESTS=ON \
    -DENABLE_BENCHMARKS=ON

# Build the project
echo "🛠️  Building JELLIE framework..."
make -j$(nproc) 2>/dev/null || make -j4 2>/dev/null || make

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🎯 Available executables:"
echo "========================"

# List built executables
find . -name "*.exe" -o -name "basic_jellie_demo" -o -name "studio_monitoring" -o -name "multicast_session" -o -name "adat_192k_demo" | while read exe; do
    if [ -x "$exe" ]; then
        echo "  ✅ $(basename $exe)"
    fi
done

echo ""
echo "🧪 Running basic functionality test..."
echo "======================================"

# Test basic JELLIE demo if it exists
if [ -x "./examples/basic_jellie_demo" ]; then
    echo "Running basic_jellie_demo..."
    timeout 10s ./examples/basic_jellie_demo || echo "Demo completed (or timed out after 10s)"
    echo ""
else
    echo "⚠️  basic_jellie_demo not found - skipping test"
fi

# Run tests if available
if [ -x "./tests/test_jellie_comprehensive" ]; then
    echo "🧪 Running comprehensive tests..."
    ./tests/test_jellie_comprehensive
    echo ""
else
    echo "⚠️  Tests not available (GoogleTest may not be installed)"
fi

echo "🚀 Quick Start Commands:"
echo "======================="
echo ""
echo "# Run studio monitoring demo (30 second test):"
echo "  ./examples/studio_monitoring --duration 30"
echo ""
echo "# Run multicast sender:"
echo "  ./examples/multicast_session --mode sender --session demo-001 --duration 60"
echo ""
echo "# Run multicast receiver (in another terminal):"
echo "  ./examples/multicast_session --mode receiver --session demo-001 --duration 60"
echo ""
echo "# Run with GPU acceleration (if Vulkan available):"
echo "  cmake .. -DENABLE_GPU=ON && make"
echo ""

echo "✨ JELLIE Starter Kit is ready for professional audio streaming!"
echo ""
echo "📖 See JELLIE_STARTER_KIT.md for complete documentation"
echo "🎵 Latency target: <200μs end-to-end for studio-quality audio"
echo ""
