#!/bin/bash

echo "🚀 Building PNBTR+JELLIE DSP Standalone Testbeds"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Please run this script from the PNBTR_JELLIE_DSP directory"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

echo ""
echo "🔧 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

echo ""
echo "🔨 Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build completed successfully!"
echo ""

# Check if audio testbed was built
if [ -f "standalone/audio_testbed/pnbtr_audio_test" ]; then
    echo "🎯 Audio Testbed ready: ./standalone/audio_testbed/pnbtr_audio_test"
    echo ""
    echo "Try these commands:"
    echo "  # Run comprehensive test suite"
    echo "  ./standalone/audio_testbed/pnbtr_audio_test --run-all-tests"
    echo ""
    echo "  # Test specific audio file" 
    echo "  ./standalone/audio_testbed/pnbtr_audio_test test_audio/sample.wav"
    echo ""
    echo "  # Show help"
    echo "  ./standalone/audio_testbed/pnbtr_audio_test --help"
    echo ""
    
    # Run a quick demonstration
    echo "🧪 Running quick demonstration..."
    echo ""
    ./standalone/audio_testbed/pnbtr_audio_test --run-all-tests
else
    echo "⚠️  Audio testbed not found. Check build output above."
fi

echo ""
echo "🏁 Build script completed!"
echo ""
echo "Revolutionary Features Available:"
echo "  ✅ Zero-Noise Dither Replacement → Mathematical LSB reconstruction"
echo "  ✅ 50ms Neural Audio Extrapolation → Packet loss recovery"
echo "  ✅ GPU-Accelerated Processing → <1ms latency targets"
echo "  ✅ Professional Quality Analysis → SNR, THD+N, LUFS metrics"
echo "" 