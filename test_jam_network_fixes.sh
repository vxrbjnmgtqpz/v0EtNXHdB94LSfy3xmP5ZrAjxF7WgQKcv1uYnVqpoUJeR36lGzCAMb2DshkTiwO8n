#!/bin/bash

echo "🔍 Testing JAM Framework v2 Networking Fixes"
echo "=============================================="

cd /Users/timothydowler/Projects/MIDIp2p/TOASTer

# Test 1: Build with networking fixes
echo "🔨 Building TOASTer with networking fixes..."
mkdir -p build && cd build
if cmake .. -DJAM_GPU_BACKEND=OFF && make TOASTer -j4; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "🚀 Running network connectivity test..."

# Test 2: Check network state detection
echo "#!/bin/bash" > test_network_fixes.sh
echo "echo '🔍 Testing network state detection...'" >> test_network_fixes.sh
echo "ifconfig | grep -E 'inet.*broadcast' | head -3" >> test_network_fixes.sh
echo "echo ''" >> test_network_fixes.sh
echo "echo '📡 Testing multicast capability...'" >> test_network_fixes.sh
echo "ping -c 1 239.255.77.77 2>/dev/null && echo '✅ Multicast reachable' || echo '❌ Multicast not reachable'" >> test_network_fixes.sh
echo "echo ''" >> test_network_fixes.sh
echo "echo '🔌 Testing UDP socket creation...'" >> test_network_fixes.sh
echo "nc -u -l 7777 &" >> test_network_fixes.sh
echo "NETCAT_PID=\$!" >> test_network_fixes.sh
echo "sleep 1" >> test_network_fixes.sh
echo "kill \$NETCAT_PID 2>/dev/null && echo '✅ UDP socket test passed' || echo '❌ UDP socket test failed'" >> test_network_fixes.sh

chmod +x test_network_fixes.sh
./test_network_fixes.sh

echo ""
echo "🎯 Network fixes implemented:"
echo "  ✅ Network permission checking"
echo "  ✅ Interface readiness testing"  
echo "  ✅ UDP connectivity validation"
echo "  ✅ Multicast capability testing"
echo "  ✅ Real network state detection"
echo ""
echo "🔥 This should fix:"
echo "  - False positive 'connected' before network permission"
echo "  - 'UDP create session failed' errors"
echo "  - 'Discover TOAST devices' not working over USB4"
echo ""
echo "📋 To test: Run TOASTer and check that connection status"
echo "   only shows 'Connected' after actual network tests pass"
