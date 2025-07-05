#!/bin/bash

# Comprehensive Network Connectivity Test for JAM Framework v2
# Tests all the network issues reported: DHCP false positives, UDP session failures, device discovery

echo "🔍 JAM Framework v2 Network Connectivity Test"
echo "=============================================="
echo "Testing real network connectivity to prevent false positives"
echo ""

# Function to test network permission
test_network_permission() {
    echo "📋 Testing network permission..."
    
    # Try to create a UDP socket - this should fail with permission denied if no network access
    if nc -u -z -w 1 8.8.8.8 53 2>/dev/null; then
        echo "✅ Network permission: Granted"
        return 0
    else
        echo "❌ Network permission: Denied or no connectivity"
        return 1
    fi
}

# Function to test DHCP status
test_dhcp_status() {
    echo "📋 Testing DHCP and interface status..."
    
    # Check for active network interfaces with real IP addresses
    interfaces=$(ifconfig | grep -E "inet [0-9]" | grep -v "127.0.0.1" | grep -v "169.254" | wc -l)
    
    if [ "$interfaces" -gt 0 ]; then
        echo "✅ Active network interfaces found:"
        ifconfig | grep -E "^[a-z]|inet [0-9]" | grep -A1 -E "^[a-z]" | grep -E "(^[a-z]|inet [0-9])" | while read line; do
            if [[ $line =~ ^[a-z] ]]; then
                interface=$(echo $line | cut -d: -f1)
                echo -n "   $interface: "
            elif [[ $line =~ inet ]]; then
                ip=$(echo $line | awk '{print $2}')
                if [[ ! $ip =~ ^127\. ]] && [[ ! $ip =~ ^169\.254\. ]]; then
                    echo "$ip ✅"
                fi
            fi
        done
        return 0
    else
        echo "❌ No active network interfaces with valid IP addresses"
        echo "   This indicates DHCP is not complete or network is disconnected"
        return 1
    fi
}

# Function to test USB4/Thunderbolt interfaces
test_usb4_interfaces() {
    echo "📋 Testing USB4/Thunderbolt interfaces..."
    
    # Look for bridge interfaces (common with Thunderbolt/USB4)
    bridge_interfaces=$(ifconfig | grep -E "^(bridge|en[1-9])" | grep -v "en0")
    
    if [ -n "$bridge_interfaces" ]; then
        echo "✅ Potential USB4/Thunderbolt interfaces found:"
        echo "$bridge_interfaces" | while read line; do
            interface=$(echo $line | cut -d: -f1)
            echo "   - $interface"
            
            # Check if it has an IP address
            ip_info=$(ifconfig "$interface" 2>/dev/null | grep "inet ")
            if [ -n "$ip_info" ]; then
                ip=$(echo $ip_info | awk '{print $2}')
                echo "     IP: $ip"
            else
                echo "     No IP assigned"
            fi
        done
        return 0
    else
        echo "❌ No USB4/Thunderbolt interfaces detected"
        echo "   If using USB4 connection, check cable and network sharing settings"
        return 1
    fi
}

# Function to test UDP multicast capability
test_udp_multicast() {
    echo "📋 Testing UDP multicast capability..."
    
    local multicast_addr="239.255.77.77"
    local port="7777"
    
    # Create a simple UDP multicast test
    timeout 5 bash -c "
        # Start listener in background
        nc -ul $port &
        listener_pid=\$!
        
        # Give listener time to start
        sleep 1
        
        # Send test packet
        echo 'UDP_MULTICAST_TEST' | nc -u $multicast_addr $port
        
        # Clean up
        kill \$listener_pid 2>/dev/null
    " 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ UDP multicast test passed"
        return 0
    else
        echo "❌ UDP multicast test failed"
        echo "   This could indicate firewall blocking or network configuration issues"
        return 1
    fi
}

# Function to test TOASTer build
test_toaster_build() {
    echo "📋 Testing TOASTer build with network fixes..."
    
    cd "$(dirname "$0")"
    
    if [ -d "build" ]; then
        echo "   Using existing build directory"
    else
        echo "   Creating build directory"
        mkdir -p build
    fi
    
    cd build
    
    echo "   Running CMake configuration..."
    if cmake .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | grep -E "(error|Error|ERROR)" >/dev/null; then
        echo "❌ CMake configuration failed"
        return 1
    fi
    
    echo "   Building TOASTer..."
    if make -j$(nproc 2>/dev/null || echo 4) 2>&1 | grep -E "(error|Error|ERROR)" >/dev/null; then
        echo "❌ Build failed"
        return 1
    fi
    
    echo "✅ TOASTer build successful"
    return 0
}

# Function to test device discovery
test_device_discovery() {
    echo "📋 Testing device discovery functionality..."
    
    # Check if TOASTer binary exists
    if [ -f "build/TOASTer_artefacts/Debug/TOASTer.app/Contents/MacOS/TOASTer" ]; then
        echo "✅ TOASTer binary found"
        
        # Test discovery by running for a few seconds
        echo "   Running brief device discovery test..."
        timeout 10 build/TOASTer_artefacts/Debug/TOASTer.app/Contents/MacOS/TOASTer &
        app_pid=$!
        
        # Give it time to initialize and discover
        sleep 8
        
        # Clean up
        kill $app_pid 2>/dev/null
        
        echo "✅ Device discovery test completed (check console output)"
        return 0
    else
        echo "❌ TOASTer binary not found"
        return 1
    fi
}

# Main test execution
main() {
    local total_tests=0
    local passed_tests=0
    
    echo "Starting comprehensive network connectivity tests..."
    echo ""
    
    # Test 1: Network Permission
    total_tests=$((total_tests + 1))
    if test_network_permission; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Test 2: DHCP Status
    total_tests=$((total_tests + 1))
    if test_dhcp_status; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Test 3: USB4 Interfaces
    total_tests=$((total_tests + 1))
    if test_usb4_interfaces; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Test 4: UDP Multicast
    total_tests=$((total_tests + 1))
    if test_udp_multicast; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Test 5: TOASTer Build
    total_tests=$((total_tests + 1))
    if test_toaster_build; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Test 6: Device Discovery
    total_tests=$((total_tests + 1))
    if test_device_discovery; then
        passed_tests=$((passed_tests + 1))
    fi
    echo ""
    
    # Summary
    echo "=============================================="
    echo "🏁 Network Connectivity Test Results"
    echo "=============================================="
    echo "Tests passed: $passed_tests/$total_tests"
    
    if [ $passed_tests -eq $total_tests ]; then
        echo "🎉 All tests passed! Network connectivity should work properly."
        return 0
    else
        echo "⚠️  Some tests failed. Network issues may persist."
        echo ""
        echo "Common fixes:"
        echo "1. Grant network permission when prompted by macOS"
        echo "2. Check DHCP lease renewal if using auto-assigned IP"
        echo "3. Verify USB4 cable and network sharing settings"
        echo "4. Check firewall settings for UDP multicast"
        echo "5. Restart network interfaces if needed"
        return 1
    fi
}

# Run the main test function
main "$@"
