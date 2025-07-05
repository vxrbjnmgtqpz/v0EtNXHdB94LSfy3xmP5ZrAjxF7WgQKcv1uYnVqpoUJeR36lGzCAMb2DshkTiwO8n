#!/bin/bash

# Test UDP multicast functionality on macOS
echo "🧪 Testing UDP Multicast on macOS..."

# Check if multicast routing is enabled
echo "📡 Checking multicast routing..."
netstat -rn | grep -E "239\.|224\."

# Test if we can bind to multicast address
echo "📡 Testing multicast bind capability..."
nc -u -l 7777 &
NC_PID=$!
sleep 1

# Try to send a test packet
echo "📡 Sending test multicast packet..."
echo "TEST_PACKET" | nc -u 239.255.77.77 7777

# Cleanup
kill $NC_PID 2>/dev/null

echo "📡 Checking for TOASTer process..."
ps aux | grep TOASTer

echo "📡 Checking network interfaces..."
ifconfig | grep -A1 -B1 "inet.*192\.168\|inet.*10\.\|inet.*172\."

echo "📡 Testing basic UDP connectivity..."
# Try to listen on port and see if anything receives
timeout 5 tcpdump -i any -n port 7777 &
TCPDUMP_PID=$!

# Launch TOASTer in background to test
echo "📡 Test complete. Review output above for network issues."
