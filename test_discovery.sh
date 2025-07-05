#!/bin/bash

echo "🔍 Testing JAM Framework v2 Discovery Messages..."

# Listen for UDP multicast on the TOAST port
echo "📡 Listening for discovery/heartbeat messages on 239.255.77.77:7777..."
echo "   (This will show any TOAST messages being sent)"

# Use tcpdump to monitor multicast traffic
sudo tcpdump -i any -n host 239.255.77.77 and port 7777 &
TCPDUMP_PID=$!

echo "🎯 Monitor started (PID: $TCPDUMP_PID)"
echo "📱 Now start TWO TOASTer instances and click 'Connect' in both"
echo "⏰ Monitoring for 30 seconds..."

sleep 30

echo "🛑 Stopping monitor..."
sudo kill $TCPDUMP_PID 2>/dev/null

echo "✅ Test complete. If you saw packets, UDP multicast is working!"
