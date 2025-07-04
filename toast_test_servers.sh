#!/bin/bash
# Enhanced TOAST Test Server Script - Simulates multiple connected devices for comprehensive testing

echo "🧪 Starting Enhanced TOAST Multi-Device Test Environment"
echo "========================================================="

# Function to start a more realistic TCP server that responds to TOAST connections
start_toast_server() {
    local port=$1
    local name=$2
    
    echo "📡 Starting TOAST server '$name' on port $port"
    
    # More realistic server that responds to multiple types of requests
    while true; do
        {
            echo "HTTP/1.1 200 OK"
            echo "Content-Type: application/json"
            echo "Server: TOASTer-TestDevice-$name"
            echo ""
            echo "{\"device\":\"$name\",\"port\":$port,\"status\":\"ready\",\"sync\":\"enabled\",\"timestamp\":$(date +%s)}"
        } | nc -l $port
        sleep 0.1
    done &
    
    echo $! # Return process ID
}

# Start multiple test servers on different ports
echo "🚀 Launching test servers..."
SERVER1_PID=$(start_toast_server 8080 "TOASTer-Device-1")
SERVER2_PID=$(start_toast_server 8081 "TOASTer-Device-2") 
SERVER3_PID=$(start_toast_server 8082 "TOASTer-Device-3")
SERVER4_PID=$(start_toast_server 3000 "TOASTer-Device-WebDev")
SERVER5_PID=$(start_toast_server 9000 "TOASTer-Device-Alt")

# Give servers a moment to start
sleep 1

echo ""
echo "🌐 Enhanced test servers running:"
echo "   📱 Device 1: localhost:8080 (Primary)"
echo "   📱 Device 2: localhost:8081 (Secondary)"  
echo "   📱 Device 3: localhost:8082 (Tertiary)"
echo "   📱 Web Dev: localhost:3000 (Development)"
echo "   📱 Alt Device: localhost:9000 (Alternative)"
echo ""
echo "💡 Testing Instructions:"
echo "   1. Launch TOASTer.app from: build/TOASTer_artefacts/Release/TOASTer.app"
echo "   2. Select 'DHCP Auto' protocol in the Network Connection panel"
echo "   3. Click Connect - should auto-discover and connect to Device 1"
echo "   4. Test transport sync: Play/Stop should work across multiple instances"
echo "   5. Launch another TOASTer instance to test multi-device sync"
echo ""
echo "🔧 Advanced Testing:"
echo "   - Launch multiple TOASTer instances for real multi-device simulation"
echo "   - Test Bonjour discovery with 'Discover Devices' dropdown"
echo "   - Verify automatic transport synchronization between devices"
echo "   - Test different protocol types (TCP, UDP, DHCP Auto)"
echo ""
echo "Press Ctrl+C to stop all test servers..."

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping all test servers..."
    kill $SERVER1_PID $SERVER2_PID $SERVER3_PID $SERVER4_PID $SERVER5_PID 2>/dev/null
    echo "✅ All servers stopped. Test environment shut down."
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM

# Keep script running and show periodic status
echo "⏱️  Servers active. Monitoring connections..."
while true; do
    sleep 10
    echo "📊 $(date): Test servers still running - Device ports 8080-8082, 3000, 9000"
done
