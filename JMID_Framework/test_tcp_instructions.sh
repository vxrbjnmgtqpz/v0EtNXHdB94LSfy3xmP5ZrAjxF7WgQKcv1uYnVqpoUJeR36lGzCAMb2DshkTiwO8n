#!/bin/bash

echo "🚀 TOAST TCP Two-Process Communication Test"
echo "=========================================="
echo ""
echo "This test demonstrates real TCP communication between separate processes"
echo "using the TOAST protocol for MIDI message exchange."
echo ""

echo "📋 Test Instructions:"
echo "1. Terminal 1: ./toast_server     (starts server on port 8080)"
echo "2. Terminal 2: ./toast_client     (connects and sends MIDI sequence)"
echo "3. Terminal 3: ./toast_client [ip] (for remote testing)"
echo ""

echo "🎯 Expected Results:"
echo "✅ Server accepts client connections"
echo "✅ Client sends MIDI noteOn/noteOff sequence (C4-C5)"
echo "✅ Server receives and acknowledges all messages"
echo "✅ Bidirectional message exchange verified"
echo "✅ Clean disconnection handling"
echo ""

echo "🌐 Network Testing:"
echo "- Same machine: ./toast_client"
echo "- Remote machine: ./toast_client <server-ip>"
echo "- Multiple clients: Run multiple toast_client instances"
echo ""

echo "🛠️  Build Commands:"
echo "Server: g++ -std=c++17 -I./include -I./build_standalone/_deps/nlohmann_json-src/include -pthread toast_server.cpp -L./build_standalone -ljmid_framework -o toast_server"
echo "Client: g++ -std=c++17 -I./include -I./build_standalone/_deps/nlohmann_json-src/include -pthread toast_client.cpp -L./build_standalone -ljmid_framework -o toast_client"
echo ""

echo "Ready to test TCP TOAST communication!"
echo "Run './toast_server' in one terminal, then './toast_client' in another."
