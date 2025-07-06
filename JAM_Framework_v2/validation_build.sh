#!/bin/bash

# Simple validation tool builder
# This builds just the validation tools without complex dependencies

echo "🔍 Building JAM Framework v2 Validation Tools..."

# Create build directory
mkdir -p build
cd build

# Simple compilation for JSON performance test
echo "📊 Building JSON Performance Validation..."
g++ -std=c++17 -O2 \
    -I../include \
    -I../examples \
    ../examples/json_performance_validation.cpp \
    -o json_performance_validation \
    -lcurl -ljson-c 2>/dev/null || \
g++ -std=c++17 -O2 \
    -I../include \
    -I../examples \
    ../examples/json_performance_validation.cpp \
    -o json_performance_validation

# Simple compilation for network diagnostic tool
echo "🌐 Building Network Diagnostic Tool..."
g++ -std=c++17 -O2 \
    -I../include \
    -I../examples \
    ../examples/network_diagnostic_tool.cpp \
    -o network_diagnostic_tool

echo "✅ Validation tools built successfully!"
echo "   - json_performance_validation"
echo "   - network_diagnostic_tool"
echo ""
echo "🚀 Run with:"
echo "   ./json_performance_validation"
echo "   ./network_diagnostic_tool"
