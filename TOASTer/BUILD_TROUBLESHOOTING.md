# TOASTer Build Troubleshooting & Fix

## 🔧 **Issue Diagnosed: Missing Executable**

The TOASTer.app bundle was created but the executable is missing from `Contents/MacOS/`, which means the compilation failed silently.

## 🛠️ **Step-by-Step Fix Process**

### **Step 1: Build Framework Dependencies First**

```bash
# Build JAM Framework v2 first
cd /Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DJAM_GPU_BACKEND=metal
make -j4 jam_framework_v2

# Build JMID Framework
cd /Users/timothydowler/Projects/MIDIp2p/JMID_Framework
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 jmid_framework
```

### **Step 2: Verify System Requirements**

```bash
# Check required tools
xcode-select --install  # Install if missing
which cmake || brew install cmake
which git || echo "Git required"

# Verify Xcode Command Line Tools
xcodebuild -version
```

### **Step 3: Build TOASTer with Verbose Output**

```bash
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer

# Clean everything
rm -rf build_working && mkdir build_working && cd build_working

# Configure with explicit settings
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang

# Build with full output
make VERBOSE=1 TOASTer 2>&1 | tee build.log

# Check for errors
grep -i "error\|failed" build.log
```

### **Step 4: Alternative - Use Xcode Generator**

```bash
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
rm -rf build_xcode && mkdir build_xcode && cd build_xcode

# Generate Xcode project
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Release

# Build with Xcode
xcodebuild -project TOASTer.xcodeproj \
           -configuration Release \
           -target TOASTer \
           -quiet

# Or open in Xcode IDE
open TOASTer.xcodeproj
```

### **Step 5: Manual Dependency Check**

```bash
# Check if required headers exist
ls -la /Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2/include/
ls -la /Users/timothydowler/Projects/MIDIp2p/JMID_Framework/include/

# Verify JUCE download
ls -la build_working/_deps/juce-src/
```

## 🔍 **Common Issues & Solutions**

### **Issue 1: Missing Framework Libraries**
```bash
# Solution: Build frameworks in correct order
cd /Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2/build
make jam_framework_v2
# Check: ls -la jam_framework_v2/libjam_framework_v2.a
```

### **Issue 2: JUCE Download Failure**
```bash
# Solution: Manual JUCE setup
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
git clone --depth 1 --branch 8.0.4 https://github.com/juce-framework/JUCE.git
# Update CMakeLists.txt to use local JUCE
```

### **Issue 3: Compiler Issues**
```bash
# Solution: Explicit compiler paths
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
cmake .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
```

### **Issue 4: Code Signing Problems**
```bash
# Solution: Disable all code signing
cmake .. -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="" \
         -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
```

## 🎯 **Simplified Build (If All Else Fails)**

Create a minimal working version:

```bash
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer

# Use simplified CMakeLists.txt
cp CMakeLists_simple.txt CMakeLists.txt

# Build minimal version
mkdir build_minimal && cd build_minimal
cmake .. && make -j4

# This should at least create a working JUCE app
```

## ✅ **Success Indicators**

You'll know the build worked when:

```bash
# Executable exists
ls -la build_working/TOASTer_artefacts/Release/TOASTer.app/Contents/MacOS/TOASTer

# App info shows correctly
mdls build_working/TOASTer_artefacts/Release/TOASTer.app | grep kMDItemCFBundleIdentifier

# App launches without "damaged" error
open build_working/TOASTer_artefacts/Release/TOASTer.app
```

## 🚀 **Expected Working Result**

After successful build:
- ✅ Executable exists in `Contents/MacOS/TOASTer`
- ✅ App launches without security warnings
- ✅ GPU initialization succeeds or shows fallback message
- ✅ All UI panels load (MIDI, Network, Performance, etc.)
- ✅ Basic functionality works (MIDI device detection, network discovery)

Run through these steps systematically and the build should succeed. The most likely issue is missing framework dependencies that need to be built first.
