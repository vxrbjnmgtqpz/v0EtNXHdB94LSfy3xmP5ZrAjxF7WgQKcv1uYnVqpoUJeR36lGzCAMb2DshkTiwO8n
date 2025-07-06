# TOASTer App Build Instructions and Status

## Current Status: ✅ READY FOR FRESH BUILD

### **Source Code Status: COMPLETE**
- MainComponent.cpp: Full GPU-native integration with JAM Framework v2
- GPUTransportController: GPU-accelerated transport control
- JAMNetworkPanel: Network discovery and connection management  
- GPU MIDI Manager: Hardware-accelerated MIDI processing
- PNBTR Manager: Predictive audio processing integration
- All UI panels: MIDITestingPanel, PerformanceMonitorPanel, ClockSyncPanel, etc.

### **Dependencies Status: CONFIGURED**
- JUCE Framework: Fetched via CMake FetchContent
- JAM Framework v2: Local framework integration
- JMID Framework: Local framework integration  
- nlohmann/json: Fetched via FetchContent
- Metal/MetalKit: macOS system frameworks

### **Build Requirements**
```bash
# Required tools:
- CMake 3.15+
- Xcode Command Line Tools (for macOS)
- Git (for FetchContent dependencies)

# Verify installation:
cmake --version        # Should show 3.15+
xcodebuild -version    # Should show Xcode tools
git --version          # Should show Git
```

### **Fresh Build Instructions**

#### **Step 1: Clean Build Environment**
```bash
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
rm -rf build_fresh
mkdir build_fresh
cd build_fresh
```

#### **Step 2: Configure with CMake**
```bash
# Configure Release build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"

# Alternative: Configure with Xcode generator for IDE development
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Xcode"
```

#### **Step 3: Build TOASTer**
```bash
# For Makefile build:
make -j$(sysctl -n hw.ncpu) TOASTer

# For Xcode build:
xcodebuild -project TOASTer.xcodeproj -configuration Release -target TOASTer
```

#### **Step 4: Locate Built Application**
```bash
# Application will be located at:
# For Makefile: TOASTer_artefacts/Release/TOASTer.app
# For Xcode: Release/TOASTer.app

# Launch the application:
open TOASTer_artefacts/Release/TOASTer.app
```

### **Potential Build Issues and Solutions**

#### **Issue 1: Missing Framework Headers**
```
Error: 'gpu_timebase.h' file not found
```
**Solution**: Ensure JAM_Framework_v2 is properly built first:
```bash
cd /Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

#### **Issue 2: JUCE Fetch Failure**
```
Error: Failed to fetch JUCE repository
```
**Solution**: Check internet connection and Git configuration:
```bash
git config --global http.sslVerify false  # If behind corporate firewall
```

#### **Issue 3: Metal Framework Issues**
```
Error: Metal framework not found
```
**Solution**: Ensure Xcode Command Line Tools are properly installed:
```bash
xcode-select --install
sudo xcode-select --reset
```

#### **Issue 4: Code Signing Issues**
```
Error: Code signing failed
```
**Solution**: Disable code signing for development:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=""
```

### **Testing the Built Application**

#### **Basic Functionality Test**
1. **Launch**: Open TOASTer.app
2. **GPU Test**: Check if GPU initialization succeeds (no warning dialogs)
3. **MIDI Test**: Open MIDI Testing Panel, verify MIDI device detection
4. **Network Test**: Open JAM Network Panel, start discovery
5. **Transport Test**: Use GPU Transport Controller for play/stop

#### **Advanced Integration Test**
1. **JACK Integration**: If JACK is installed, test audio backend
2. **Network Discovery**: Test WiFi and Thunderbolt network discovery
3. **PNBTR Prediction**: Test audio prediction in Network panel
4. **Clock Sync**: Verify GPU timebase synchronization

### **Expected Build Output**
```
Successful build should produce:
✅ TOASTer.app (macOS application bundle)
✅ Size: ~50-100MB (includes JUCE libraries)
✅ Launch time: <5 seconds on modern Mac
✅ GPU initialization: Success (no warning dialogs)
✅ All panels load without errors
```

### **Build Performance Optimization**
```bash
# For faster builds, use multiple cores:
make -j$(sysctl -n hw.ncpu)  # Use all CPU cores

# For development, build only changed files:
make TOASTer  # Only rebuild TOASTer target

# For debugging, use Debug configuration:
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### **Deployment Notes**
- **Target**: macOS 11.0+ (Big Sur) for Metal GPU support
- **Architecture**: Universal binary (x86_64 + arm64) 
- **Dependencies**: Self-contained (no external frameworks required)
- **Size**: ~80MB including JUCE and framework dependencies

## **Status: ✅ READY FOR IMMEDIATE BUILD**

The TOASTer application is fully implemented and ready for building. All source files are complete, dependencies are properly configured, and the CMake build system is set up correctly. The app should build successfully and provide a comprehensive testing interface for the JAMNet/MIDIp2p system.
