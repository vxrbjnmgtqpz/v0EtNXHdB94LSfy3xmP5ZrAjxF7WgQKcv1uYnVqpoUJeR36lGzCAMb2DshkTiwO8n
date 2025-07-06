# TOASTer Build Instructions - Ready to Execute

## ✅ **SETUP COMPLETED**

I've already prepared everything for you:

1. **✅ Icon Support Added**: Updated CMakeLists.txt with `ICON_BIG "Resources/TOASTer.icns"`
2. **✅ Icon Generation Scripts**: Created both `generate_icons.sh` and `create_toaster_icon.sh`
3. **✅ Resources Directory**: Created `/Users/timothydowler/Projects/MIDIp2p/TOASTer/Resources/`
4. **✅ Basic Icon**: System icon copied to `Resources/TOASTer.icns` as placeholder

## 🚀 **BUILD COMMANDS TO EXECUTE**

Run these commands in Terminal:

```bash
# Step 1: Navigate to TOASTer directory
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer

# Step 2: (Optional) Add your custom icon
# Save your toaster icon image as 'toaster_icon.png' then run:
# ./generate_icons.sh toaster_icon.png

# Step 3: Create fresh build directory
rm -rf build_fresh
mkdir build_fresh
cd build_fresh

# Step 4: Configure build with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Step 5: Build TOASTer application
make -j$(sysctl -n hw.ncpu) TOASTer

# Step 6: Launch the built application
open TOASTer_artefacts/Release/TOASTer.app
```

## 🎯 **Expected Results**

After building successfully, you should see:

```
✅ TOASTer.app created in: TOASTer_artefacts/Release/
✅ Application size: ~50-100MB
✅ Icon: Will show basic app icon (or your custom toaster icon if added)
✅ Launch time: <5 seconds
✅ GPU initialization: Success message (or fallback warning)
```

## 🔧 **If Build Issues Occur**

**Missing CMake:**
```bash
# Install via Homebrew
brew install cmake

# Or install Xcode Command Line Tools
xcode-select --install
```

**JUCE Download Issues:**
```bash
# Check internet connection, then clean and retry
rm -rf build_fresh
mkdir build_fresh && cd build_fresh
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Framework Dependency Issues:**
```bash
# Build frameworks first
cd /Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2
mkdir -p build && cd build
cmake .. && make -j4

cd /Users/timothydowler/Projects/MIDIp2p/JMID_Framework  
mkdir -p build && cd build
cmake .. && make -j4

# Then return to TOASTer build
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer/build_fresh
cmake .. && make -j4 TOASTer
```

## 🎨 **To Add Your Custom Icon**

1. Save your toaster icon image as: `toaster_icon.png` in the TOASTer directory
2. Run: `./generate_icons.sh toaster_icon.png`
3. Rebuild: `make TOASTer`

The icon will automatically be integrated into the app bundle!

## 📱 **Testing the Built App**

Once TOASTer.app launches:

1. **Check GPU Status**: Should show successful GPU initialization
2. **Test MIDI Panel**: Verify MIDI device detection
3. **Test Network Panel**: Check WiFi/network discovery
4. **Test Transport**: Use play/stop controls
5. **Check Performance**: Monitor GPU performance metrics

**Status: ✅ READY TO BUILD - Execute the commands above!**
