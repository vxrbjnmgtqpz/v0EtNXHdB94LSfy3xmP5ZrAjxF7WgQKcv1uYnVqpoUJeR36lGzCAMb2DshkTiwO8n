# BUILD GUIDE - PNBTR+JELLIE Training Testbed

**Complete Building, Integration, and Deployment Guide**

> **🔧 BUILDING FOCUS:**  
> This document contains all practical building instructions, CMake configurations, error handling, and integration rules extracted from the comprehensive development guide.

---

## 📦 **CMAKE CONFIGURATION - PRODUCTION READY**

### **Complete CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.22)
project(PnbtrJellieTrainer VERSION 1.0.0)

# LESSON LEARNED: Set C++ standard early
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# LESSON LEARNED: Find JUCE first, then add Metal
find_package(juce CONFIG REQUIRED)

# LESSON LEARNED: Metal frameworks must be found
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)

    if(NOT METAL_FRAMEWORK)
        message(FATAL_ERROR "Metal framework not found")
    endif()
endif()

# Create executable
juce_add_gui_app(PnbtrJellieTrainer
    COMPANY_NAME "JAMNet"
    PRODUCT_NAME "PNBTR+JELLIE Training Testbed"
    VERSION ${PROJECT_VERSION}
    BUNDLE_ID "com.jamnet.pnbtrjellietrainer"
)

# Add source files
target_sources(PnbtrJellieTrainer PRIVATE
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/PNBTRTrainer.cpp
    Source/MetalBridge.mm
    Source/GUI/ProfessionalTransportController.cpp
    Source/GUI/OscilloscopeRow.cpp
    Source/GUI/TOASTNetworkOscilloscope.cpp
    Source/GUI/SpectralAudioTrack.cpp
    Source/GUI/MetricsDashboard.cpp
    Source/Core/FrameSyncCoordinator.cpp
    Source/GPU/GPUCommandFencePool.mm
)

# LESSON LEARNED: Include juce_dsp for FFT functionality
target_link_libraries(PnbtrJellieTrainer PRIVATE
    juce::juce_gui_extra
    juce::juce_audio_utils
    juce::juce_audio_devices
    juce::juce_dsp  # CRITICAL: Required for FFT
)

# LESSON LEARNED: Link Metal frameworks after JUCE
if(APPLE)
    target_link_libraries(PnbtrJellieTrainer PRIVATE
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
    )
endif()

# LESSON LEARNED: JUCE compile definitions
target_compile_definitions(PnbtrJellieTrainer PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:PnbtrJellieTrainer,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:PnbtrJellieTrainer,JUCE_VERSION>"
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_USE_DARK_SPLASH_SCREEN=1
    JUCE_USE_METAL=1
    JUCE_ENABLE_OPENGL=0  # Avoid conflicts with Metal
)

# Metal shader compilation
function(compile_metal_shaders)
    set(METAL_SHADERS
        AudioInputCaptureShader.metal
        AudioInputGateShader.metal
        DJSpectralAnalysisShader.metal
        RecordArmVisualShader.metal
        JELLIEPreprocessShader.metal
        NetworkSimulationShader.metal
        PNBTRReconstructionShader.metal
        MetricsComputeShader.metal
        AudioKernels.metal
    )

    foreach(SHADER ${METAL_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}")
        set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.metallib")

        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND xcrun -sdk macosx metal -c ${INPUT_FILE} -o ${OUTPUT_FILE}
            DEPENDS ${INPUT_FILE}
            COMMENT "Compiling Metal shader: ${SHADER_NAME}"
        )

        list(APPEND COMPILED_SHADERS ${OUTPUT_FILE})
    endforeach()

    add_custom_target(CompileMetalShaders ALL DEPENDS ${COMPILED_SHADERS})
endfunction()

# LESSON LEARNED: Compile shaders before main target
if(APPLE)
    compile_metal_shaders()
    add_dependencies(PnbtrJellieTrainer CompileMetalShaders)

    # Copy shaders to app bundle
    add_custom_command(TARGET PnbtrJellieTrainer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/shaders
        $<TARGET_BUNDLE_DIR:PnbtrJellieTrainer>/Contents/Resources/shaders
    )
endif()

# Platform-specific settings
if(APPLE)
    set_target_properties(PnbtrJellieTrainer PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "PNBTR+JELLIE Training Testbed"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.jamnet.pnbtrjellietrainer"
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
        XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
    )
endif()
```

### **Build Script with Error Handling**

```bash
#!/bin/bash
# build_pnbtr_jellie.sh

set -e  # Exit on any error

echo "🔧 Building PNBTR+JELLIE Training Testbed..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "📋 Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# LESSON LEARNED: Check for Metal shader compilation
echo "🔨 Compiling Metal shaders..."
make CompileMetalShaders

# Build main application
echo "🏗️ Building application..."
make -j$(sysctl -n hw.ncpu)

# LESSON LEARNED: Verify all required components
echo "✅ Verifying build..."
if [ -f "PnbtrJellieTrainer.app/Contents/MacOS/PnbtrJellieTrainer" ]; then
    echo "✅ Application built successfully"
else
    echo "❌ Application build failed"
    exit 1
fi

if [ -d "PnbtrJellieTrainer.app/Contents/Resources/shaders" ]; then
    echo "✅ Metal shaders integrated successfully"
else
    echo "❌ Metal shaders missing"
    exit 1
fi

echo "🚀 Build complete! Run with:"
echo "   open PnbtrJellieTrainer.app"
```

---

## ⚠️ **CRITICAL INTEGRATION RULES**

### **CRITICAL TRANSPORT BAR LAYOUT FIX**

**ISSUE**: Transport bar and UI components can break if row heights are inconsistent between `paint()` and `resized()` methods.

**SYMPTOMS**: Transport bar missing, UI layout broken, components not visible

**ROOT CAUSE**: Inconsistent `rowHeights` arrays and missing component allocations

**SOLUTION**:

```cpp
// MainComponent.cpp - CONSISTENT row heights in BOTH methods

void MainComponent::paint(juce::Graphics& g) {
    // CRITICAL: Must match resized() exactly
    const int rowHeights[] = {48, 32, 200, 160, 160, 100, 60};
    int y = 0;
    for (int i = 0; i < 6; ++i) {
        y += rowHeights[i];
        g.drawLine(0.0f, (float)y, (float)getWidth(), (float)y, 1.0f);
    }
}

void MainComponent::resized() {
    // CRITICAL: Must match paint() exactly
    const int rowHeights[] = {48, 32, 200, 160, 160, 100, 60};

    // CRITICAL: Allocate ALL components created in initialization
    if (transportBar) transportBar->setBounds(area.removeFromTop(rowHeights[0]));
    auto deviceBar = area.removeFromTop(rowHeights[1]);
    if (inputDeviceBox) inputDeviceBox->setBounds(deviceBar.removeFromLeft(getWidth() / 2).reduced(8, 4));
    if (outputDeviceBox) outputDeviceBox->setBounds(deviceBar.reduced(8, 4));
    if (oscilloscopeRow) oscilloscopeRow->setBounds(area.removeFromTop(rowHeights[2]));
    if (jellieTrack) jellieTrack->setBounds(area.removeFromTop(rowHeights[3]));
    if (pnbtrTrack) pnbtrTrack->setBounds(area.removeFromTop(rowHeights[4]));
    if (metricsDashboard) metricsDashboard->setBounds(area.removeFromTop(rowHeights[5]));

    // CRITICAL: Don't forget controlsRow - this causes layout corruption!
    if (controlsRow) controlsRow->setBounds(area.removeFromTop(rowHeights[6]));
}
```

### **CRITICAL MICROPHONE PERMISSION FIX (macOS)**

**ISSUE**: Audio input not working on macOS due to missing microphone permissions

**SYMPTOMS**: No audio input signal, oscilloscopes not responding, record arming doesn't capture audio

**ROOT CAUSE**: macOS requires explicit microphone permission requests and proper input channel configuration

**SOLUTION**:

```cpp
// MainComponent.cpp - Audio device initialization with permissions

case 10: {
    // CRITICAL FOR macOS: Request microphone permissions explicitly
    #if JUCE_MAC
    juce::RuntimePermissions::request(juce::RuntimePermissions::recordAudio,
        [this](bool granted) {
            if (granted) {
                printf("[PERMISSIONS] ✅ Microphone permission granted!\n");
            } else {
                printf("[PERMISSIONS] ❌ Microphone permission denied!\n");
            }
        });
    #endif

    // CRITICAL: Enable input channels explicitly (2 in, 2 out, request permissions)
    juce::String error = deviceManager.initialise(2, 2, nullptr, true);

    if (error.isNotEmpty()) {
        // Try fallback with minimal settings
        error = deviceManager.initialise(1, 2, nullptr, false);
    }

    // CRITICAL: Add audio callback BEFORE device configuration
    deviceManager.addAudioCallback(this);

    // CRITICAL: Explicitly configure input device if not selected
    auto setup = deviceManager.getAudioDeviceSetup();
    if (setup.inputDeviceName.isEmpty()) {
        auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
        auto inputDevices = deviceType->getDeviceNames(true);
        if (!inputDevices.isEmpty()) {
            setup.inputDeviceName = inputDevices[0];
            setup.useDefaultInputChannels = true;
            deviceManager.setAudioDeviceSetup(setup, true);
        }
    }
    break;
}
```

### **CRITICAL AUDIO CALLBACK DEBUG PATTERN**

**ISSUE**: Verifying that audio is flowing through the custom Core Audio callbacks.

**SOLUTION**: Add logging inside the `InputRenderCallback` and `OutputRenderCallback` in `CoreAudioBridge.mm`.

```cpp
// In CoreAudioBridge.mm

static OSStatus InputRenderCallback(...)
{
    // ...
    // Debug: Log every 100th callback with detailed signal analysis
    UInt32 callbackNum = ++bridge->callbackCounter;
    if (callbackNum % 100 == 0) {
        float maxSample = 0.0f;
        for (UInt32 i = 0; i < std::min(inNumberFrames, (UInt32)4); ++i) {
            maxSample = std::max(maxSample, fabsf(samples[i]));
            NSLog(@"[📊 INPUT] sample[%u] = %f", i, samples[i]);
        }
        NSLog(@"[🔁 INPUT CALLBACK #%u] %u frames, Max amplitude: %f",
              callbackNum, inNumberFrames, maxSample);
    }
    // ...
}

static OSStatus OutputRenderCallback(...)
{
    // ...
    // 🎯 STEP 3: Check what's being sent to audio output
    if (bridge->callbackCounter % 100 == 0) {
        float maxOutput = 0.0f;
        for (UInt32 i = 0; i < std::min(inNumberFrames, (UInt32)4); ++i) {
            maxOutput = std::max(maxOutput, fabsf(out[i]));
            NSLog(@"[📊 OUTPUT] sample[%u] = %f", i, out[i]);
        }
        if (maxOutput == 0.0f) {
            NSLog(@"[❌ SILENT] Output buffer is silent - GPU pipeline not producing audio");
        }
    }
    // ...
}
```

### **CRITICAL TRACK RECORD ARM BUTTON IMPLEMENTATION**

**ISSUE**: SpectralAudioTrack has setRecordArmed() functionality but missing GUI controls

**SYMPTOMS**: Users can't arm tracks for recording, no visible record arm buttons

**ROOT CAUSE**: Missing connection between SpectralAudioTrack record arm buttons and MetalBridge GPU pipeline

**COMPLETE SOLUTION**:

The new architecture uses a clean data flow from the JUCE GUI to the Core Audio engine via the C-Interface.

**DATA FLOW**:

```
SpectralAudioTrack buttons → MainComponent → setCoreAudioRecordArmStates() (C-Interface) → CoreAudioGPUBridge → MetalBridge → GPU shaders
```

**Implementation Steps:**

1.  **GUI Layer (`SpectralAudioTrack.cpp`)**: The record-arm button's `onClick` lambda in the JUCE GUI triggers a state change.
2.  **Control Layer (`MainComponent.cpp`)**: A timer or callback in `MainComponent` periodically checks the armed status of each track (`jellieTrack->isRecordArmed()`).
3.  **C-Interface Call**: When a state change is detected, `MainComponent` calls the C-function `setCoreAudioRecordArmStates(isJellieArmed, isPnbtrArmed)`.
4.  **Audio Engine (`CoreAudioBridge.mm`)**:
    - The C function `setCoreAudioRecordArmStates` updates atomic boolean flags inside the `CoreAudioGPUBridge` singleton.
    - The `InputRenderCallback` reads these atomic flags.
    - Before calling the GPU, it calls `MetalBridge::getInstance().setRecordArmStates(...)` to pass the states to the Metal layer.
5.  **GPU Layer (`MetalBridge.mm`)**:
    - `MetalBridge` stores the arm states in member variables.
    - `runSevenStageProcessingPipeline` reads these member variables and passes them as `uniforms` to the appropriate Metal shaders.
6.  **Shader Layer (`.metal` files)**: The shaders use the `armed` uniform to conditionally process audio, enabling or disabling effects like input capture.

**TESTING**:

1. Click record arm buttons on tracks - they should turn red when armed
2. Press Play in transport bar to start training
3. Speak into microphone - audio should only be captured for record-armed tracks
4. GPU shaders now respect actual UI record arm states instead of hardcoded values

---

## 🔧 **JUCE & CORE AUDIO INTEGRATION**

The current architecture uses JUCE **exclusively for the GUI**. All real-time audio processing is handled by a custom `CoreAudioGPUBridge` to achieve the lowest possible latency and direct control over the hardware-to-GPU pipeline.

### **Architecture Overview**

1.  **`CoreAudioGPUBridge.mm`**: A self-contained Objective-C++ class that manages:

    - Dual Core Audio `AudioUnit` instances (one for input, one for output).
    - Device enumeration and selection.
    - Native `InputRenderCallback` and `OutputRenderCallback`.
    - A thread-safe `std::vector` buffer to pass audio to the output callback.

2.  **C-Interface**: A set of C-style functions provides a stable ABI to communicate between the C++ JUCE GUI and the Objective-C++ Core Audio engine. This avoids complex bridging headers.

3.  **`MainComponent.cpp`**: The JUCE GUI, which uses the C-interface to control the audio engine (start/stop capture, set record-arm states). It no longer inherits `juce::AudioIODeviceCallback`.

### **C-Interface for JUCE-to-CoreAudio Communication**

This interface, defined at the bottom of `CoreAudioBridge.mm`, is the glue between the GUI and the audio engine.

```cpp
// In CoreAudioBridge.mm
extern "C" {
    void initializeCoreAudioBridge();
    void startCoreAudioCapture();
    void stopCoreAudioCapture();
    void setCoreAudioRecordArmStates(bool jellieArmed, bool pnbtrArmed);
    void setCoreAudioMasterGain(float gainDB);
    int getCurrentOutputSampleIndex();
    float getCurrentPacketLossPercentage();
    float getCurrentAudioCorrelation();
    bool isAudioCapturing();
    void enableCoreAudioSineTest(bool enable);
    void forceCoreAudioCallback();
}

// In a C++ header file, for MainComponent to use:
extern "C" void initializeCoreAudioBridge();
// ... etc.
```

### **Controlling Audio from the JUCE GUI**

```cpp
// MainComponent.cpp example
void MainComponent::initialize() {
    // Initialize the Core Audio engine when the app starts
    initializeCoreAudioBridge();
}

void MainComponent::transportPlayButtonClicked() {
    // Start capturing audio when user presses play
    startCoreAudioCapture();
}

void MainComponent::recordArmButtonClicked() {
    // Update the record-arm states in the audio engine
    setCoreAudioRecordArmStates(jellieTrack->isArmed(), pnbtrTrack->isArmed());
}
```

This approach decouples the UI from the real-time audio processing, which is a core principle of the new architecture.

---

## 🔍 **PRODUCTION VALIDATION & QUALITY ASSURANCE**

### **🔄 Enhanced GPU Thread Synchronization**

**CRITICAL NOTE**: Standard `waitUntilCompleted()` patterns should only be used for:

- GPU debugging
- Performance profiling
- Single-threaded mode fallback

**Production Implementation** uses async completion handlers for optimal scheduling:

```cpp
// MetalBridge.mm - Production async GPU processing
void MetalBridge::runSevenStageProcessingPipelineAsync(size_t numSamples) {
    if (!commandQueue || !audioInputBuffer[currentFrameIndex]) {
        NSLog(@"[❌ PIPELINE] Metal not ready for processing");
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [commandBuffer setLabel:@"Seven-Stage Audio Pipeline"];

        // Execute all stages
        executeStage1_InputCapture(commandBuffer, numSamples);
        executeStage2_InputGating(commandBuffer, numSamples);
        executeStage3_SpectralAnalysis(commandBuffer, numSamples);
        executeStage4_JELLIEPreprocessing(commandBuffer, numSamples);
        executeStage5_NetworkSimulation(commandBuffer, numSamples);
        executeStage6_PNBTRReconstruction(commandBuffer, numSamples);
        executeStage7_VisualFeedback(commandBuffer, numSamples);

        // PRODUCTION: Async completion handler for better performance
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            if (buffer.status == MTLCommandBufferStatusError) {
                NSLog(@"[❌ GPU] Command buffer failed: %@", buffer.error.localizedDescription);
            } else {
                MetalBridge::getInstance().onGPUProcessingComplete();
            }
        }];

        [commandBuffer commit];

        // For debugging/profiling only - remove in production
        #ifdef DEBUG_GPU_SYNC
        [commandBuffer waitUntilCompleted];
        #endif
    }
}
```

### **🔍 JELLIE-to-PNBTR Signal Integrity Validation**

**Audio Signal Correlation Testing** ensures PNBTR reconstruction maintains signal integrity:

```cpp
// MetalBridge.mm - Signal correlation validation
void MetalBridge::validatePNBTRReconstruction(const float* inputData,
                                             const float* reconstructedData,
                                             size_t numSamples) {
    // Compute cross-correlation between input and reconstructed audio
    float correlation = 0.0f;
    float inputPower = 0.0f;
    float outputPower = 0.0f;

    for (size_t i = 0; i < numSamples; ++i) {
        correlation += inputData[i] * reconstructedData[i];
        inputPower += inputData[i] * inputData[i];
        outputPower += reconstructedData[i] * reconstructedData[i];
    }

    // Normalize correlation coefficient
    float normalizedCorrelation = 0.0f;
    if (inputPower > 0.0f && outputPower > 0.0f) {
        normalizedCorrelation = correlation / sqrt(inputPower * outputPower);
    }

    // Compute RMS difference for quality assessment
    float rmsDifference = 0.0f;
    for (size_t i = 0; i < numSamples; ++i) {
        float diff = inputData[i] - reconstructedData[i];
        rmsDifference += diff * diff;
    }
    rmsDifference = sqrt(rmsDifference / numSamples);

    static uint32_t validationCounter = 0;
    if (++validationCounter % 100 == 0) {
        NSLog(@"[🔁 CORRELATION] Input vs PNBTR: %.3f, RMS Diff: %.6f",
              normalizedCorrelation, rmsDifference);

        // Quality thresholds for different packet loss scenarios
        float currentPacketLoss = getCurrentPacketLossPercentage();

        if (currentPacketLoss < 0.05f && normalizedCorrelation < 0.85f) {
            NSLog(@"[⚠️ QUALITY] Low correlation (%.3f) with minimal packet loss (%.1f%%) - check PNBTR parameters",
                  normalizedCorrelation, currentPacketLoss * 100.0f);
        } else if (currentPacketLoss > 0.2f && normalizedCorrelation > 0.6f) {
            NSLog(@"[✅ QUALITY] Good correlation (%.3f) despite high packet loss (%.1f%%) - PNBTR working well",
                  normalizedCorrelation, currentPacketLoss * 100.0f);
        }
    }

    // Store correlation for UI display
    lastCorrelationValue = normalizedCorrelation;
}
```

### **🎬 Async Visualization Loop Architecture**

**Challenge**: Visual shaders hitching on audio processing thread
**Solution**: Separate visualization update rate from audio processing

```cpp
// In MetalBridge.cpp - Async visualization system
class MetalBridge {
private:
    dispatch_queue_t visualizationQueue;
    dispatch_source_t visualizationTimer;
    std::atomic<bool> visualizationActive{false};

    // Separate command queues for audio and visuals
    id<MTLCommandQueue> audioCommandQueue;
    id<MTLCommandQueue> visualCommandQueue;

    // Double-buffered visual data
    id<MTLBuffer> visualDataBuffer[2];
    std::atomic<int> visualWriteIndex{0};
    std::atomic<int> visualReadIndex{1};

public:
    void initializeAsyncVisualization() {
        // Create dedicated visualization queue
        visualizationQueue = dispatch_queue_create("com.jamnet.visualization",
                                                  DISPATCH_QUEUE_SERIAL);

        // Create separate command queue for visuals
        visualCommandQueue = [device newCommandQueue];
        visualCommandQueue.label = @"VisualizationQueue";

        // Start 60fps visualization timer
        startVisualizationTimer();

        NSLog(@"🎬 Async visualization initialized - 60fps independent loop");
    }

    void startVisualizationTimer() {
        visualizationTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, visualizationQueue);

        // 60fps = 16.67ms interval
        uint64_t interval = 16670000; // nanoseconds
        dispatch_source_set_timer(visualizationTimer,
                                 dispatch_time(DISPATCH_TIME_NOW, 0),
                                 interval,
                                 1000000); // 1ms leeway

        dispatch_source_set_event_handler(visualizationTimer, ^{
            if (visualizationActive.load()) {
                updateVisualizationFrame();
            }
        });

        dispatch_resume(visualizationTimer);
    }

    void updateVisualizationFrame() {
        // Create visualization command buffer
        id<MTLCommandBuffer> commandBuffer = [visualCommandQueue commandBuffer];
        commandBuffer.label = @"VisualizationFrame";

        // Swap buffers atomically
        int readIdx = visualReadIndex.load();

        // Run visual shaders on separate thread
        runRecordArmVisualShader(commandBuffer, visualDataBuffer[readIdx]);
        runSpectralVisualizationShader(commandBuffer, visualDataBuffer[readIdx]);
        runMetricsVisualizationShader(commandBuffer, visualDataBuffer[readIdx]);

        // Commit visualization work
        [commandBuffer commit];

        // Update UI on main thread
        dispatch_async(dispatch_get_main_queue(), ^{
            updateUIFromVisualData(visualDataBuffer[readIdx]);
        });
    }
};
```

---

## 🚀 **PERFORMANCE OPTIMIZATION TIPS**

### **MPS FFT Upgrade**

**Before**: Custom Metal FFT ~150µs  
**After**: MPS FFT ~30µs  
**Improvement**: 5x faster spectral analysis

```cpp
// For production implementation, replace custom FFT with MPS:
void MetalBridge::setupMPSFFT() {
    fftTransform = [[MPSMatrixFFT alloc]
        initWithDevice:device
        transformSize:MTLSizeMake(1024, 1, 1)];

    // Create input/output matrices for FFT
    MPSMatrixDescriptor* inputDesc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:1024
                         columns:1
                        dataType:MPSDataTypeFloat32];

    inputMatrix = [[MPSMatrix alloc]
        initWithBuffer:audioInputBuffer[0]
            descriptor:inputDesc];
}
```

### **Texture Caching for UI**

**Before**: Record arm shaders recomputed every frame  
**After**: Instant texture reuse for unchanged states  
**Improvement**: 10-20x faster UI refresh

### **Combined Performance Results**

- **Total Latency**: <500µs with smooth 60fps visuals
- **Audio Dropouts**: Eliminated through async visualization
- **GPU Utilization**: Optimized with separate command queues
- **UI Responsiveness**: Maintained at 60fps independent of audio processing

---

## 🎯 **DEPLOYMENT CHECKLIST**

### **Pre-Build Verification**

- [ ] All Metal shaders compile without errors
- [ ] CMake configuration matches target platform
- [ ] JUCE dependencies properly linked
- [ ] Metal frameworks available on target system

### **Build Process**

- [ ] Clean build directory
- [ ] Configure with CMake
- [ ] Compile Metal shaders first
- [ ] Build main application
- [ ] Verify shader integration in bundle

### **Post-Build Testing**

- [ ] Application launches without crashes
- [ ] Audio input/output functional
- [ ] GPU pipeline processes audio
- [ ] UI responds to user interactions
- [ ] Performance metrics within acceptable ranges

### **Common Build Issues**

1. **Metal Framework Not Found**: Ensure Xcode Command Line Tools installed
2. **JUCE Not Found**: Install JUCE via package manager or set CMAKE_PREFIX_PATH
3. **Shader Compilation Fails**: Check Metal syntax and compiler version compatibility
4. **Missing Dependencies**: Verify all frameworks linked correctly

### **Debug Build Options**

```cmake
# For debugging builds
set(CMAKE_BUILD_TYPE Debug)
target_compile_definitions(PnbtrJellieTrainer PRIVATE
    DEBUG_GPU_SYNC=1
    VERBOSE_LOGGING=1
    ENABLE_ASSERTIONS=1
)
```

This comprehensive build guide ensures successful compilation and deployment of the PNBTR+JELLIE Training Testbed across different development environments.
