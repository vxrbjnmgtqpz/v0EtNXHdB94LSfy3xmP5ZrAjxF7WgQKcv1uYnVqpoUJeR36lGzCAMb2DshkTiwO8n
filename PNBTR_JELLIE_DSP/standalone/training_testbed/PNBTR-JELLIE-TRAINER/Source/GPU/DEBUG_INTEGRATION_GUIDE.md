# GPU Debug Integration Usage Guide

## Overview
This debug system helps isolate GPU frame coordination issues by providing detailed logging and dummy GPU kernels to test the pipeline without complex DSP processing.

## Quick Integration

### 1. Add to MetalBridge.h
```cpp
#include "GPU/MetalBridgeDebugIntegration.h"
#include "Core/FrameStateTracker.h"

// In MetalBridge class private section:
MetalBridgeDebugIntegration& debugIntegration;
FrameStateTracker frameStateTracker;
bool enableDebugMode = false;  // Toggle for debug mode
```

### 2. Initialize in MetalBridge::initialize()
```cpp
bool MetalBridge::initialize() {
    // ... existing initialization code ...
    
    // Initialize debug system
    if (!debugIntegration.initialize(metalDevice)) {
        NSLog(@"[WARNING] Debug integration failed to initialize");
    }
    
    // Configure debug mode
    MetalBridgeDebugIntegration::DebugConfig config;
    config.enableFrameLogging = true;
    config.enableGPUValidation = true;
    config.useDummyProcessing = true;  // START WITH TRUE TO TEST PIPELINE
    config.enableDetailedPipelineLogging = true;
    config.debugTestGain = 1.0f;
    debugIntegration.setDebugConfig(config);
    
    enableDebugMode = true;  // Enable debug mode
    
    return initialized;
}
```

### 3. Replace GPU Processing Section
In your `runSevenStageProcessingPipelineWithSync()` or wherever you dispatch GPU commands:

```cpp
void MetalBridge::runSevenStageProcessingPipelineWithSync(size_t numSamples) {
    if (!initialized || !metalLibrary) return;
    
    // ... existing setup code ...
    
    if (enableDebugMode) {
        // Use debug processing instead of complex DSP
        bool success = debugIntegration.processAudioFrameWithFullDebugging(
            commandBuffer,
            audioInputBuffer[currentFrameIndex],
            reconstructedBuffer[currentFrameIndex],  // or whatever output buffer you use
            numSamples,
            currentFrameIndex
        );
        
        if (!success) {
            NSLog(@"[ERROR] Debug processing failed for frame %d", currentFrameIndex);
        }
        
        return;  // Skip normal processing
    }
    
    // ... existing normal GPU processing code ...
}
```

## Expected Output

When working correctly, you should see:
```
🔧 METAL BRIDGE DEBUG INTEGRATION INITIALIZED 🔧
Frame logging: ✅
GPU validation: ✅
Dummy processing: ✅
Pipeline logging: ✅
===================================================

[✅ CHECKPOINT 1] Frame processing started | Frame: 0
[✅ CHECKPOINT 2] CoreAudio→MetalBridge input validated | Peak: 0.000947 | Frame: 0
[DEBUG MODE] Using dummy GPU processing for frame 0
[GPUDebugHelper] Processing frame 0 with 512 samples
[GPU DISPATCH] Frame 0 | DISPATCH: ✅ COMMIT: ❌ COMPLETE: ❌ GPU_RDY: ❌
[GPU COMMIT] Frame 0 | DISPATCH: ✅ COMMIT: ✅ COMPLETE: ❌ GPU_RDY: ❌
[✅ CHECKPOINT 3] Dummy GPU processing dispatched | Frame: 0
[✅ CHECKPOINT 4] GPU command buffer committed | Frame: 0
[GPU COMPLETE] Frame 0 | DISPATCH: ✅ COMMIT: ✅ COMPLETE: ✅ GPU_RDY: ✅
[GPUDebugHelper] ✅ Frame 0 completed successfully
```

## Debugging Strategy

### Phase 1: Test Basic Pipeline
1. Set `useDummyProcessing = true`
2. Run and verify you get successful frame completion logs
3. If this fails, the issue is in GPU setup, not DSP

### Phase 2: Test With Validation
1. Keep `useDummyProcessing = true`
2. Set `enableGPUValidation = true`
3. Check that non-zero sample counts are logged

### Phase 3: Gradually Add Real Processing
1. Once dummy processing works, slowly replace with real DSP
2. Add one kernel at a time
3. Monitor for frame completion failures

## Common Issues Fixed

- **Silent GPU Output**: Debug system will show exactly where pipeline breaks
- **Frame Coordination**: Detailed logging shows which stages complete
- **Command Buffer Issues**: Tracks dispatch → commit → complete lifecycle
- **Buffer Validation**: Confirms audio data is flowing through pipeline

## Toggle Debug Mode
You can toggle debug mode at runtime:
```cpp
// In your GUI or debug menu:
metaBridge.enableDebugMode = !metaBridge.enableDebugMode;
```

This allows you to A/B test between debug processing and normal processing.
