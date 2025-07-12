#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>
#endif
#include "MetalBridge.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <thread> // For GPU processing thread
#include <chrono> // For sleep
#include "RingBuffer.h" // Our new lock-free ring buffer

/**
 * Core Audio → Metal GPU Pipeline Bridge
 * Clean dual-AudioUnit implementation (input + output separated)
 * Bypasses JUCE audio processing for direct Core Audio → Metal processing
 */

//==============================================================================
// Global State and Bridge Struct
//==============================================================================

struct CoreAudioGPUBridge; // Forward declaration

// Global singleton instance of the bridge.
static CoreAudioGPUBridge* globalBridge = nullptr;

/**
 * @struct CoreAudioGPUBridge
 * @brief Manages all Core Audio and GPU processing state.
 */
struct CoreAudioGPUBridge {
    AudioUnit inputUnit = nullptr;
    AudioUnit outputUnit = nullptr;
    AudioDeviceID selectedInputDevice = kAudioObjectUnknown;
    AudioDeviceID selectedOutputDevice = kAudioObjectUnknown;

    LockFreeRingBuffer inputRingBuffer;
    LockFreeRingBuffer outputRingBuffer;

    std::atomic<bool> isCapturing{false};
    std::atomic<UInt32> callbackCounter{0};
    
    std::thread gpuProcessingThread;
    std::atomic<bool> runGpuThread{false};

    std::atomic<bool> jellieRecordArmed{false};
    std::atomic<bool> pnbtrRecordArmed{false};

    std::vector<AudioDeviceID> availableInputDevices;
    std::vector<std::string> inputDeviceNames;
    std::vector<AudioDeviceID> availableOutputDevices;
    std::vector<std::string> outputDeviceNames;
};


//==============================================================================
// Forward Declarations for Core Audio & Bridge Functions
//==============================================================================

static OSStatus InputRenderCallback(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList* ioData);
static OSStatus OutputRenderCallback(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList* ioData);
static void gpuProcessingLoop(CoreAudioGPUBridge* bridge);
void stopAUHALInputOutput(CoreAudioGPUBridge* bridge);
bool startAUHALInputOutput(CoreAudioGPUBridge* bridge, AudioDeviceID inputDev, AudioDeviceID outputDev);
static void categorizeDevice(CoreAudioGPUBridge* bridge, AudioDeviceID deviceID);
static std::string getDeviceName(AudioDeviceID deviceID);
void enumerateDevices(CoreAudioGPUBridge* bridge);


//==============================================================================
// GPU Processing Thread
//==============================================================================

static void gpuProcessingLoop(CoreAudioGPUBridge* bridge) {
    NSLog(@"[GPU Thread] Starting loop.");
    MetalBridge& metalBridge = MetalBridge::getInstance();

    uint64_t nextFrameIndex = 0;
    while(bridge->runGpuThread.load()) {
        AudioFrame inputFrame;
        if (bridge->inputRingBuffer.pop(inputFrame)) {
            AudioFrame outputFrame = inputFrame;
            outputFrame.frameIndex = nextFrameIndex++;

            metalBridge.setRecordArmStates(bridge->jellieRecordArmed, bridge->pnbtrRecordArmed);

            metalBridge.processAudioBlock(
                (const float*)inputFrame.samples,
                (float*)outputFrame.samples,
                inputFrame.sample_count
            );

            if (!bridge->outputRingBuffer.push(outputFrame)) {
                NSLog("[GPU Thread] Warning: Output ring buffer full, dropping frame.");
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    NSLog("[GPU Thread] Exiting loop.");
}


//==============================================================================
// Core Audio Callbacks
//==============================================================================

static OSStatus InputRenderCallback(void* inRefCon,
                                    AudioUnitRenderActionFlags* ioActionFlags,
                                    const AudioTimeStamp* inTimeStamp,
                                    UInt32 inBusNumber,
                                    UInt32 inNumberFrames,
                                    AudioBufferList* ioData)
{
    CoreAudioGPUBridge* bridge = (CoreAudioGPUBridge*)inRefCon;
    if (!bridge || !bridge->isCapturing) {
        return noErr;
    }

    // Allocate buffer list for AudioUnitRender - handle mono input correctly
    AudioBufferList* bufferList = (AudioBufferList*)malloc(sizeof(AudioBufferList));
    bufferList->mNumberBuffers = 1;
    bufferList->mBuffers[0].mNumberChannels = 1; // Mono input (hardware detected 1 channel)
    bufferList->mBuffers[0].mDataByteSize = inNumberFrames * sizeof(float);
    bufferList->mBuffers[0].mData = malloc(bufferList->mBuffers[0].mDataByteSize);

    OSStatus status = AudioUnitRender(bridge->inputUnit, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, bufferList);

    if (status != noErr) {
        NSLog(@"[❌] InputRenderCallback: AudioUnitRender failed with status %d", status);
        free(bufferList->mBuffers[0].mData);
        free(bufferList);
        return status;
    }

    // Debug: Log successful render every 100 callbacks
    UInt32 callbackNum = ++bridge->callbackCounter;
    if (callbackNum % 100 == 0) {
        float* samples = (float*)bufferList->mBuffers[0].mData;
        float maxSample = 0.0f;
        for (UInt32 i = 0; i < std::min(inNumberFrames, (UInt32)4); ++i) {
            maxSample = std::max(maxSample, fabsf(samples[i]));
        }
        NSLog(@"[✅ INPUT CALLBACK #%u] %u frames, Max amplitude: %f", callbackNum, inNumberFrames, maxSample);
    }

    // Convert mono input to stereo AudioFrame
    AudioFrame frame;
    frame.hostTime = inTimeStamp->mHostTime;
    frame.frameIndex = frame.hostTime; // Use hostTime as canonical frameIndex
    frame.sample_count = inNumberFrames;

    float* inputSamples = (float*)bufferList->mBuffers[0].mData;
    for (uint32_t i = 0; i < inNumberFrames; ++i) {
        frame.samples[0][i] = inputSamples[i];  // Left channel = mono input
        frame.samples[1][i] = inputSamples[i];  // Right channel = duplicate mono
    }

    if (!bridge->inputRingBuffer.push(frame)) {
        // NSLog("[InputRenderCallback] Warning: Input ring buffer full, dropping frame.");
    }
    
    free(bufferList->mBuffers[0].mData);
    free(bufferList);
    return noErr;
}

static OSStatus OutputRenderCallback(void* inRefCon,
                                     AudioUnitRenderActionFlags* ioActionFlags,
                                     const AudioTimeStamp* inTimeStamp,
                                     UInt32 inBusNumber,
                                     UInt32 inNumberFrames,
                                     AudioBufferList* ioData)
{
    CoreAudioGPUBridge* bridge = (CoreAudioGPUBridge*)inRefCon;
    float* out = (float*)ioData->mBuffers[0].mData;

    AudioFrame frame;
    if (bridge && bridge->outputRingBuffer.pop(frame)) {
        MetalBridge& metalBridge = MetalBridge::getInstance();
        auto* frameSyncCoordinator = metalBridge.getFrameSyncCoordinator();
        uint64_t frameIndex = frame.frameIndex;

        NSLog("[RENDER] frameIndex = %llu", frameIndex);
        while (!frameSyncCoordinator->isFrameReadyFor(SyncRole::AudioOutput, frameIndex)) {
            __asm__ __volatile__("" ::: "memory");
        }

        for (uint32_t i = 0; i < frame.sample_count; ++i) {
            if ((i * 2 + 1) < (inNumberFrames * 2)) {
                out[i * 2] = frame.samples[0][i];
                out[i * 2 + 1] = frame.samples[1][i];
            }
        }

        static int outputCheckpointCounter = 0;
        if (++outputCheckpointCounter % 200 == 0) {
            float outputPeak = 0.0f;
            for (uint32_t i = 0; i < inNumberFrames * 2; ++i) {
                outputPeak = std::max(outputPeak, fabsf(out[i]));
            }
            if (outputPeak > 0.0001f) {
                NSLog("[✅ CHECKPOINT 6] Hardware Output: Peak %.6f - Audio reaching speakers", outputPeak);
            } else {
                NSLog("[❌ CHECKPOINT 6] SILENT OUTPUT - Final stage failed");
            }
        }
        frameSyncCoordinator->markStageComplete(SyncRole::AudioOutput, frameIndex);
        frameSyncCoordinator->resetFrame(frameIndex);
    } else {
        memset(out, 0, inNumberFrames * sizeof(float) * 2);
        static int silentOutputCounter = 0;
        if (++silentOutputCounter % 1000 == 0) {
            NSLog("[⚠️ CHECKPOINT 6] No audio frame available from GPU processing");
        }
    }
    
    return noErr;
}


//==============================================================================
// Bridge Lifecycle and Control Functions
//==============================================================================

void init(CoreAudioGPUBridge* bridge) {
    NSLog(@"[CoreAudio→Metal] Initializing Core Audio bridge...");
    
    // Initialize MetalBridge
    MetalBridge::getInstance().initialize();
    
    // Enumerate available devices
    enumerateDevices(bridge);
    
    // Set default devices
    if (!bridge->availableInputDevices.empty()) {
        bridge->selectedInputDevice = bridge->availableInputDevices[0];
        NSLog(@"[🎯] Default input device: %s (ID: %u)", 
              bridge->inputDeviceNames[0].c_str(), bridge->selectedInputDevice);
    }
    
    if (!bridge->availableOutputDevices.empty()) {
        bridge->selectedOutputDevice = bridge->availableOutputDevices[0];
        NSLog(@"[🔊] Default output device: %s (ID: %u)", 
              bridge->outputDeviceNames[0].c_str(), bridge->selectedOutputDevice);
    }
    
    NSLog(@"[CoreAudio→Metal] Bridge initialized successfully");
}

void startCapture(CoreAudioGPUBridge* bridge) {
    if (!bridge || bridge->isCapturing) return;
    
    if (startAUHALInputOutput(bridge, bridge->selectedInputDevice, bridge->selectedOutputDevice)) {
        bridge->isCapturing = true;
        bridge->runGpuThread = true;
        bridge->gpuProcessingThread = std::thread(gpuProcessingLoop, bridge);
        NSLog(@"[CoreAudio→Metal] Capture started.");
    } else {
        NSLog(@"[CoreAudio→Metal] FAILED to start capture.");
    }
}

void stopCapture(CoreAudioGPUBridge* bridge) {
    if (!bridge || !bridge->isCapturing) return;
    
    stopAUHALInputOutput(bridge);
    bridge->isCapturing = false;

    bridge->runGpuThread = false;
    if (bridge->gpuProcessingThread.joinable()) {
        bridge->gpuProcessingThread.join();
    }
    NSLog(@"[CoreAudio→Metal] Capture stopped.");
}

void enumerateDevices(CoreAudioGPUBridge* bridge) {
    if (!bridge) return;
    
    bridge->availableInputDevices.clear();
    bridge->availableOutputDevices.clear();
    bridge->inputDeviceNames.clear();
    bridge->outputDeviceNames.clear();
    
    AudioObjectPropertyAddress propertyAddress = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    UInt32 dataSize = 0;
    OSStatus status = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize);
    if (status != noErr) return;
    
    UInt32 deviceCount = dataSize / sizeof(AudioDeviceID);
    std::vector<AudioDeviceID> devices(deviceCount);
    status = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize, devices.data());
    if (status != noErr) return;
    
    for (AudioDeviceID deviceID : devices) {
        categorizeDevice(bridge, deviceID);
    }
    
    NSLog(@"[CoreAudio→Metal] Found %lu input devices, %lu output devices",
          bridge->availableInputDevices.size(), bridge->availableOutputDevices.size());
}

static void categorizeDevice(CoreAudioGPUBridge* bridge, AudioDeviceID deviceID) {
    AudioObjectPropertyAddress propertyAddress;
    propertyAddress.mSelector = kAudioDevicePropertyStreamConfiguration;
    propertyAddress.mScope = kAudioDevicePropertyScopeInput;
    propertyAddress.mElement = kAudioObjectPropertyElementMaster;
    
    UInt32 dataSize = 0;
    OSStatus result = AudioObjectGetPropertyDataSize(deviceID, &propertyAddress, 0, NULL, &dataSize);
    
    if (result == noErr && dataSize > 0) {
        AudioBufferList* bufferList = (AudioBufferList*)malloc(dataSize);
        result = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, NULL, &dataSize, bufferList);
        
        if (result == noErr && bufferList->mNumberBuffers > 0) {
            bridge->availableInputDevices.push_back(deviceID);
            bridge->inputDeviceNames.push_back(getDeviceName(deviceID));
        }
        free(bufferList);
    }
    
    propertyAddress.mScope = kAudioDevicePropertyScopeOutput;
    result = AudioObjectGetPropertyDataSize(deviceID, &propertyAddress, 0, NULL, &dataSize);
    
    if (result == noErr && dataSize > 0) {
        AudioBufferList* bufferList = (AudioBufferList*)malloc(dataSize);
        result = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, NULL, &dataSize, bufferList);
        
        if (result == noErr && bufferList->mNumberBuffers > 0) {
            bridge->availableOutputDevices.push_back(deviceID);
            bridge->outputDeviceNames.push_back(getDeviceName(deviceID));
        }
        free(bufferList);
    }
}

static std::string getDeviceName(AudioDeviceID deviceID) {
    CFStringRef deviceName = NULL;
    UInt32 dataSize = sizeof(deviceName);
    AudioObjectPropertyAddress propertyAddress = {
        kAudioDevicePropertyDeviceNameCFString,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    OSStatus result = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, NULL, &dataSize, &deviceName);
    if (result == noErr && deviceName != NULL) {
        char name[256];
        CFStringGetCString(deviceName, name, sizeof(name), kCFStringEncodingUTF8);
        CFRelease(deviceName);
        return std::string(name);
    }
    return "Unknown Device";
}

bool startAUHALInputOutput(CoreAudioGPUBridge* bridge, AudioDeviceID inputDev, AudioDeviceID outputDev) {
    OSStatus err = noErr;
    UInt32 enable = 1;
    UInt32 disable = 0;
    AudioStreamBasicDescription fmt = {};
    AURenderCallbackStruct inputCB = {};
    AURenderCallbackStruct outputCB = {};
    AudioStreamBasicDescription hwASBD = {};
    AudioStreamBasicDescription desired = {};
    AudioStreamBasicDescription actual = {};
    UInt32 propSize = 0;
    UInt32 maxFrames = 0;
    
    // Validate devices first
    if (inputDev == 0 || outputDev == 0) {
        NSLog(@"[❌] Invalid device IDs: input=%u, output=%u", inputDev, outputDev);
        return false;
    }
    
    // Check if devices are alive
    AudioObjectPropertyAddress aliveAddress = {
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    UInt32 alive = 0;
    UInt32 dataSize = sizeof(alive);
    OSStatus status = AudioObjectGetPropertyData(inputDev, &aliveAddress, 0, NULL, &dataSize, &alive);
    if (status != noErr || !alive) {
        NSLog(@"[❌] Input device %u is not alive", inputDev);
        return false;
    }
    
    status = AudioObjectGetPropertyData(outputDev, &aliveAddress, 0, NULL, &dataSize, &alive);
    if (status != noErr || !alive) {
        NSLog(@"[❌] Output device %u is not alive", outputDev);
        return false;
    }
    
    bridge->selectedInputDevice = inputDev;
    bridge->selectedOutputDevice = outputDev;
    
    // Clean shutdown first
    stopAUHALInputOutput(bridge);
    
    AudioComponentDescription desc = {};
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_HALOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    
    AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
    if (!comp) {
        NSLog(@"[❌] Could not find HAL AudioUnit component");
        return false;
    }

    err = AudioComponentInstanceNew(comp, &bridge->inputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to create input AudioUnit: %d", err);
        return false;
    }
    
    err = AudioComponentInstanceNew(comp, &bridge->outputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to create output AudioUnit: %d", err);
        AudioComponentInstanceDispose(bridge->inputUnit);
        bridge->inputUnit = nullptr;
        return false;
    }
    
    // Configure input unit
    err = AudioUnitSetProperty(bridge->inputUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enable, sizeof(enable));
    if (err != noErr) {
        NSLog(@"[❌] Failed to enable input on input unit: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitSetProperty(bridge->inputUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &disable, sizeof(disable));
    if (err != noErr) {
        NSLog(@"[❌] Failed to disable output on input unit: %d", err);
        goto cleanup;
    }
    
    // Configure output unit
    err = AudioUnitSetProperty(bridge->outputUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enable, sizeof(enable));
    if (err != noErr) {
        NSLog(@"[❌] Failed to enable output on output unit: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitSetProperty(bridge->outputUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &disable, sizeof(disable));
    if (err != noErr) {
        NSLog(@"[❌] Failed to disable input on output unit: %d", err);
        goto cleanup;
    }
    
    // Set devices
    err = AudioUnitSetProperty(bridge->inputUnit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &inputDev, sizeof(inputDev));
    if (err != noErr) {
        NSLog(@"[❌] Failed to set input device on input unit: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitSetProperty(bridge->outputUnit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &outputDev, sizeof(outputDev));
    if (err != noErr) {
        NSLog(@"[❌] Failed to set output device on output unit: %d", err);
        goto cleanup;
    }
    
    // ---- BEGIN STREAM FORMAT NEGOTIATION FIX ----
    // 1️⃣ Query hardware default format on input scope (bus 1)
    propSize = sizeof(hwASBD);
    err = AudioUnitGetProperty(bridge->inputUnit,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Input, 1,
        &hwASBD, &propSize);
    if (err != noErr) {
        NSLog(@"[❌] Error querying hardware format: %d", err);
        goto cleanup;
    }
    
    NSLog(@"[🔍] Hardware format: %.0f Hz, %u channels, format ID: %u", 
          hwASBD.mSampleRate, hwASBD.mChannelsPerFrame, hwASBD.mFormatID);
    
    // 2️⃣ Build desired ASBD matching hardware
    desired.mSampleRate       = hwASBD.mSampleRate;
    desired.mFormatID         = kAudioFormatLinearPCM;
    desired.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    desired.mChannelsPerFrame = hwASBD.mChannelsPerFrame;
    desired.mFramesPerPacket  = 1;
    desired.mBitsPerChannel   = sizeof(float) * 8;
    desired.mBytesPerFrame    = desired.mChannelsPerFrame * sizeof(float);
    desired.mBytesPerPacket   = desired.mBytesPerFrame * desired.mFramesPerPacket;
    
    NSLog(@"[🎯] Setting desired format: %.0f Hz, %u channels, %u bits", 
          desired.mSampleRate, desired.mChannelsPerFrame, desired.mBitsPerChannel);
    
    // 3️⃣ Apply format to both input (bus 1) and output (bus 0) scopes
    err = AudioUnitSetProperty(bridge->inputUnit,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Input, 1,
        &desired, sizeof(desired));
    if (err != noErr) {
        NSLog(@"[❌] Error setting input stream format: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitSetProperty(bridge->inputUnit,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Output, 1,
        &desired, sizeof(desired));
    if (err != noErr) {
        NSLog(@"[❌] Error setting input unit output format: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitSetProperty(bridge->outputUnit,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Input, 0,
        &desired, sizeof(desired));
    if (err != noErr) {
        NSLog(@"[❌] Error setting output stream format: %d", err);
        goto cleanup;
    }
    
    // 4️⃣ Confirm the actual format matches expectations
    propSize = sizeof(actual);
    AudioUnitGetProperty(bridge->inputUnit,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Input, 1,
        &actual, &propSize);
    
    NSLog(@"[✅] Actual input format: %.0f Hz, %u channels, format ID: %u", 
          actual.mSampleRate, actual.mChannelsPerFrame, actual.mFormatID);
    
    if (actual.mSampleRate != desired.mSampleRate || 
        actual.mChannelsPerFrame != desired.mChannelsPerFrame) {
        NSLog(@"[⚠️] Format mismatch detected!");
    }
    
    // 5️⃣ Retrieve MaximumFramesPerSlice for buffer sizing
    propSize = sizeof(maxFrames);
    AudioUnitGetProperty(bridge->inputUnit,
        kAudioUnitProperty_MaximumFramesPerSlice,
        kAudioUnitScope_Global, 0,
        &maxFrames, &propSize);
    
    NSLog(@"[📏] MaximumFramesPerSlice: %u", maxFrames);
    // ---- END STREAM FORMAT NEGOTIATION FIX ----
    
    // Set callbacks
    inputCB.inputProc = InputRenderCallback;
    inputCB.inputProcRefCon = bridge;
    err = AudioUnitSetProperty(bridge->inputUnit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &inputCB, sizeof(inputCB));
    if (err != noErr) {
        NSLog(@"[❌] Failed to set input callback: %d", err);
        goto cleanup;
    }

    outputCB.inputProc = OutputRenderCallback;
    outputCB.inputProcRefCon = bridge;
    err = AudioUnitSetProperty(bridge->outputUnit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &outputCB, sizeof(outputCB));
    if (err != noErr) {
        NSLog(@"[❌] Failed to set output callback: %d", err);
        goto cleanup;
    }

    // Initialize units
    err = AudioUnitInitialize(bridge->inputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to initialize input unit: %d", err);
        goto cleanup;
    }
    
    err = AudioUnitInitialize(bridge->outputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to initialize output unit: %d", err);
        AudioUnitUninitialize(bridge->inputUnit);
        goto cleanup;
    }
    
    // Start units
    err = AudioOutputUnitStart(bridge->inputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to start input unit: %d", err);
        AudioUnitUninitialize(bridge->inputUnit);
        AudioUnitUninitialize(bridge->outputUnit);
        goto cleanup;
    }

    err = AudioOutputUnitStart(bridge->outputUnit);
    if (err != noErr) {
        NSLog(@"[❌] Failed to start output unit: %d", err);
        AudioOutputUnitStop(bridge->inputUnit);
        AudioUnitUninitialize(bridge->inputUnit);
        AudioUnitUninitialize(bridge->outputUnit);
        goto cleanup;
    }

    NSLog(@"[✅] AudioUnits started successfully with input device %u, output device %u", inputDev, outputDev);
    return true;

cleanup:
    if (bridge->inputUnit) {
        AudioComponentInstanceDispose(bridge->inputUnit);
        bridge->inputUnit = nullptr;
    }
    if (bridge->outputUnit) {
        AudioComponentInstanceDispose(bridge->outputUnit);
        bridge->outputUnit = nullptr;
    }
    return false;
}

void stopAUHALInputOutput(CoreAudioGPUBridge* bridge) {
    if (bridge->inputUnit) {
        AudioOutputUnitStop(bridge->inputUnit);
        AudioUnitUninitialize(bridge->inputUnit);
        AudioComponentInstanceDispose(bridge->inputUnit);
        bridge->inputUnit = nullptr;
    }
    if (bridge->outputUnit) {
        AudioOutputUnitStop(bridge->outputUnit);
        AudioUnitUninitialize(bridge->outputUnit);
        AudioComponentInstanceDispose(bridge->outputUnit);
        bridge->outputUnit = nullptr;
    }
}

void setInputDevice(CoreAudioGPUBridge* bridge, int deviceIndex) {
    if (deviceIndex < 0 || deviceIndex >= bridge->availableInputDevices.size()) {
        NSLog(@"[❌] Invalid input device index: %d", deviceIndex);
        return;
    }
    
    AudioDeviceID newDevice = bridge->availableInputDevices[deviceIndex];
    
    // Check if device is already selected
    if (bridge->selectedInputDevice == newDevice) {
        NSLog(@"[ℹ️] Input device %u already selected", newDevice);
        return;
    }
    
    // Validate device is alive before switching
    AudioObjectPropertyAddress aliveAddress = {
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    UInt32 alive = 0;
    UInt32 dataSize = sizeof(alive);
    OSStatus status = AudioObjectGetPropertyData(newDevice, &aliveAddress, 0, NULL, &dataSize, &alive);
    if (status != noErr || !alive) {
        NSLog(@"[❌] Input device %u is not alive, cannot select", newDevice);
        return;
    }
    
    NSLog(@"🎯 Selecting input device: %s (ID: %u)", 
          bridge->inputDeviceNames[deviceIndex].c_str(), newDevice);
    
    // Check device properties
    AudioObjectPropertyAddress runningAddress = {
        kAudioDevicePropertyDeviceIsRunning,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    UInt32 running = 0;
    dataSize = sizeof(running);
    status = AudioObjectGetPropertyData(newDevice, &runningAddress, 0, NULL, &dataSize, &running);
    NSLog(@"🎙️ Device IsAlive: %u", alive);
    NSLog(@"🚦 Device IsRunning: %u", running);
    
    // Store the new device
    bridge->selectedInputDevice = newDevice;
    
    // Only restart if currently capturing
    if (bridge->isCapturing) {
        NSLog(@"🔄 AudioUnit restart required for device change");
        bool wasCapturing = bridge->isCapturing;
        stopCapture(bridge);
        if (wasCapturing) {
            startCapture(bridge);
        }
    } else {
        NSLog(@"[ℹ️] Device selected, will be used on next capture start");
    }
}

void setOutputDevice(CoreAudioGPUBridge* bridge, int deviceIndex) {
    if (deviceIndex < 0 || deviceIndex >= bridge->availableOutputDevices.size()) {
        NSLog(@"[❌] Invalid output device index: %d", deviceIndex);
        return;
    }
    
    bridge->selectedOutputDevice = bridge->availableOutputDevices[deviceIndex];
    NSLog(@"🔊 Output device set to: %s (ID: %u)", 
          bridge->outputDeviceNames[deviceIndex].c_str(), bridge->selectedOutputDevice);
    
    // Restart capture if currently running
    if (bridge->isCapturing) {
        stopCapture(bridge);
        startCapture(bridge);
    }
}

void setRecordArmStates(CoreAudioGPUBridge* bridge, bool jellieArmed, bool pnbtrArmed) {
    bridge->jellieRecordArmed = jellieArmed;
    bridge->pnbtrRecordArmed = pnbtrArmed;
    
    NSLog(@"[RECORD ARM] JELLIE: %s, PNBTR: %s", 
          jellieArmed ? "ARMED" : "DISARMED", 
          pnbtrArmed ? "ARMED" : "DISARMED");
}


//==============================================================================
// C-Interface for JUCE
//==============================================================================
extern "C" {
    void initializeCoreAudioBridge() {
        if (!globalBridge) {
            globalBridge = new CoreAudioGPUBridge();
            init(globalBridge);
        }
    }

    void startCoreAudioCapture() {
        if (globalBridge) startCapture(globalBridge);
    }

    void stopCoreAudioCapture() {
        if (globalBridge) stopCapture(globalBridge);
    }

    void setCoreAudioRecordArmStates(bool jellieArmed, bool pnbtrArmed) {
        if (globalBridge) {
            globalBridge->jellieRecordArmed = jellieArmed;
            globalBridge->pnbtrRecordArmed = pnbtrArmed;
        }
    }
    
    int getCoreAudioInputDeviceCount() {
        if (globalBridge) {
            return (int)globalBridge->availableInputDevices.size();
        }
        return 0;
    }
    
    int getCoreAudioOutputDeviceCount() {
        if (globalBridge) {
            return (int)globalBridge->availableOutputDevices.size();
        }
        return 0;
    }
    
    const char* getCoreAudioInputDeviceName(int index) {
        if (globalBridge && index >= 0 && index < globalBridge->inputDeviceNames.size()) {
            return globalBridge->inputDeviceNames[index].c_str();
        }
        return "Unknown";
    }
    
    const char* getCoreAudioOutputDeviceName(int index) {
        if (globalBridge && index >= 0 && index < globalBridge->outputDeviceNames.size()) {
            return globalBridge->outputDeviceNames[index].c_str();
        }
        return "Unknown";
    }
    
    bool isCoreAudioCapturing() {
        if (globalBridge) {
            return globalBridge->isCapturing;
        }
        return false;
    }
    
    void shutdownCoreAudioBridge() {
        if (globalBridge) {
            stopCapture(globalBridge);
            delete globalBridge;
            globalBridge = nullptr;
            NSLog(@"[CoreAudio→Metal] Bridge shutdown completed");
        }
    }
    
    // Legacy C interface functions for MainComponent compatibility
    void* createCoreAudioGPUBridge() {
        initializeCoreAudioBridge();
        return (void*)globalBridge;
    }
    
    void destroyCoreAudioGPUBridge() {
        shutdownCoreAudioBridge();
    }
    
    void checkMetalBridgeStatus() {
        try {
            auto& metalBridge = MetalBridge::getInstance();
            NSLog(@"[✅] MetalBridge instance: AVAILABLE");
            NSLog(@"[🔧] MetalBridge appears to be initialized");
            
            // Additional diagnostics from comprehensive guide
            if (metalBridge.isInitialized()) {
                NSLog(@"[✅] MetalBridge: Fully initialized and ready");
            } else {
                NSLog(@"[⚠️] MetalBridge: Not fully initialized");
            }
            
        } catch (const std::exception& e) {
            NSLog(@"[❌] MetalBridge ERROR: %s", e.what());
        } catch (...) {
            NSLog(@"[❌] MetalBridge: Unknown error or not initialized");
        }
    }
    
    void diagnoseCoreAudioError(OSStatus error, const char* operation) {
        NSLog(@"[🔍 CORE AUDIO DIAGNOSTIC] Operation: %s, Error: %d", operation, error);
        
        switch (error) {
            case -10851:
                NSLog(@"[📋 ERROR -10851] kAudioUnitErr_InvalidProperty - Device selection issue");
                NSLog(@"[💡 FIX] Check device is alive and supports required format");
                break;
            case -10867:
                NSLog(@"[📋 ERROR -10867] kAudioUnitErr_CannotDoInCurrentContext - Stream format issue");
                NSLog(@"[💡 FIX] Ensure matching sample rates and channel counts");
                break;
            case -50:
                NSLog(@"[📋 ERROR -50] paramErr - Invalid parameter");
                NSLog(@"[💡 FIX] Check buffer sizes and format specifications");
                break;
            case -10863:
                NSLog(@"[📋 ERROR -10863] kAudioUnitErr_FormatNotSupported");
                NSLog(@"[💡 FIX] Use supported format (44.1/48kHz, 16/24-bit)");
                break;
            case -10868:
                NSLog(@"[📋 ERROR -10868] kAudioUnitErr_InvalidScope");
                NSLog(@"[💡 FIX] Check input/output scope configuration");
                break;
            default:
                NSLog(@"[📋 ERROR %d] Unknown Core Audio error", error);
                break;
        }
    }
    
    void enableCoreAudioSineTest(bool enable) {
        NSLog(@"[🔧] SINE TEST: %s", enable ? "ENABLED" : "DISABLED");
        NSLog(@"[ℹ️] Note: Sine test mode not implemented in dual-AudioUnit version");
        NSLog(@"[💡] Use direct audio input testing instead");
    }
    
    void forceCoreAudioCallback() {
        NSLog(@"[🔧] FORCE CALLBACK: Checking audio processing state...");
        
        if (!globalBridge) {
            NSLog(@"[❌] CoreAudio bridge not initialized");
            return;
        }
        
        if (!globalBridge->isCapturing) {
            NSLog(@"[⚠️] Audio capture not running - starting capture...");
            startCapture(globalBridge);
        }
        
        bool jellieArmed = globalBridge->jellieRecordArmed;
        bool pnbtrArmed = globalBridge->pnbtrRecordArmed;
        
        NSLog(@"[🎯] Current record arm states: JELLIE=%s, PNBTR=%s",
              jellieArmed ? "ARMED" : "DISARMED",
              pnbtrArmed ? "ARMED" : "DISARMED");
        
        if (!jellieArmed && !pnbtrArmed) {
            NSLog(@"[⚠️] No tracks are record-armed - enabling both for testing");
            setCoreAudioRecordArmStates(true, true);
        }
        
        NSLog(@"[💡] Look for '[🔁 INPUT CALLBACK' messages in console");
    }
    
    void useDefaultInputDevice() {
        if (globalBridge) {
            printf("[🎯 DEFAULT INPUT] Using default input device\n");
            setInputDevice(globalBridge, 0);
        }
    }

    void setCoreAudioInputDevice(int deviceIndex) {
        if (globalBridge) {
            setInputDevice(globalBridge, deviceIndex);
        }
    }

    void setCoreAudioOutputDevice(int deviceIndex) {
        if (globalBridge) {
            setOutputDevice(globalBridge, deviceIndex);
        }
    }
}

// If compiling as C++, ensure Apple types are available:
#ifndef UInt32
typedef unsigned int UInt32;
#endif
#ifndef OSStatus
typedef int OSStatus;
#endif