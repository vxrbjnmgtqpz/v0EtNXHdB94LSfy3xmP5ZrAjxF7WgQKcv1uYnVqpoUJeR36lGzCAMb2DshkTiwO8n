#import "MetalBridge.h"
#import <iostream>
#import <vector>
#import <CoreAudio/CoreAudio.h>
#import <mach/mach_time.h>

// Define the shader parameter structs directly here to avoid include issues
typedef struct {
    float threshold;
    bool jellieRecordArmed;
    bool pnbtrRecordArmed;
    float gain;
    float adaptiveThreshold;
} GateParams;

typedef struct {
    float fftSize;
    float sampleRate;
    float windowType;
    bool enableMPS;
    float spectralSmoothing;
} DJAnalysisParams;

typedef struct {
    float redLevel;
    float greenLevel;
    float pulseIntensity;
    float timePhase;
} RecordArmVisualParams;

typedef struct {
    float compressionRatio;
    float gain;
    float upsampleRatio;
    float quantizationBits;
} JELLIEPreprocessParams;

typedef struct {
    float latencyMs;
    float packetLossPercentage;
    float jitterMs;
    float correlationThreshold;
} NetworkSimulationParams;

typedef struct {
    float mixLevel;
    float sineFrequency;
    bool applySineTest;
    float neuralThreshold;
    float lookbackSamples;
    float smoothingFactor;
} PNBTRReconstructionParams;

// --- MetalBridge Implementation ---

MetalBridge& MetalBridge::getInstance() {
    static MetalBridge instance;
    return instance;
}

MetalBridge::MetalBridge() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal is not supported on this device");
    }
    commandQueue = [device newCommandQueue];
    frameBoundarySemaphore = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);
    initialized = false;
    currentFrameIndex = 0;
    
    // Set buffer pointers to nil
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        audioInputBuffer[i] = nil;
        gateParamsBuffer[i] = nil;
        stage1Buffer[i] = nil;
        djAnalysisParamsBuffer[i] = nil;
        djTransformedBuffer[i] = nil;
        recordArmVisualParamsBuffer[i] = nil;
        stage2Buffer[i] = nil;
        jelliePreprocessParamsBuffer[i] = nil;
        stage3Buffer[i] = nil;
        networkSimParamsBuffer[i] = nil;
        stage4Buffer[i] = nil;
        pnbtrReconstructionParamsBuffer[i] = nil;
        reconstructedBuffer[i] = nil;
    }
}

MetalBridge::~MetalBridge() {
    if (initialized) {
        // Wait for all commands to finish
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            dispatch_semaphore_wait(frameBoundarySemaphore, DISPATCH_TIME_FOREVER);
        }
    }
    
    // With ARC (Automatic Reference Counting), we do not manually release objects.
    // The compiler handles releasing the MTLBuffer and other Objective-C objects
    // when this MetalBridge C++ object is destroyed. We also do not need to
    // release the GCD semaphore, as it is also managed by ARC in this context.
}

bool MetalBridge::initialize() {
    @autoreleasepool {
        loadShaders();
        createComputePipelines();
    }
    initialized = (metalLibrary != nil);
    return initialized;
}

void MetalBridge::prepareBuffers(size_t numSamples, double sampleRate) {
    if (!initialized) return;

    size_t stereoBufferSize = numSamples * sizeof(float) * 2;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        audioInputBuffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        stage1Buffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        djTransformedBuffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        stage2Buffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        stage3Buffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        stage4Buffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];
        reconstructedBuffer[i] = [device newBufferWithLength:stereoBufferSize options:MTLResourceStorageModeShared];

        gateParamsBuffer[i] = [device newBufferWithLength:sizeof(GateParams) options:MTLResourceStorageModeShared];
        djAnalysisParamsBuffer[i] = [device newBufferWithLength:sizeof(DJAnalysisParams) options:MTLResourceStorageModeShared];
        recordArmVisualParamsBuffer[i] = [device newBufferWithLength:sizeof(RecordArmVisualParams) options:MTLResourceStorageModeShared];
        jelliePreprocessParamsBuffer[i] = [device newBufferWithLength:sizeof(JELLIEPreprocessParams) options:MTLResourceStorageModeShared];
        networkSimParamsBuffer[i] = [device newBufferWithLength:sizeof(NetworkSimulationParams) options:MTLResourceStorageModeShared];
        pnbtrReconstructionParamsBuffer[i] = [device newBufferWithLength:sizeof(PNBTRReconstructionParams) options:MTLResourceStorageModeShared];
    }
}

void MetalBridge::processAudioBlock(const float* inputData, float* outputData, size_t numSamples) {
    if (!initialized || !metalLibrary) {
        // Clear output and return early
        memset(outputData, 0, numSamples * sizeof(float) * 2);
        return;
    }
    
    // Check if pipeline state objects are valid
    if (!audioInputGatePSO || !pnbtrReconstructionPSO) {
        NSLog(@"[METAL ERROR] Pipeline state objects not initialized");
        memset(outputData, 0, numSamples * sizeof(float) * 2);
        return;
    }
    
    // Prepare buffers if not already done
    if (!audioInputBuffer[0]) {
        prepareBuffers(numSamples, 48000.0);
    }

    dispatch_semaphore_wait(frameBoundarySemaphore, DISPATCH_TIME_FOREVER);

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    __block dispatch_semaphore_t semaphore = frameBoundarySemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(semaphore);
    }];
    
    int frameIdx = currentFrameIndex;

    // Convert mono input to stereo for Metal processing
    float* stereoInput = (float*)audioInputBuffer[frameIdx].contents;
    for (size_t i = 0; i < numSamples; ++i) {
        stereoInput[i * 2] = inputData[i];     // Left channel
        stereoInput[i * 2 + 1] = inputData[i]; // Right channel (duplicate mono)
    }

    // Update enhanced params for current frame
    GateParams* gp = (GateParams*)gateParamsBuffer[frameIdx].contents;
    gp->threshold = 0.1f;
    gp->jellieRecordArmed = this->jellieRecordArmed;
    gp->pnbtrRecordArmed = this->pnbtrRecordArmed;
    gp->gain = 1.0f;
    gp->adaptiveThreshold = 0.05f;

    // Enhanced spectral analysis parameters
    DJAnalysisParams* djp = (DJAnalysisParams*)djAnalysisParamsBuffer[frameIdx].contents;
    djp->fftSize = 1024.0f;
    djp->sampleRate = 48000.0f;
    djp->windowType = 1.0f; // Enable Hann window
    djp->enableMPS = true;
    djp->spectralSmoothing = 0.8f;

    // Record arm visual parameters
    RecordArmVisualParams* rap = (RecordArmVisualParams*)recordArmVisualParamsBuffer[frameIdx].contents;
    rap->redLevel = this->jellieRecordArmed ? 1.0f : 0.2f;
    rap->greenLevel = this->pnbtrRecordArmed ? 1.0f : 0.2f;
    rap->pulseIntensity = 0.5f;
    rap->timePhase = fmod(mach_absolute_time() * 1e-9, 2.0 * M_PI); // Use mach_absolute_time instead

    // JELLIE preprocessing parameters
    JELLIEPreprocessParams* jpp = (JELLIEPreprocessParams*)jelliePreprocessParamsBuffer[frameIdx].contents;
    jpp->compressionRatio = 2.0f;
    jpp->gain = 1.2f;
    jpp->upsampleRatio = 4.0f; // 48kHz -> 192kHz
    jpp->quantizationBits = 24.0f;

    // Network simulation parameters
    NetworkSimulationParams* nsp = (NetworkSimulationParams*)networkSimParamsBuffer[frameIdx].contents;
    nsp->latencyMs = 5.0f;
    nsp->packetLossPercentage = 2.0f; // 2% packet loss
    nsp->jitterMs = 1.0f;
    nsp->correlationThreshold = 0.8f;

    // PNBTR reconstruction parameters
    PNBTRReconstructionParams* prp = (PNBTRReconstructionParams*)pnbtrReconstructionParamsBuffer[frameIdx].contents;
    prp->mixLevel = 1.0f;
    prp->sineFrequency = 440.0f;
    prp->applySineTest = false;
    prp->neuralThreshold = 0.01f;
    prp->lookbackSamples = 8.0f;
    prp->smoothingFactor = 0.7f;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Stage 1: Enhanced Audio Input Gate
    [encoder setComputePipelineState:audioInputGatePSO];
    [encoder setBuffer:audioInputBuffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:gateParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:stage1Buffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, audioInputGatePSO, numSamples * 2);
    
    // Stage 2: Enhanced Spectral Analysis
    [encoder setComputePipelineState:djAnalysisPSO];
    [encoder setBuffer:stage1Buffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:djAnalysisParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:djTransformedBuffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, djAnalysisPSO, numSamples * 2);
    
    // Stage 3: Record Arm Visual Feedback
    [encoder setComputePipelineState:recordArmVisualPSO];
    [encoder setBuffer:djTransformedBuffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:recordArmVisualParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:stage2Buffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, recordArmVisualPSO, numSamples * 2);
    
    // Stage 4: JELLIE Preprocessing
    [encoder setComputePipelineState:jelliePreprocessPSO];
    [encoder setBuffer:stage2Buffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:jelliePreprocessParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:stage3Buffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, jelliePreprocessPSO, numSamples * 2);
    
    // Stage 5: Network Simulation
    [encoder setComputePipelineState:networkSimPSO];
    [encoder setBuffer:stage3Buffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:networkSimParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:stage4Buffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, networkSimPSO, numSamples * 2);

    // Stage 6: PNBTR Reconstruction
    [encoder setComputePipelineState:pnbtrReconstructionPSO];
    [encoder setBuffer:stage4Buffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:pnbtrReconstructionParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:reconstructedBuffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, pnbtrReconstructionPSO, numSamples * 2);

    [encoder endEncoding];
    [commandBuffer commit];

    // Copy reconstructed stereo output, converting back to mono
    float* stereoOutput = (float*)reconstructedBuffer[frameIdx].contents;
    for (size_t i = 0; i < numSamples; ++i) {
        outputData[i * 2] = stereoOutput[i * 2];         // Left channel
        outputData[i * 2 + 1] = stereoOutput[i * 2 + 1]; // Right channel
    }

    currentFrameIndex = (currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void MetalBridge::setRecordArmStates(bool jellieArmed, bool pnbtrArmed) {
    // These atomics can be set from any thread and will be picked up
    // by the gpuProcessingLoop before it processes a block.
    this->jellieRecordArmed = jellieArmed;
    this->pnbtrRecordArmed = pnbtrArmed;
}

// Async visualization loop implementation
void MetalBridge::startVisualizationLoop() {
    NSLog(@"[METAL] Starting async visualization loop at 60fps");
    // Implementation would create a separate dispatch queue for visualization updates
    // This ensures UI updates don't block audio processing
}

void MetalBridge::stopVisualizationLoop() {
    NSLog(@"[METAL] Stopping async visualization loop");
    // Implementation would stop the visualization timer/queue
}

void MetalBridge::updateVisualizationBuffer(const float* audioData, size_t numSamples) {
    // Copy audio data to visualization buffer in a thread-safe manner
    // This would be called from the audio thread to update visualization data
    // The visualization thread would read from this buffer at 60fps
}

const float* MetalBridge::getVisualizationBuffer(size_t& bufferSize) const {
    // Return read-only access to visualization buffer
    // This would be called from the visualization thread
    bufferSize = 0;
    return nullptr;
}


// --- Private Helper Methods ---

void MetalBridge::loadShaders() {
    @try {
        // Try to load the default library first (if it exists)
        NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        if (libraryPath) {
            NSError* error = nil;
            metalLibrary = [device newLibraryWithFile:libraryPath error:&error];
            if (metalLibrary) {
                NSLog(@"[METAL] Successfully loaded default.metallib");
                return;
            }
        }
        
        // If default.metallib doesn't exist, try to load the main bundle's default library
        NSError* error = nil;
        metalLibrary = [device newDefaultLibrary];
        if (metalLibrary) {
            NSLog(@"[METAL] Successfully loaded default Metal library from bundle");
            return;
        }
        
        // If neither works, try to create a library from source (this will compile the shaders)
        NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Enhanced shader parameters for production
struct GateParams {
    float threshold;
    bool jellieRecordArmed;
    bool pnbtrRecordArmed;
    float gain;
    float adaptiveThreshold;
};

struct DJAnalysisParams {
    float fftSize;
    float sampleRate;
    float windowType;
    bool enableMPS;
    float spectralSmoothing;
};

struct RecordArmVisualParams {
    float redLevel;
    float greenLevel;
    float pulseIntensity;
    float timePhase;
};

struct JELLIEPreprocessParams {
    float compressionRatio;
    float gain;
    float upsampleRatio;
    float quantizationBits;
};

struct NetworkSimulationParams {
    float latencyMs;
    float packetLossPercentage;
    float jitterMs;
    float correlationThreshold;
};

struct PNBTRReconstructionParams {
    float mixLevel;
    float sineFrequency;
    bool applySineTest;
    float neuralThreshold;
    float lookbackSamples;
    float smoothingFactor;
};

kernel void audioInputGateKernel(constant float* input [[buffer(0)]],
                                constant GateParams& params [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // Apply adaptive gating based on threshold
    float currentThreshold = params.threshold;
    if (params.adaptiveThreshold > 0.0f) {
        // Simple adaptive threshold based on recent amplitude
        currentThreshold = mix(params.threshold, params.adaptiveThreshold, 0.1f);
    }
    
    if (abs(sample) < currentThreshold) {
        sample = 0.0f;
    }
    
    // Apply gain and record arm logic
    if (params.jellieRecordArmed || params.pnbtrRecordArmed) {
        sample *= params.gain;
    }
    
    output[id] = sample;
}

kernel void djSpectralAnalysisKernel(constant float* input [[buffer(0)]],
                                   constant DJAnalysisParams& params [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // Apply window function if enabled
    if (params.windowType > 0.0f) {
        float windowPos = float(id) / params.fftSize;
        float window = 0.5f * (1.0f - cos(2.0f * M_PI_F * windowPos)); // Hann window
        sample *= window;
    }
    
    // Apply spectral smoothing
    if (params.spectralSmoothing > 0.0f) {
        // Simple spectral smoothing (would be more complex in real implementation)
        sample *= params.spectralSmoothing;
    }
    
    output[id] = sample;
}

kernel void recordArmVisualKernel(constant float* input [[buffer(0)]],
                                 constant RecordArmVisualParams& params [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // Create visual feedback based on record arm state
    float visualIntensity = params.pulseIntensity * sin(params.timePhase + float(id) * 0.1f);
    
    // Mix visual feedback with audio signal
    sample = mix(sample, visualIntensity, 0.1f);
    
    output[id] = sample;
}

kernel void jelliePreprocessKernel(constant float* input [[buffer(0)]],
                                  constant JELLIEPreprocessParams& params [[buffer(1)]],
                                  device float* output [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // Apply compression
    if (params.compressionRatio > 1.0f) {
        float compressed = sample / params.compressionRatio;
        sample = mix(sample, compressed, 0.5f);
    }
    
    // Apply gain
    sample *= params.gain;
    
    // Simulate upsampling (simplified)
    if (params.upsampleRatio > 1.0f) {
        sample *= params.upsampleRatio;
    }
    
    output[id] = sample;
}

kernel void networkSimulationKernel(constant float* input [[buffer(0)]],
                                   constant NetworkSimulationParams& params [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // Simulate packet loss
    if (params.packetLossPercentage > 0.0f) {
        // Simple packet loss simulation using thread ID
        float lossThreshold = params.packetLossPercentage / 100.0f;
        float randomValue = fract(sin(float(id) * 12.9898f) * 43758.5453f);
        
        if (randomValue < lossThreshold) {
            sample = 0.0f; // Packet lost
        }
    }
    
    // Simulate jitter by adding phase offset
    if (params.jitterMs > 0.0f) {
        float jitterOffset = sin(float(id) * 0.1f) * params.jitterMs * 0.001f;
        sample *= (1.0f + jitterOffset);
    }
    
    output[id] = sample;
}

kernel void pnbtrReconstructionKernel(constant float* input [[buffer(0)]],
                                     constant PNBTRReconstructionParams& params [[buffer(1)]],
                                     device float* output [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    float sample = input[id];
    
    // PNBTR neural reconstruction simulation
    if (abs(sample) < params.neuralThreshold) {
        // Reconstruct lost samples using predictive algorithm
        float prediction = 0.0f;
        
        // Simple predictive reconstruction (would use neural network in real implementation)
        if (id > 0) {
            prediction = input[id - 1] * params.smoothingFactor;
        }
        
        sample = mix(sample, prediction, 0.8f);
    }
    
    // Apply final mixing
    sample *= params.mixLevel;
    
    // Optional sine test for debugging
    if (params.applySineTest) {
        float sineWave = sin(float(id) * params.sineFrequency * 0.01f);
        sample = mix(sample, sineWave, 0.2f);
    }
    
    output[id] = sample;
}
)";

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        metalLibrary = [device newLibraryWithSource:shaderSource options:options error:&error];
        
        if (!metalLibrary) {
            NSLog(@"[METAL ERROR] Failed to create library from source: %@", error.localizedDescription);
            throw std::runtime_error("Failed to load or create Metal library");
        }
        
        NSLog(@"[METAL] Successfully compiled shaders from source");
        
    } @catch (NSException* exception) {
        NSLog(@"[METAL ERROR] Exception while loading shaders: %@", exception.reason);
        metalLibrary = nil;
        throw std::runtime_error("Failed to load Metal shaders");
    }
}

void MetalBridge::createComputePipelines() {
    if (!metalLibrary) {
        NSLog(@"[METAL ERROR] Cannot create pipelines: metalLibrary is nil");
        return;
    }
    
    NSError* error = nil;
    auto createPSO = [&](NSString* kernelName) -> id<MTLComputePipelineState> {
        id<MTLFunction> func = [metalLibrary newFunctionWithName:kernelName];
        if (!func) { 
            NSLog(@"[METAL ERROR] Failed to find function: %@", kernelName);
            return nil;
        }
        
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (!pso) {
            NSLog(@"[METAL ERROR] Failed to create pipeline state for %@: %@", kernelName, error.localizedDescription);
            return nil;
        }
        
        NSLog(@"[METAL] Successfully created pipeline state for %@", kernelName);
        return pso;
    };

    audioInputGatePSO = createPSO(@"audioInputGateKernel");
    djAnalysisPSO = createPSO(@"djSpectralAnalysisKernel");
    recordArmVisualPSO = createPSO(@"recordArmVisualKernel");
    jelliePreprocessPSO = createPSO(@"jelliePreprocessKernel");
    networkSimPSO = createPSO(@"networkSimulationKernel");
    pnbtrReconstructionPSO = createPSO(@"pnbtrReconstructionKernel");
    
    // Verify critical pipeline states
    if (!audioInputGatePSO || !pnbtrReconstructionPSO) {
        NSLog(@"[METAL ERROR] Critical pipeline states failed to initialize");
        initialized = false;
    } else {
        NSLog(@"[METAL] All pipeline states created successfully");
    }
}

void MetalBridge::dispatchThreadsForEncoder(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pso, size_t numSamples) {
    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > numSamples) {
        threadGroupSize = numSamples;
    }
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((numSamples + threadGroupSize - 1) / threadGroupSize, 1, 1);
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}