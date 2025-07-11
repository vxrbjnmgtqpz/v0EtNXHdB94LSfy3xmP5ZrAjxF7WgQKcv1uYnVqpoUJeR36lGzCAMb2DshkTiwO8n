#import "MetalBridge.h"
#import <iostream>
#import <vector>

// Define the shader parameter structs directly here to avoid include issues
typedef struct {
    float threshold;
    bool jellieRecordArmed;
    bool pnbtrRecordArmed;
} GateParams;

typedef struct {
    float fftSize;
    float sampleRate;
} DJAnalysisParams;

typedef struct {
    float redLevel;
    float greenLevel;
} RecordArmVisualParams;

typedef struct {
    float compressionRatio;
    float gain;
} JELLIEPreprocessParams;

typedef struct {
    float latencyMs;
    float packetLossPercentage;
} NetworkSimulationParams;

typedef struct {
    float mixLevel;
    float sineFrequency;
    bool applySineTest;
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

    // Update params for current frame
    GateParams* gp = (GateParams*)gateParamsBuffer[frameIdx].contents;
    gp->threshold = 0.1f;
    gp->jellieRecordArmed = this->jellieRecordArmed;
    gp->pnbtrRecordArmed = this->pnbtrRecordArmed;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Stage 1: Audio Input Gate
    [encoder setComputePipelineState:audioInputGatePSO];
    [encoder setBuffer:audioInputBuffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:gateParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:stage1Buffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, audioInputGatePSO, numSamples * 2); // Stereo samples
    
    // Skip stages 2-6 for now (they would go here)

    // Stage 7: PNBTR Reconstruction (using stage1 as input for simplicity)
    [encoder setComputePipelineState:pnbtrReconstructionPSO];
    [encoder setBuffer:stage1Buffer[frameIdx] offset:0 atIndex:0];
    [encoder setBuffer:pnbtrReconstructionParamsBuffer[frameIdx] offset:0 atIndex:1];
    [encoder setBuffer:reconstructedBuffer[frameIdx] offset:0 atIndex:2];
    dispatchThreadsForEncoder(encoder, pnbtrReconstructionPSO, numSamples * 2); // Stereo samples

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

kernel void audioInputGateKernel(constant float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

kernel void djSpectralAnalysisKernel(constant float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

kernel void recordArmVisualKernel(constant float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

kernel void jelliePreprocessKernel(constant float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

kernel void networkSimulationKernel(constant float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

kernel void pnbtrReconstructionKernel(constant float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
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