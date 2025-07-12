#import "GPUDebugHelper.h"
#import <iostream>

GPUDebugHelper& GPUDebugHelper::getInstance() {
    static GPUDebugHelper instance;
    return instance;
}

#ifdef __OBJC__

bool GPUDebugHelper::initializeWithDevice(id<MTLDevice> device) {
    metalDevice = device;
    
    NSError* error = nil;
    defaultLibrary = [metalDevice newDefaultLibrary];
    if (!defaultLibrary) {
        std::cerr << "[GPUDebugHelper] Failed to load default Metal library" << std::endl;
        return false;
    }
    
    // Create validation counter buffer
    validationCounterBuffer = [metalDevice newBufferWithLength:sizeof(uint32_t) 
                                                       options:MTLResourceStorageModeShared];
    if (!validationCounterBuffer) {
        std::cerr << "[GPUDebugHelper] Failed to create validation counter buffer" << std::endl;
        return false;
    }
    
    if (!createPipelines()) {
        return false;
    }
    
    initialized = true;
    std::cout << "[GPUDebugHelper] Initialized successfully" << std::endl;
    return true;
}

bool GPUDebugHelper::createPipelines() {
    NSError* error = nil;
    
    // Create pass-through pipeline
    id<MTLFunction> passThroughFunction = [defaultLibrary newFunctionWithName:@"audioDummyPassThroughKernel"];
    if (!passThroughFunction) {
        std::cerr << "[GPUDebugHelper] Failed to find audioDummyPassThroughKernel function" << std::endl;
        return false;
    }
    
    passThroughPipeline = [metalDevice newComputePipelineStateWithFunction:passThroughFunction error:&error];
    if (!passThroughPipeline) {
        std::cerr << "[GPUDebugHelper] Failed to create pass-through pipeline: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create test pattern pipeline
    id<MTLFunction> testPatternFunction = [defaultLibrary newFunctionWithName:@"audioDebugTestPatternKernel"];
    if (!testPatternFunction) {
        std::cerr << "[GPUDebugHelper] Failed to find audioDebugTestPatternKernel function" << std::endl;
        return false;
    }
    
    testPatternPipeline = [metalDevice newComputePipelineStateWithFunction:testPatternFunction error:&error];
    if (!testPatternPipeline) {
        std::cerr << "[GPUDebugHelper] Failed to create test pattern pipeline: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create validation pipeline
    id<MTLFunction> validationFunction = [defaultLibrary newFunctionWithName:@"audioValidationKernel"];
    if (!validationFunction) {
        std::cerr << "[GPUDebugHelper] Failed to find audioValidationKernel function" << std::endl;
        return false;
    }
    
    validationPipeline = [metalDevice newComputePipelineStateWithFunction:validationFunction error:&error];
    if (!validationPipeline) {
        std::cerr << "[GPUDebugHelper] Failed to create validation pipeline: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    return true;
}

bool GPUDebugHelper::processDummyAudio(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> inputBuffer,
    id<MTLBuffer> outputBuffer,
    size_t numSamples,
    int frameIndex,
    DebugMode mode) {
    
    if (!initialized) {
        std::cerr << "[GPUDebugHelper] Not initialized!" << std::endl;
        return false;
    }
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) {
        std::cerr << "[GPUDebugHelper] Failed to create compute encoder" << std::endl;
        return false;
    }
    
    // Choose pipeline based on mode
    id<MTLComputePipelineState> pipeline;
    switch (mode) {
        case DebugMode::PassThrough:
            pipeline = passThroughPipeline;
            break;
        case DebugMode::TestPattern:
            pipeline = testPatternPipeline;
            break;
        case DebugMode::Validation:
            pipeline = validationPipeline;
            break;
    }
    
    [encoder setComputePipelineState:pipeline];
    
    // Set up parameters
    DummyParams params = {
        .frameOffset = 0,
        .numSamples = static_cast<uint32_t>(numSamples),
        .testGain = 1.0f,
        .frameIndex = static_cast<uint32_t>(frameIndex)
    };
    
    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:inputBuffer offset:0 atIndex:1];
    [encoder setBuffer:outputBuffer offset:0 atIndex:2];
    
    if (mode == DebugMode::Validation) {
        // Reset counter
        uint32_t zero = 0;
        memcpy([validationCounterBuffer contents], &zero, sizeof(uint32_t));
        [encoder setBuffer:validationCounterBuffer offset:0 atIndex:3];
    }
    
    // Dispatch
    MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
    MTLSize threadgroupCount = MTLSizeMake((numSamples + 63) / 64, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    return true;
}

bool GPUDebugHelper::processDummyAudioWithLogging(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> inputBuffer,
    id<MTLBuffer> outputBuffer,
    size_t numSamples,
    int frameIndex,
    FrameStateTracker& frameTracker,
    DebugMode mode) {
    
    std::cout << "[GPUDebugHelper] Processing frame " << frameIndex 
              << " with " << numSamples << " samples" << std::endl;
    
    // Mark dispatch stage
    frameTracker.markGPUDispatched(frameIndex);
    
    bool success = processDummyAudio(commandBuffer, inputBuffer, outputBuffer, 
                                   numSamples, frameIndex, mode);
    
    if (!success) {
        std::cerr << "[GPUDebugHelper] Failed to process dummy audio for frame " << frameIndex << std::endl;
        return false;
    }
    
    // Mark commit stage (will be called when command buffer is committed)
    frameTracker.markGPUCommitted(frameIndex);
    
    // Add completion handler that will mark completion stage
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.status == MTLCommandBufferStatusCompleted) {
            frameTracker.markGPUCompleted(frameIndex);
            std::cout << "[GPUDebugHelper] ✅ Frame " << frameIndex << " completed successfully" << std::endl;
        } else {
            std::cerr << "[GPUDebugHelper] ❌ Frame " << frameIndex << " failed with status: " << (int)cb.status << std::endl;
            if (cb.error) {
                std::cerr << "Error: " << [[cb.error localizedDescription] UTF8String] << std::endl;
            }
        }
    }];
    
    return true;
}

uint32_t GPUDebugHelper::getValidationNonZeroCount() const {
    if (!validationCounterBuffer) return 0;
    uint32_t* ptr = (uint32_t*)[validationCounterBuffer contents];
    return *ptr;
}

void GPUDebugHelper::resetValidationCounter() {
    if (validationCounterBuffer) {
        uint32_t zero = 0;
        memcpy([validationCounterBuffer contents], &zero, sizeof(uint32_t));
    }
}

#endif
