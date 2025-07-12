#import "MetalBridgeDebugIntegration.h"
#import <iostream>
#import <chrono>
#import <iomanip>

MetalBridgeDebugIntegration& MetalBridgeDebugIntegration::getInstance() {
    static MetalBridgeDebugIntegration instance;
    return instance;
}

MetalBridgeDebugIntegration::MetalBridgeDebugIntegration() 
    : gpuDebugHelper(GPUDebugHelper::getInstance()) {
}

#ifdef __OBJC__

bool MetalBridgeDebugIntegration::initialize(id<MTLDevice> device) {
    if (!gpuDebugHelper.initializeWithDevice(device)) {
        std::cerr << "[MetalBridgeDebugIntegration] Failed to initialize GPU debug helper" << std::endl;
        return false;
    }
    
    debugInitialized = true;
    
    std::cout << "\n🔧 METAL BRIDGE DEBUG INTEGRATION INITIALIZED 🔧" << std::endl;
    std::cout << "Frame logging: " << (debugConfig.enableFrameLogging ? "✅" : "❌") << std::endl;
    std::cout << "GPU validation: " << (debugConfig.enableGPUValidation ? "✅" : "❌") << std::endl;
    std::cout << "Dummy processing: " << (debugConfig.useDummyProcessing ? "✅" : "❌") << std::endl;
    std::cout << "Pipeline logging: " << (debugConfig.enableDetailedPipelineLogging ? "✅" : "❌") << std::endl;
    std::cout << "===================================================\n" << std::endl;
    
    return true;
}

bool MetalBridgeDebugIntegration::processAudioFrameWithFullDebugging(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> inputBuffer,
    id<MTLBuffer> outputBuffer,
    size_t numSamples,
    int frameIndex) {
    
    if (!debugInitialized) {
        std::cerr << "[MetalBridgeDebugIntegration] Not initialized!" << std::endl;
        return false;
    }
    
    // Start timing
    if (debugConfig.enableLatencyTracking) {
        startFrameTimer(frameIndex);
    }
    
    // Reset frame state
    frameTracker.resetFrame(frameIndex);
    
    // CHECKPOINT 1: Frame started
    logCheckpoint(1, "Frame processing started", frameIndex);
    
    // Validate input buffer has non-zero data
    float* inputData = (float*)[inputBuffer contents];
    float maxInput = 0.0f;
    for (size_t i = 0; i < numSamples * 2; ++i) {
        maxInput = std::max(maxInput, std::abs(inputData[i]));
    }
    
    // CHECKPOINT 2: Input validated
    logCheckpoint(2, "CoreAudio→MetalBridge input validated", frameIndex, maxInput);
    
    bool gpuProcessingSuccess = false;
    
    if (debugConfig.useDummyProcessing) {
        // Use dummy processing to test GPU pipeline without DSP
        std::cout << "[DEBUG MODE] Using dummy GPU processing for frame " << frameIndex << std::endl;
        
        auto mode = debugConfig.enableGPUValidation ? 
            GPUDebugHelper::DebugMode::Validation : 
            GPUDebugHelper::DebugMode::PassThrough;
        
        gpuProcessingSuccess = gpuDebugHelper.processDummyAudioWithLogging(
            commandBuffer, inputBuffer, outputBuffer, numSamples, frameIndex, frameTracker, mode
        );
        
        // CHECKPOINT 3: GPU processing attempted
        if (gpuProcessingSuccess) {
            logCheckpoint(3, "Dummy GPU processing dispatched", frameIndex);
        } else {
            logCheckpoint(3, "❌ Dummy GPU processing failed", frameIndex);
        }
        
    } else {
        // TODO: This is where you would call your actual DSP processing
        // For now, we'll use dummy processing to test the pipeline
        std::cout << "[NORMAL MODE] Would call actual DSP processing for frame " << frameIndex << std::endl;
        
        gpuProcessingSuccess = gpuDebugHelper.processDummyAudioWithLogging(
            commandBuffer, inputBuffer, outputBuffer, numSamples, frameIndex, frameTracker,
            GPUDebugHelper::DebugMode::PassThrough
        );
        
        logCheckpoint(3, "Actual DSP processing would be called here", frameIndex);
    }
    
    if (!gpuProcessingSuccess) {
        logCheckpoint(4, "❌ GPU processing failed completely", frameIndex);
        failedFrames++;
        return false;
    }
    
    // Commit command buffer - this is critical!
    [commandBuffer commit];
    frameTracker.markGPUCommitted(frameIndex);
    
    // CHECKPOINT 4: Command buffer committed
    logCheckpoint(4, "GPU command buffer committed", frameIndex);
    
    // Mark CPU stage as complete (we're the CPU calling this)
    frameTracker.markStageComplete("CPU", frameIndex);
    
    // The GPU completion will be handled by the completion handler in GPUDebugHelper
    
    totalFramesProcessed++;
    successfulFrames++;
    
    // End timing
    if (debugConfig.enableLatencyTracking) {
        endFrameTimer(frameIndex);
    }
    
    // Log summary periodically
    if (debugConfig.enableFrameLogging && 
        totalFramesProcessed % debugConfig.logEveryNFrames == 0) {
        logPerformanceStats();
        frameTracker.logPipelineStatus();
    }
    
    return true;
}

void MetalBridgeDebugIntegration::logCheckpoint(int checkpointNumber, const std::string& message, 
                                              int frameIndex, float peakValue) {
    if (!debugConfig.enableFrameLogging) return;
    
    std::ostringstream oss;
    oss << "[";
    
    // Color-code based on checkpoint
    if (message.find("❌") != std::string::npos) {
        oss << "❌ CHECKPOINT " << checkpointNumber;
    } else {
        oss << "✅ CHECKPOINT " << checkpointNumber;
    }
    
    oss << "] " << message;
    
    if (peakValue > 0.0f) {
        oss << " | Peak: " << std::fixed << std::setprecision(6) << peakValue;
    }
    
    oss << " | Frame: " << frameIndex;
    
    std::cout << oss.str() << std::endl;
}

bool MetalBridgeDebugIntegration::validateGPUPipeline(int frameIndex) {
    // Check if frame completed all stages
    bool isReady = frameTracker.isFrameReady(frameIndex);
    
    if (debugConfig.enableDetailedPipelineLogging) {
        frameTracker.logDetailed(frameIndex, "[VALIDATE]");
    }
    
    return isReady;
}

void MetalBridgeDebugIntegration::startFrameTimer(int frameIndex) {
    frameStartTimes[frameIndex] = std::chrono::high_resolution_clock::now();
}

void MetalBridgeDebugIntegration::endFrameTimer(int frameIndex) {
    auto it = frameStartTimes.find(frameIndex);
    if (it != frameStartTimes.end()) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second);
        
        float latencyUs = duration.count();
        frameLatencies.push_back(latencyUs);
        
        // Keep only last 1000 measurements
        if (frameLatencies.size() > 1000) {
            frameLatencies.erase(frameLatencies.begin());
        }
        
        frameStartTimes.erase(it);
    }
}

void MetalBridgeDebugIntegration::logPerformanceStats() {
    if (frameLatencies.empty()) return;
    
    float avgLatency = 0.0f;
    float maxLatency = 0.0f;
    float minLatency = frameLatencies[0];
    
    for (float latency : frameLatencies) {
        avgLatency += latency;
        maxLatency = std::max(maxLatency, latency);
        minLatency = std::min(minLatency, latency);
    }
    avgLatency /= frameLatencies.size();
    
    std::cout << "\n📊 PERFORMANCE STATS 📊" << std::endl;
    std::cout << "Total frames: " << totalFramesProcessed << std::endl;
    std::cout << "Successful: " << successfulFrames << " (" 
              << (100.0f * successfulFrames / totalFramesProcessed) << "%)" << std::endl;
    std::cout << "Failed: " << failedFrames << " (" 
              << (100.0f * failedFrames / totalFramesProcessed) << "%)" << std::endl;
    std::cout << "Avg latency: " << std::fixed << std::setprecision(2) << avgLatency << "µs" << std::endl;
    std::cout << "Min latency: " << minLatency << "µs" << std::endl;
    std::cout << "Max latency: " << maxLatency << "µs" << std::endl;
    
    if (debugConfig.enableGPUValidation) {
        uint32_t nonZeroCount = gpuDebugHelper.getValidationNonZeroCount();
        std::cout << "Non-zero samples: " << nonZeroCount << std::endl;
    }
    
    std::cout << "========================\n" << std::endl;
}

#endif
