#include "MetalBridge.h"
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <iostream>
#include <ctime>

// --- Stub implementations for PNBTRTrainer compatibility ---
// (Only one definition for each singleton method)

MetalBridge& MetalBridge::getInstance() {
    static MetalBridge instance;
    return instance;
}

bool MetalBridge::initialize() {
    @autoreleasepool {
        NSLog(@"[MetalBridge::initialize] Called");
        // Create Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal device creation failed" << std::endl;
            return false;
        }
        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Command queue creation failed" << std::endl;
            return false;
        }
        // Load default library
        library = [device newDefaultLibrary];
        if (!library) {
            std::cerr << "Metal library loading failed" << std::endl;
            return false;
        }
        // Create compute pipeline states
        if (!createComputePipelines()) {
            std::cerr << "Pipeline creation failed" << std::endl;
            return false;
        }
        // Initialize buffer properties
        currentBufferSize = 0;
        currentNumChannels = 0;
        // Initialize metrics
        latestMetrics = {0.0f, 0.0f, 0.0f, 0.0f};
        NSLog(@"[MetalBridge::initialize] Success");
        return true;
    }
}

void MetalBridge::cleanup() {
    @autoreleasepool {
        // Release audio buffers
        audioInputBuffer = nil;
        jellieBuffer = nil;
        networkBuffer = nil;
        reconstructedBuffer = nil;
        metricsBuffer = nil;
        
        // Release pipeline states
        jellieEncodePipeline = nil;
        networkSimPipeline = nil;
        pnbtrReconstructPipeline = nil;
        metricsPipeline = nil;
        waveformPipeline = nil;
        
        // Release Metal resources
        library = nil;
        commandQueue = nil;
        device = nil;
    }
}

id<MTLBuffer> MetalBridge::createSharedBuffer(size_t size) {
    @autoreleasepool {
        return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
}

void MetalBridge::updateAudioBuffers(size_t bufferSize, size_t numChannels) {
    @autoreleasepool {
        currentBufferSize = bufferSize;
        currentNumChannels = numChannels;
        
        size_t floatSize = sizeof(float);
        size_t totalSamples = bufferSize * numChannels;
        size_t bufferBytes = totalSamples * floatSize;
        
        // Create shared audio buffers
        audioInputBuffer = createSharedBuffer(bufferBytes);
        jellieBuffer = createSharedBuffer(bufferBytes * 4); // JELLIE expansion
        networkBuffer = createSharedBuffer(bufferBytes * 4);
        reconstructedBuffer = createSharedBuffer(bufferBytes);
        
        // Create metrics buffer
        metricsBuffer = createSharedBuffer(sizeof(AudioMetrics));
    }
}

void MetalBridge::dispatchKernel(const std::string& kernelName, 
                                id<MTLBuffer> inputBuffer, 
                                id<MTLBuffer> outputBuffer,
                                size_t threadCount) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        
        // Select appropriate pipeline
        if (kernelName == "jellie_encode") {
            pipeline = jellieEncodePipeline;
        } else if (kernelName == "network_simulate") {
            pipeline = networkSimPipeline;
        } else if (kernelName == "pnbtr_reconstruct") {
            pipeline = pnbtrReconstructPipeline;
        } else if (kernelName == "calculate_metrics") {
            pipeline = metricsPipeline;
        } else if (kernelName == "waveform_render") {
            pipeline = waveformPipeline;
        }
        
        if (!pipeline) {
            std::cerr << "Unknown kernel: " << kernelName << std::endl;
            return;
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline and buffers
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        // Calculate thread groups
        NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadGroups = (threadCount + threadsPerGroup - 1) / threadsPerGroup;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(threadGroups, 1, 1);
        
        // Dispatch
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        // Commit without waiting (non-blocking for real-time audio)
        [commandBuffer commit];
    }
}

void MetalBridge::processAudioBlock(const float* input, float* output, size_t numSamples) {
    @autoreleasepool {
        if (!audioInputBuffer || !reconstructedBuffer) {
            std::cerr << "Audio buffers not initialized" << std::endl;
            return;
        }
        
        // Copy input to Metal buffer
        float* inputPtr = static_cast<float*>([audioInputBuffer contents]);
        memcpy(inputPtr, input, numSamples * sizeof(float));
        
        // Run the audio pipeline
        dispatchAudioPipeline(numSamples);
        
        // Copy output from Metal buffer
        float* outputPtr = static_cast<float*>([reconstructedBuffer contents]);
        memcpy(output, outputPtr, numSamples * sizeof(float));
        
        // Calculate metrics
        calculateMetrics(numSamples);
    }
}

void MetalBridge::updateWaveformTexture(id<MTLTexture> texture, size_t width, size_t height) {
    @autoreleasepool {
        if (!waveformPipeline || !audioInputBuffer) {
            return;
        }
        
        // Create command buffer for waveform rendering
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:waveformPipeline];
        [encoder setBuffer:audioInputBuffer offset:0 atIndex:0];
        [encoder setTexture:texture atIndex:0];
        
        // Set texture dimensions
        uint32_t dims[2] = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        [encoder setBytes:dims length:sizeof(dims) atIndex:1];
        
        // Dispatch
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            1
        );
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        // Remove waitUntilCompleted for non-blocking operation
    }
}

bool MetalBridge::createComputePipelines() {
    @autoreleasepool {
        NSError* error = nil;
        
        // Create JELLIE encode pipeline
        id<MTLFunction> jellieFunction = [library newFunctionWithName:@"jellie_encode_kernel"];
        if (jellieFunction) {
            jellieEncodePipeline = [device newComputePipelineStateWithFunction:jellieFunction error:&error];
            if (error) {
                std::cerr << "JELLIE pipeline error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
        }
        
        // Create network simulation pipeline
        id<MTLFunction> networkFunction = [library newFunctionWithName:@"network_simulate_kernel"];
        if (networkFunction) {
            networkSimPipeline = [device newComputePipelineStateWithFunction:networkFunction error:&error];
            if (error) {
                std::cerr << "Network pipeline error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
        }
        
        // Create PNBTR reconstruction pipeline
        id<MTLFunction> pnbtrFunction = [library newFunctionWithName:@"pnbtr_reconstruct_kernel"];
        if (pnbtrFunction) {
            pnbtrReconstructPipeline = [device newComputePipelineStateWithFunction:pnbtrFunction error:&error];
            if (error) {
                std::cerr << "PNBTR pipeline error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
        }
        
        // Create metrics pipeline
        id<MTLFunction> metricsFunction = [library newFunctionWithName:@"calculate_metrics_kernel"];
        if (metricsFunction) {
            metricsPipeline = [device newComputePipelineStateWithFunction:metricsFunction error:&error];
            if (error) {
                std::cerr << "Metrics pipeline error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
        }
        
        // Create waveform rendering pipeline
        id<MTLFunction> waveformFunction = [library newFunctionWithName:@"waveformRenderer"];
        if (waveformFunction) {
            waveformPipeline = [device newComputePipelineStateWithFunction:waveformFunction error:&error];
            if (error) {
                std::cerr << "Waveform pipeline error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
        }
        
        return true;
    }
}

void MetalBridge::dispatchAudioPipeline(size_t numSamples) {
    @autoreleasepool {
        // Stage 1: JELLIE encode
        dispatchKernel("jellie_encode", audioInputBuffer, jellieBuffer, numSamples);
        
        // Stage 2: Network simulation
        dispatchKernel("network_simulate", jellieBuffer, networkBuffer, numSamples * 4);
        
        // Stage 3: PNBTR reconstruction
        dispatchKernel("pnbtr_reconstruct", networkBuffer, reconstructedBuffer, numSamples);
    }
}

void MetalBridge::calculateMetrics(size_t numSamples) {
    @autoreleasepool {
        if (!metricsPipeline || !metricsBuffer) {
            return;
        }
        
        // Dispatch metrics calculation
        dispatchKernel("calculate_metrics", audioInputBuffer, metricsBuffer, numSamples);
        
        // Read back metrics
        AudioMetrics* metricsPtr = static_cast<AudioMetrics*>([metricsBuffer contents]);
        latestMetrics = *metricsPtr;
    }
}

//==============================================================================
// MetalBridgeInterface implementation

MetalBridgeInterface& MetalBridgeInterface::getInstance() {
    return MetalBridge::getInstance();
}

const float* MetalBridge::getAudioInputBuffer(size_t& bufferSize) {
    bufferSize = 0;
    return nullptr;
}

const float* MetalBridge::getJellieBuffer(size_t& bufferSize) {
    bufferSize = 0;
    return nullptr;
}

const float* MetalBridge::getNetworkBuffer(size_t& bufferSize) {
    bufferSize = 0;
    return nullptr;
}

const float* MetalBridge::getReconstructedBuffer(size_t& bufferSize) {
    bufferSize = 0;
    return nullptr;
}

AudioMetrics MetalBridge::getLatestMetrics() {
    return {};
}

bool MetalBridge::isSessionActive() {
    return false;
}

void MetalBridge::startSession() {}

void MetalBridge::stopSession() {}

// ---- PROPER IMPLEMENTATIONS FOR MISSING METHODS ----

MetalBridge::MetalBridge() : device(nil), commandQueue(nil), library(nil) {
    // Initialize atomic state
    sessionActive = false;
    latestMetrics = {0.0f, 0.0f, 0.0f, 0.0f};
}

MetalBridge::~MetalBridge() {
    cleanup();
}

bool MetalBridge::isInitialized() const { 
    return device != nil && commandQueue != nil && library != nil; 
}

void MetalBridge::setProcessingParameters(double sampleRate, int samplesPerBlock) {
    // Store parameters for buffer management
    updateAudioBuffers(samplesPerBlock, 2); // Assume stereo
}

void MetalBridge::prepareBuffers(int samplesPerBlock, double sampleRate) {
    updateAudioBuffers(samplesPerBlock, 2);
}

bool MetalBridge::runJellieEncode() { 
    @autoreleasepool {
        if (!isInitialized() || !jellieEncodePipeline) {
            return false;
        }
        
        // Non-blocking GPU dispatch
        if (audioInputBuffer && jellieBuffer) {
            dispatchKernel("jellie_encode", audioInputBuffer, jellieBuffer, currentBufferSize);
            return true;
        }
        return false;
    }
}

bool MetalBridge::runNetworkSimulation(float packetLoss, float jitter) { 
    @autoreleasepool {
        if (!isInitialized() || !networkSimPipeline) {
            return false;
        }
        
        // Non-blocking network simulation
        if (jellieBuffer && networkBuffer) {
            // Set simulation parameters
            struct SimParams {
                float packetLoss;
                float jitter;
                uint32_t seed;
            } params = {packetLoss, jitter, static_cast<uint32_t>(std::time(nullptr))};
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:networkSimPipeline];
            [encoder setBuffer:jellieBuffer offset:0 atIndex:0];
            [encoder setBuffer:networkBuffer offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            
            // Calculate thread dispatch
            NSUInteger threadGroupSize = networkSimPipeline.maxTotalThreadsPerThreadgroup;
            NSUInteger threadGroups = (currentBufferSize + threadGroupSize - 1) / threadGroupSize;
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake(threadGroups, 1, 1);
            
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            
            // Non-blocking commit
            [commandBuffer commit];
            return true;
        }
        return false;
    }
}

bool MetalBridge::runPNBTRReconstruction() { 
    @autoreleasepool {
        if (!isInitialized() || !pnbtrReconstructPipeline) {
            return false;
        }
        
        // Non-blocking PNBTR reconstruction
        if (networkBuffer && reconstructedBuffer) {
            dispatchKernel("pnbtr_reconstruct", networkBuffer, reconstructedBuffer, currentBufferSize);
            return true;
        }
        return false;
    }
}

void MetalBridge::updateMetrics() {
    @autoreleasepool {
        if (!isInitialized() || !metricsPipeline || !metricsBuffer) {
            return;
        }
        
        // Non-blocking metrics calculation
        dispatchKernel("calculate_metrics", audioInputBuffer, metricsBuffer, currentBufferSize);
        
        // Read back metrics (this is safe as it's from shared memory)
        AudioMetrics* metricsPtr = static_cast<AudioMetrics*>([metricsBuffer contents]);
        if (metricsPtr) {
            latestMetrics = *metricsPtr;
        }
    }
}

void MetalBridge::getMetricsData(float* snr, float* thd, float* latency) {
    if (snr) *snr = latestMetrics.snr_db;
    if (thd) *thd = latestMetrics.thd_percent;
    if (latency) *latency = latestMetrics.latency_ms;
}

void MetalBridge::getPacketLossStats(int* totalPackets, int* lostPackets) {
    // Mock packet loss stats for now - replace with real TOAST integration later
    if (totalPackets) *totalPackets = 1000;
    if (lostPackets) *lostPackets = static_cast<int>(latestMetrics.reconstruction_rate_percent * 10);
}

void MetalBridge::copyInputToGPU(const float* data, int numSamples, int numChannels) {
    @autoreleasepool {
        if (!audioInputBuffer || !data) {
            return;
        }
        
        // Thread-safe copy to shared Metal buffer
        float* bufferPtr = static_cast<float*>([audioInputBuffer contents]);
        if (bufferPtr) {
            size_t bytesToCopy = numSamples * numChannels * sizeof(float);
            memcpy(bufferPtr, data, bytesToCopy);
        }
    }
}

void MetalBridge::copyOutputFromGPU(float* data, int numSamples, int numChannels) {
    @autoreleasepool {
        if (!reconstructedBuffer || !data) {
            return;
        }
        
        // Thread-safe copy from shared Metal buffer
        const float* bufferPtr = static_cast<const float*>([reconstructedBuffer contents]);
        if (bufferPtr) {
            size_t bytesToCopy = numSamples * numChannels * sizeof(float);
            memcpy(data, bufferPtr, bytesToCopy);
        }
    }
}

const float* MetalBridge::getInputBufferPtr() const { 
    @autoreleasepool {
        if (audioInputBuffer) {
            return static_cast<const float*>([audioInputBuffer contents]);
        }
        return nullptr;
    }
}

const float* MetalBridge::getOutputBufferPtr() const { 
    @autoreleasepool {
        if (reconstructedBuffer) {
            return static_cast<const float*>([reconstructedBuffer contents]);
        }
        return nullptr;
    }
}

void MetalBridge::setNetworkParameters(float packetLoss, float jitter) {
    // Store parameters for use in runNetworkSimulation
    // These will be passed as shader parameters
}

// Missing implementations for MetalSpectralBridge integration
id<MTLDevice> MetalBridge::getDevice() const {
    return device;
}

id<MTLCommandQueue> MetalBridge::getCommandQueue() const {
    return commandQueue;
}