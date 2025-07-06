#include "VulkanRenderEngine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

/**
 * Vulkan GPU Audio Backend Validation Test
 * Tests the basic functionality of VulkanRenderEngine implementation
 */

using namespace JAMNet;

bool testVulkanDeviceDetection() {
    std::cout << "\n=== Testing Vulkan Device Detection ===" << std::endl;
    
    VulkanRenderEngine engine;
    RenderConfig config = {
        .sampleRate = 48000,
        .bufferSize = 64,
        .channels = 2
    };
    
    bool success = engine.initialize(config);
    
    if (success) {
        std::cout << "✓ Vulkan device detection and initialization successful" << std::endl;
        std::cout << "✓ GPU available: " << (engine.isGPUAvailable() ? "Yes" : "No") << std::endl;
        
        engine.shutdown();
        return true;
    } else {
        std::cout << "✗ Vulkan device detection failed" << std::endl;
        return false;
    }
}

bool testBufferOperations() {
    std::cout << "\n=== Testing Vulkan Buffer Operations ===" << std::endl;
    
    VulkanRenderEngine engine;
    RenderConfig config = {
        .sampleRate = 48000,
        .bufferSize = 64,
        .channels = 2
    };
    
    if (!engine.initialize(config)) {
        std::cout << "✗ Failed to initialize engine for buffer test" << std::endl;
        return false;
    }
    
    // Test buffer access
    void* sharedBuffer = engine.getSharedBuffer();
    size_t bufferSize = engine.getSharedBufferSize();
    
    if (sharedBuffer && bufferSize > 0) {
        std::cout << "✓ Shared buffer accessible: " << bufferSize << " bytes" << std::endl;
        
        // Test buffer flush
        bool flushResult = engine.flushBufferToGPU();
        std::cout << "✓ Buffer flush: " << (flushResult ? "Success" : "Failed") << std::endl;
        
        engine.shutdown();
        return true;
    } else {
        std::cout << "✗ Shared buffer not accessible" << std::endl;
        engine.shutdown();
        return false;
    }
}

bool testAudioProcessing() {
    std::cout << "\n=== Testing Vulkan Audio Processing ===" << std::endl;
    
    VulkanRenderEngine engine;
    RenderConfig config = {
        .sampleRate = 48000,
        .bufferSize = 64,
        .channels = 2
    };
    
    if (!engine.initialize(config)) {
        std::cout << "✗ Failed to initialize engine for audio test" << std::endl;
        return false;
    }
    
    // Create test input samples (sine wave)
    std::vector<float> inputSamples(config.bufferSize * config.channels);
    const float frequency = 440.0f; // A4 note
    const float sampleRate = static_cast<float>(config.sampleRate);
    
    for (uint32_t i = 0; i < config.bufferSize; i++) {
        float sample = 0.5f * sin(2.0f * M_PI * frequency * i / sampleRate);
        inputSamples[i * config.channels] = sample;     // Left channel
        inputSamples[i * config.channels + 1] = sample; // Right channel
    }
    
    // Test audio rendering
    auto start = std::chrono::high_resolution_clock::now();
    bool renderResult = engine.renderToBuffer(inputSamples.data(), config.bufferSize);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto renderTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (renderResult) {
        std::cout << "✓ Audio rendering successful" << std::endl;
        std::cout << "✓ Render time: " << renderTime.count() << " µs" << std::endl;
        
        // Check if render time is reasonable for real-time (should be < 1ms)
        if (renderTime.count() < 1000) {
            std::cout << "✓ Render time within real-time constraints" << std::endl;
        } else {
            std::cout << "⚠ Render time exceeds real-time threshold (1ms)" << std::endl;
        }
        
        engine.shutdown();
        return true;
    } else {
        std::cout << "✗ Audio rendering failed" << std::endl;
        engine.shutdown();
        return false;
    }
}

bool testTimestampAccuracy() {
    std::cout << "\n=== Testing Timestamp Accuracy ===" << std::endl;
    
    VulkanRenderEngine engine;
    RenderConfig config = {
        .sampleRate = 48000,
        .bufferSize = 64,
        .channels = 2
    };
    
    if (!engine.initialize(config)) {
        std::cout << "✗ Failed to initialize engine for timestamp test" << std::endl;
        return false;
    }
    
    // Test timestamp retrieval
    uint64_t timestamp1 = engine.getCurrentTimestamp();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    uint64_t timestamp2 = engine.getCurrentTimestamp();
    
    uint64_t timeDiff = timestamp2 - timestamp1;
    
    std::cout << "✓ Timestamp 1: " << timestamp1 << " ns" << std::endl;
    std::cout << "✓ Timestamp 2: " << timestamp2 << " ns" << std::endl;
    std::cout << "✓ Time difference: " << timeDiff << " ns (~" << timeDiff / 1000000 << " ms)" << std::endl;
    
    // Check if time difference is reasonable (should be around 10ms)
    if (timeDiff >= 9000000 && timeDiff <= 11000000) { // 9-11ms tolerance
        std::cout << "✓ Timestamp accuracy within expected range" << std::endl;
        engine.shutdown();
        return true;
    } else {
        std::cout << "⚠ Timestamp accuracy outside expected range" << std::endl;
        engine.shutdown();
        return false;
    }
}

bool testPNBTRPrediction() {
    std::cout << "\n=== Testing PNBTR Prediction ===" << std::endl;
    
    VulkanRenderEngine engine;
    RenderConfig config = {
        .sampleRate = 48000,
        .bufferSize = 64,
        .channels = 2
    };
    
    if (!engine.initialize(config)) {
        std::cout << "✗ Failed to initialize engine for PNBTR test" << std::endl;
        return false;
    }
    
    // Test PNBTR prediction
    uint32_t predictSamples = 2400; // 50ms @ 48kHz
    bool predictResult = engine.renderPredictedAudio(predictSamples);
    
    if (predictResult) {
        std::cout << "✓ PNBTR prediction successful for " << predictSamples << " samples" << std::endl;
        engine.shutdown();
        return true;
    } else {
        std::cout << "✗ PNBTR prediction failed" << std::endl;
        engine.shutdown();
        return false;
    }
}

int main() {
    std::cout << "Vulkan GPU Audio Backend Validation Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    bool allTestsPassed = true;
    
    allTestsPassed &= testVulkanDeviceDetection();
    allTestsPassed &= testBufferOperations();
    allTestsPassed &= testAudioProcessing();
    allTestsPassed &= testTimestampAccuracy();
    allTestsPassed &= testPNBTRPrediction();
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    if (allTestsPassed) {
        std::cout << "✓ All tests passed! Vulkan implementation is functional." << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some tests failed. Check output above for details." << std::endl;
        return 1;
    }
}
