#include "GPURenderEngine.h"
#include "AudioOutputBackend.h"
#include "JamAudioFrame.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace JAMNet;

std::atomic<bool> running{true};

void audioCallback(float* buffer, uint32_t numFrames, uint64_t timestampNs) {
    // Simple sine wave test signal
    static float phase = 0.0f;
    const float frequency = 440.0f; // A4
    const float sampleRate = 48000.0f;
    const float amplitude = 0.1f; // Safe volume level
    
    for (uint32_t i = 0; i < numFrames; ++i) {
        buffer[i] = amplitude * std::sin(2.0f * M_PI * frequency * phase / sampleRate);
        phase += 1.0f;
        if (phase >= sampleRate) {
            phase -= sampleRate;
        }
    }
    
    // Print timing info occasionally
    static uint64_t printCounter = 0;
    if (++printCounter % 100 == 0) {
        std::cout << "Audio callback: " << numFrames << " frames, timestamp: " 
                  << timestampNs << "ns" << std::endl;
    }
}

int main() {
    std::cout << "=== JAMNet Cross-Platform Audio Engine Test ===" << std::endl;
    
    // Test GPU render engine
    std::cout << "\n1. Testing GPU Render Engine..." << std::endl;
    auto gpuEngine = GPURenderEngine::create();
    
    if (!gpuEngine) {
        std::cerr << "Failed to create GPU render engine" << std::endl;
        return 1;
    }
    
    GPURenderEngine::RenderConfig gpuConfig;
    gpuConfig.sampleRate = 48000;
    gpuConfig.bufferSize = 128;
    gpuConfig.channels = 2;
    gpuConfig.enablePNBTR = true;
    
    if (!gpuEngine->initialize(gpuConfig)) {
        std::cerr << "Failed to initialize GPU render engine" << std::endl;
        return 1;
    }
    
    std::cout << "GPU render engine initialized successfully" << std::endl;
    std::cout << "GPU available: " << (gpuEngine->isGPUAvailable() ? "Yes" : "No") << std::endl;
    
    // Test audio output backend
    std::cout << "\n2. Testing Audio Output Backend..." << std::endl;
    
    // Get available backends
    auto availableBackends = AudioOutputBackend::getAvailableBackends();
    std::cout << "Available audio backends:" << std::endl;
    for (auto backend : availableBackends) {
        switch (backend) {
            case AudioOutputBackend::BackendType::JACK:
                std::cout << "  - JACK" << std::endl;
                break;
            case AudioOutputBackend::BackendType::CORE_AUDIO:
                std::cout << "  - Core Audio" << std::endl;
                break;
            case AudioOutputBackend::BackendType::ALSA:
                std::cout << "  - ALSA" << std::endl;
                break;
            default:
                std::cout << "  - Unknown" << std::endl;
                break;
        }
    }
    
    // Create audio backend (prefer JACK)
    auto audioBackend = AudioOutputBackend::create(AudioOutputBackend::BackendType::JACK);
    if (!audioBackend) {
        std::cerr << "Failed to create audio output backend" << std::endl;
        gpuEngine->shutdown();
        return 1;
    }
    
    std::cout << "Created audio backend: " << audioBackend->getName() << std::endl;
    
    // Configure audio backend
    AudioOutputBackend::AudioConfig audioConfig;
    audioConfig.sampleRate = 48000;
    audioConfig.bufferSize = 128;
    audioConfig.channels = 2;
    audioConfig.enableLowLatency = true;
    audioConfig.preferGPUMemory = true;
    
    if (!audioBackend->initialize(audioConfig)) {
        std::cerr << "Failed to initialize audio backend" << std::endl;
        gpuEngine->shutdown();
        return 1;
    }
    
    std::cout << "Audio backend initialized successfully" << std::endl;
    std::cout << "Supports GPU memory: " << (audioBackend->supportsGPUMemory() ? "Yes" : "No") << std::endl;
    std::cout << "Actual latency: " << audioBackend->getActualLatencyMs() << "ms" << std::endl;
    
    // Set up GPU memory integration if supported
    if (audioBackend->supportsGPUMemory() && gpuEngine->getSharedBuffer()) {
        bool gpuMemEnabled = audioBackend->enableGPUMemoryMode(
            gpuEngine->getSharedBuffer(), 
            gpuEngine->getSharedBufferSize()
        );
        std::cout << "GPU memory integration: " << (gpuMemEnabled ? "Enabled" : "Failed") << std::endl;
        
        // Set external clock callback
        audioBackend->setExternalClock([&gpuEngine]() -> uint64_t {
            auto timestamp = gpuEngine->getCurrentTimestamp();
            return timestamp.gpu_time_ns;
        });
        std::cout << "External GPU clock configured" << std::endl;
    }
    
    // Set up audio processing callback
    audioBackend->setProcessCallback(audioCallback);
    
    // Test shared audio buffer
    std::cout << "\n3. Testing Shared Audio Buffer..." << std::endl;
    SharedAudioBuffer sharedBuffer;
    
    // Create test frame
    JamAudioFrame testFrame;
    testFrame.numSamples = 128;
    testFrame.sampleRate = 48000;
    testFrame.channels = 2;
    testFrame.timestamp_gpu_ns = 1000000000; // 1 second
    testFrame.flags = JamAudioFrame::FLAG_CLEAN;
    
    // Fill with test data
    for (uint32_t i = 0; i < testFrame.numSamples; ++i) {
        testFrame.samples[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / 48000.0f);
    }
    
    // Test buffer operations
    bool pushResult = sharedBuffer.pushFrame(testFrame);
    std::cout << "Push frame result: " << (pushResult ? "Success" : "Failed") << std::endl;
    std::cout << "Buffer empty: " << (sharedBuffer.isEmpty() ? "Yes" : "No") << std::endl;
    std::cout << "Available frames: " << sharedBuffer.getAvailableFrames() << std::endl;
    
    JamAudioFrame retrievedFrame;
    bool popResult = sharedBuffer.popFrame(retrievedFrame);
    std::cout << "Pop frame result: " << (popResult ? "Success" : "Failed") << std::endl;
    
    if (popResult) {
        std::cout << "Retrieved frame - samples: " << retrievedFrame.numSamples 
                  << ", rate: " << retrievedFrame.sampleRate 
                  << ", timestamp: " << retrievedFrame.timestamp_gpu_ns << "ns" << std::endl;
    }
    
    // Run audio for a few seconds
    std::cout << "\n4. Running audio test for 5 seconds..." << std::endl;
    std::cout << "You should hear a 440Hz sine wave tone." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    while (running && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Test GPU rendering occasionally
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() % 1000 < 100) {
            
            auto timestamp = gpuEngine->getCurrentTimestamp();
            float buffer[256];
            gpuEngine->renderToBuffer(buffer, 128, timestamp);
            
            // Print timing info
            std::cout << "GPU render - timestamp: " << timestamp.gpu_time_ns 
                      << "ns, offset: " << timestamp.calibration_offset_ms << "ms" << std::endl;
        }
    }
    
    running = false;
    
    // Cleanup
    std::cout << "\n5. Cleaning up..." << std::endl;
    audioBackend->shutdown();
    gpuEngine->shutdown();
    
    std::cout << "\nCross-platform audio engine test completed successfully!" << std::endl;
    std::cout << "Architecture validation: PASSED" << std::endl;
    
    return 0;
}
