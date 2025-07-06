#include <iostream>
#include "AudioOutputBackend.h"

using namespace JAMNet;

int main() {
    std::cout << "=== Audio Backend Test ===" << std::endl;
    
    try {
        // Step 1: Get available backends
        std::cout << "Step 1: Getting available backends..." << std::endl;
        auto availableBackends = AudioOutputBackend::getAvailableBackends();
        std::cout << "Found " << availableBackends.size() << " backends" << std::endl;
        
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
        
        // Step 2: Try to create JACK backend
        std::cout << "Step 2: Creating JACK audio backend..." << std::endl;
        auto audioBackend = AudioOutputBackend::create(AudioOutputBackend::BackendType::JACK);
        if (!audioBackend) {
            std::cerr << "Failed to create JACK audio backend, trying default..." << std::endl;
            // Try default
            if (!availableBackends.empty()) {
                audioBackend = AudioOutputBackend::create(availableBackends[0]);
            }
        }
        
        if (!audioBackend) {
            std::cerr << "Failed to create any audio backend" << std::endl;
            return 1;
        }
        
        std::cout << "Created audio backend: " << audioBackend->getName() << std::endl;
        
        // Step 3: Configure
        std::cout << "Step 3: Configuring audio backend..." << std::endl;
        AudioOutputBackend::AudioConfig audioConfig;
        audioConfig.sampleRate = 48000;
        audioConfig.bufferSize = 128;
        audioConfig.channels = 2;
        audioConfig.enableLowLatency = true;
        audioConfig.preferGPUMemory = true;
        
        // Step 4: Initialize
        std::cout << "Step 4: Initializing audio backend..." << std::endl;
        bool initResult = audioBackend->initialize(audioConfig);
        if (!initResult) {
            std::cerr << "Failed to initialize audio backend" << std::endl;
            return 1;
        }
        
        std::cout << "Audio backend initialized successfully" << std::endl;
        std::cout << "Supports GPU memory: " << (audioBackend->supportsGPUMemory() ? "Yes" : "No") << std::endl;
        std::cout << "Actual latency: " << audioBackend->getActualLatencyMs() << "ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
    
    std::cout << "Audio backend test completed successfully!" << std::endl;
    return 0;
}
