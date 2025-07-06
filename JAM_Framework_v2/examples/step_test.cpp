#include <iostream>
#include "GPURenderEngine.h"
#include "AudioOutputBackend.h"

using namespace JAMNet;

int main() {
    std::cout << "=== Step-by-Step Test ===" << std::endl;
    
    try {
        // Step 1: Create GPU engine
        std::cout << "Step 1: Creating GPU engine..." << std::endl;
        auto gpuEngine = GPURenderEngine::create();
        
        if (!gpuEngine) {
            std::cerr << "Failed to create GPU render engine" << std::endl;
            return 1;
        }
        std::cout << "Step 1: SUCCESS" << std::endl;
        
        // Step 2: Setup config
        std::cout << "Step 2: Setting up config..." << std::endl;
        GPURenderEngine::RenderConfig gpuConfig;
        gpuConfig.sampleRate = 48000;
        gpuConfig.bufferSize = 128;
        gpuConfig.channels = 2;
        gpuConfig.enablePNBTR = true;
        std::cout << "Step 2: SUCCESS" << std::endl;
        
        // Step 3: Initialize GPU engine  
        std::cout << "Step 3: Initializing GPU engine..." << std::endl;
        bool initResult = gpuEngine->initialize(gpuConfig);
        if (!initResult) {
            std::cerr << "Failed to initialize GPU render engine" << std::endl;
            return 1;
        }
        std::cout << "Step 3: SUCCESS" << std::endl;
        
        // Step 4: Check GPU availability
        std::cout << "Step 4: Checking GPU availability..." << std::endl;
        bool gpuAvailable = gpuEngine->isGPUAvailable();
        std::cout << "GPU available: " << (gpuAvailable ? "Yes" : "No") << std::endl;
        std::cout << "Step 4: SUCCESS" << std::endl;
        
        // Step 5: Test audio backend enumeration
        std::cout << "Step 5: Getting available audio backends..." << std::endl;
        auto availableBackends = AudioOutputBackend::getAvailableBackends();
        std::cout << "Found " << availableBackends.size() << " backends" << std::endl;
        std::cout << "Step 5: SUCCESS" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
    
    std::cout << "All steps completed successfully!" << std::endl;
    return 0;
}
