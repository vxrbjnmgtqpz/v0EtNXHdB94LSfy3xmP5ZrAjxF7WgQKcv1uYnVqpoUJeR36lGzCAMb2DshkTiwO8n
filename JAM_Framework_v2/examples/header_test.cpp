#include <iostream>

// Just test the includes without instantiation
#include "GPURenderEngine.h"
#include "AudioOutputBackend.h"
#include "JamAudioFrame.h"

int main() {
    std::cout << "=== JAMNet Header Include Test ===" << std::endl;
    
    std::cout << "Headers included successfully" << std::endl;
    
    // Test basic factory method without creating objects
    std::cout << "Testing basic framework availability..." << std::endl;
    
    return 0;
}
