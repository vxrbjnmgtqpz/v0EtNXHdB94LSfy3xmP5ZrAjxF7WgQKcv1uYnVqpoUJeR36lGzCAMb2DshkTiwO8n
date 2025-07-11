#include <iostream>

// Import the debugging functions
extern "C" {
    void enableCoreAudioSineTest(bool enable);
    void checkMetalBridgeStatus();
    void forceCoreAudioCallback();
    void useDefaultInputDevice();
}

int main() {
    std::cout << "🔧 Testing Core Audio debugging functions directly...\n";
    
    // Test each debugging function
    std::cout << "1. Testing enableCoreAudioSineTest...\n";
    enableCoreAudioSineTest(true);
    
    std::cout << "2. Testing checkMetalBridgeStatus...\n";
    checkMetalBridgeStatus();
    
    std::cout << "3. Testing useDefaultInputDevice...\n";
    useDefaultInputDevice();
    
    std::cout << "4. Testing forceCoreAudioCallback...\n";
    forceCoreAudioCallback();
    
    std::cout << "✅ All debugging functions called successfully\n";
    return 0;
} 