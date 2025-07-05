//
// minimal_gpu_transport_test.cpp
// Ultra-minimal test to isolate the GPU transport initialization bug
//

#define __OBJC__ 1  // Force Objective-C mode
#include <iostream>
#include "JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"
#include "JAM_Framework_v2/include/gpu_native/gpu_timebase.h"

using namespace jam::gpu_transport;

int main() {
    std::cout << "🔬 MINIMAL GPU Transport Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Initialize dependency
    std::cout << "\n1. Initializing GPU Timebase..." << std::endl;
    if (!jam::gpu_native::GPUTimebase::is_initialized() && !jam::gpu_native::GPUTimebase::initialize()) {
        std::cout << "❌ Failed to initialize GPU Timebase" << std::endl;
        return 1;
    }
    std::cout << "✅ GPU Timebase ready" << std::endl;
    
    // Get singleton instance
    std::cout << "\n2. Getting GPU Transport Manager instance..." << std::endl;
    auto& manager = GPUTransportManager::getInstance();
    std::cout << "Instance address: " << &manager << std::endl;
    
    // Check initial state
    std::cout << "\n3. Checking initial initialization state..." << std::endl;
    bool initial_state = manager.isInitialized();
    std::cout << "Initial state: " << (initial_state ? "INITIALIZED" : "NOT INITIALIZED") << std::endl;
    
    if (!initial_state) {
        std::cout << "\n4. Calling initialize()..." << std::endl;
        bool init_result = manager.initialize();
        std::cout << "Initialize() returned: " << (init_result ? "true" : "false") << std::endl;
        
        std::cout << "\n5. Checking state IMMEDIATELY after initialize()..." << std::endl;
        bool post_init_state = manager.isInitialized();
        std::cout << "Post-init state: " << (post_init_state ? "INITIALIZED" : "NOT INITIALIZED") << std::endl;
        
        if (init_result && !post_init_state) {
            std::cout << "🚨 BUG CONFIRMED: initialize() returned true but isInitialized() returns false!" << std::endl;
        } else if (init_result && post_init_state) {
            std::cout << "✅ SUCCESS: Initialization worked correctly!" << std::endl;
            
            // Test basic transport operations
            std::cout << "\n6. Testing basic transport..." << std::endl;
            std::cout << "Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
            std::cout << "Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
            
            std::cout << "\n7. Testing PLAY command..." << std::endl;
            manager.play();
            manager.update();
            std::cout << "After PLAY - Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
            std::cout << "State: " << static_cast<int>(manager.getCurrentState()) << std::endl;
            
            std::cout << "\n8. Testing STOP command..." << std::endl;
            manager.stop();
            manager.update();
            std::cout << "After STOP - Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
            std::cout << "State: " << static_cast<int>(manager.getCurrentState()) << std::endl;
            
            std::cout << "\n✅ All tests completed successfully!" << std::endl;
        }
    } else {
        std::cout << "Manager was already initialized" << std::endl;
    }
    
    std::cout << "\n9. Cleanup..." << std::endl;
    jam::gpu_native::GPUTimebase::shutdown();
    
    return 0;
}
