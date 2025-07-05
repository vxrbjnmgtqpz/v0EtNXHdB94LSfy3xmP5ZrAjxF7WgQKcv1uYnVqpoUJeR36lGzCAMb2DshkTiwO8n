//
// test_gpu_transport.cpp
// Direct test of GPU transport functionality
//

#include <iostream>
#include <thread>
#include <chrono>
#include "JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"
#include "JAM_Framework_v2/include/gpu_native/gpu_timebase.h"

using namespace jam::gpu_transport;
using namespace std::chrono_literals;

void test_transport_operations() {
    std::cout << "🧪 Testing GPU Transport Operations..." << std::endl;
    
    // Get transport manager instance
    auto& manager = GPUTransportManager::getInstance();
    
    // Test initialization
    std::cout << "\n1. Testing Initialization:" << std::endl;
    std::cout << "   Manager instance address: " << &manager << std::endl;
    std::cout << "   Initial state: " << (manager.isInitialized() ? "INITIALIZED" : "NOT INITIALIZED") << std::endl;
    
    if (!manager.isInitialized()) {
        std::cout << "   Attempting initialization on instance: " << &manager << std::endl;
        bool initResult = manager.initialize();
        std::cout << "   Init result: " << (initResult ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "   Checking IMMEDIATELY after init call on same instance: " << &manager << std::endl;
        
        // Check the flag IMMEDIATELY after the call - don't use isInitialized() method
        std::cout << "   Direct atomic check IMMEDIATELY after init: " << std::flush;
        auto& mgr_ref = manager;  // Ensure we have the exact same object
        // Unfortunately we can't access private members directly, so use the method
        std::cout << "   Post-init state: " << (manager.isInitialized() ? "INITIALIZED" : "NOT INITIALIZED") << std::endl;
    }
    
    if (!manager.isInitialized()) {
        std::cout << "❌ Cannot proceed - GPU Transport Manager not initialized" << std::endl;
        return;
    }
    
    // Test initial state
    std::cout << "\n2. Testing Initial State:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Is paused: " << (manager.isPaused() ? "YES" : "NO") << std::endl;
    std::cout << "   Is recording: " << (manager.isRecording() ? "YES" : "NO") << std::endl;
    std::cout << "   Current frame: " << manager.getCurrentFrame() << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Test PLAY
    std::cout << "\n3. Testing PLAY Command:" << std::endl;
    manager.play();
    std::this_thread::sleep_for(100ms);
    manager.update(); // Process GPU events
    
    std::cout << "   After PLAY:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Current frame: " << manager.getCurrentFrame() << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Let it play for a bit
    std::cout << "\n   Playing for 1 second..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(100ms);
        manager.update();
        std::cout << "   Position: " << manager.getPositionSeconds() << "s" << std::endl;
    }
    
    // Test PAUSE
    std::cout << "\n4. Testing PAUSE Command:" << std::endl;
    manager.pause();
    std::this_thread::sleep_for(100ms);
    manager.update();
    
    std::cout << "   After PAUSE:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Is paused: " << (manager.isPaused() ? "YES" : "NO") << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Wait and check position doesn't advance
    std::cout << "\n   Checking position stays frozen during pause..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(100ms);
        manager.update();
        std::cout << "   Position: " << manager.getPositionSeconds() << "s" << std::endl;
    }
    
    // Test resume from PAUSE
    std::cout << "\n5. Testing Resume from PAUSE:" << std::endl;
    manager.play();
    std::this_thread::sleep_for(100ms);
    manager.update();
    
    std::cout << "   After resume PLAY:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Test STOP
    std::cout << "\n6. Testing STOP Command:" << std::endl;
    manager.stop();
    std::this_thread::sleep_for(100ms);
    manager.update();
    
    std::cout << "   After STOP:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Test RECORD
    std::cout << "\n7. Testing RECORD Command:" << std::endl;
    manager.record();
    std::this_thread::sleep_for(100ms);
    manager.update();
    
    std::cout << "   After RECORD:" << std::endl;
    std::cout << "   Current state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    std::cout << "   Is playing: " << (manager.isPlaying() ? "YES" : "NO") << std::endl;
    std::cout << "   Is recording: " << (manager.isRecording() ? "YES" : "NO") << std::endl;
    std::cout << "   Position (seconds): " << manager.getPositionSeconds() << std::endl;
    
    // Test BPM changes
    std::cout << "\n8. Testing BPM Control:" << std::endl;
    std::cout << "   Initial BPM: " << manager.getBPM() << std::endl;
    manager.setBPM(140.0f);
    std::this_thread::sleep_for(100ms);
    manager.update();
    std::cout << "   After setting BPM to 140: " << manager.getBPM() << std::endl;
    
    // Final stop
    std::cout << "\n9. Final Cleanup:" << std::endl;
    manager.stop();
    manager.update();
    std::cout << "   Final state: " << static_cast<int>(manager.getCurrentState()) << std::endl;
    
    std::cout << "\n✅ Transport test complete!" << std::endl;
}

int main() {
    std::cout << "🚀 GPU Transport Functionality Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Initialize GPU timebase first (required dependency)
    std::cout << "\nInitializing GPU Timebase..." << std::endl;
    if (jam::gpu_native::GPUTimebase::is_initialized() || jam::gpu_native::GPUTimebase::initialize()) {
        std::cout << "✅ GPU Timebase ready" << std::endl;
        
        test_transport_operations();
        
        // Cleanup
        std::cout << "\nShutting down..." << std::endl;
        jam::gpu_native::GPUTimebase::shutdown();
    } else {
        std::cout << "❌ Failed to initialize GPU Timebase" << std::endl;
        return 1;
    }
    
    return 0;
}
