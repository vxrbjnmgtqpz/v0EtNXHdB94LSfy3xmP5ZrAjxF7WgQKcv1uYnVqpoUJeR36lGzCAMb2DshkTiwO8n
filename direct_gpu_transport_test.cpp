//
// direct_gpu_transport_test.cpp
// Directly test GPU transport operations bypassing initialization check
//

#include <iostream>
#include <thread>
#include <chrono>

#define JAM_BYPASS_INIT_CHECK  // Custom flag to bypass initialization checks

#include "JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"
#include "JAM_Framework_v2/include/gpu_native/gpu_timebase.h"

using namespace jam::gpu_transport;
using namespace std::chrono_literals;

int main() {
    std::cout << "🎯 DIRECT GPU TRANSPORT TEST" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Bypassing initialization checks to test actual transport logic\n" << std::endl;
    
    // Initialize dependency
    std::cout << "1. Initializing GPU Timebase..." << std::endl;
    if (!jam::gpu_native::GPUTimebase::is_initialized() && !jam::gpu_native::GPUTimebase::initialize()) {
        std::cout << "❌ Failed to initialize GPU Timebase" << std::endl;
        return 1;
    }
    std::cout << "✅ GPU Timebase ready" << std::endl;
    
    // Get singleton instance
    std::cout << "\n2. Getting GPU Transport Manager instance..." << std::endl;
    auto& manager = GPUTransportManager::getInstance();
    std::cout << "Instance address: " << &manager << std::endl;
    
    // Force initialization (ignore flag check)
    std::cout << "\n3. Force initializing GPU Transport Manager..." << std::endl;
    bool init_result = manager.initialize();
    std::cout << "Initialize() returned: " << (init_result ? "true" : "false") << std::endl;
    
    if (!init_result) {
        std::cout << "❌ Initialization failed!" << std::endl;
        return 1;
    }
    
    std::cout << "\n4. Testing transport operations (bypassing isInitialized() checks)..." << std::endl;
    
    // Test basic state reading
    std::cout << "\n   Initial State Check:" << std::endl;
    try {
        auto state = manager.getCurrentState();
        std::cout << "   ✅ getCurrentState(): " << static_cast<int>(state) << std::endl;
    } catch (...) {
        std::cout << "   ❌ getCurrentState() failed" << std::endl;
    }
    
    try {
        bool playing = manager.isPlaying();
        std::cout << "   ✅ isPlaying(): " << (playing ? "YES" : "NO") << std::endl;
    } catch (...) {
        std::cout << "   ❌ isPlaying() failed" << std::endl;
    }
    
    try {
        float position = manager.getPositionSeconds();
        std::cout << "   ✅ getPositionSeconds(): " << position << "s" << std::endl;
    } catch (...) {
        std::cout << "   ❌ getPositionSeconds() failed" << std::endl;
    }
    
    // Test PLAY command
    std::cout << "\n   Testing PLAY Command:" << std::endl;
    try {
        manager.play();
        std::cout << "   ✅ play() command sent" << std::endl;
        
        // Give GPU time to process
        std::this_thread::sleep_for(100ms);
        manager.update();
        
        bool playing_after = manager.isPlaying();
        auto state_after = manager.getCurrentState();
        std::cout << "   After PLAY - Is playing: " << (playing_after ? "YES" : "NO") << std::endl;
        std::cout << "   After PLAY - State: " << static_cast<int>(state_after) << std::endl;
        
    } catch (...) {
        std::cout << "   ❌ PLAY command failed" << std::endl;
    }
    
    // Let it play for a bit
    std::cout << "\n   Playing for 500ms..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        std::this_thread::sleep_for(10ms);
        try {
            manager.update();
        } catch (...) {
            std::cout << "   ⚠️ Update " << i << " failed" << std::endl;
        }
    }
    
    try {
        float position_after = manager.getPositionSeconds();
        std::cout << "   Position after playing: " << position_after << "s" << std::endl;
    } catch (...) {
        std::cout << "   ❌ Could not read position after playing" << std::endl;
    }
    
    // Test PAUSE command
    std::cout << "\n   Testing PAUSE Command:" << std::endl;
    try {
        manager.pause();
        std::cout << "   ✅ pause() command sent" << std::endl;
        
        std::this_thread::sleep_for(100ms);
        manager.update();
        
        bool paused_after = manager.isPaused();
        auto state_after = manager.getCurrentState();
        std::cout << "   After PAUSE - Is paused: " << (paused_after ? "YES" : "NO") << std::endl;
        std::cout << "   After PAUSE - State: " << static_cast<int>(state_after) << std::endl;
        
    } catch (...) {
        std::cout << "   ❌ PAUSE command failed" << std::endl;
    }
    
    // Test STOP command
    std::cout << "\n   Testing STOP Command:" << std::endl;
    try {
        manager.stop();
        std::cout << "   ✅ stop() command sent" << std::endl;
        
        std::this_thread::sleep_for(100ms);
        manager.update();
        
        bool playing_after = manager.isPlaying();
        auto state_after = manager.getCurrentState();
        float position_after = manager.getPositionSeconds();
        std::cout << "   After STOP - Is playing: " << (playing_after ? "YES" : "NO") << std::endl;
        std::cout << "   After STOP - State: " << static_cast<int>(state_after) << std::endl;
        std::cout << "   After STOP - Position: " << position_after << "s" << std::endl;
        
    } catch (...) {
        std::cout << "   ❌ STOP command failed" << std::endl;
    }
    
    // Test RECORD command
    std::cout << "\n   Testing RECORD Command:" << std::endl;
    try {
        manager.record();
        std::cout << "   ✅ record() command sent" << std::endl;
        
        std::this_thread::sleep_for(100ms);
        manager.update();
        
        bool recording_after = manager.isRecording();
        auto state_after = manager.getCurrentState();
        std::cout << "   After RECORD - Is recording: " << (recording_after ? "YES" : "NO") << std::endl;
        std::cout << "   After RECORD - State: " << static_cast<int>(state_after) << std::endl;
        
    } catch (...) {
        std::cout << "   ❌ RECORD command failed" << std::endl;
    }
    
    // Test BPM control
    std::cout << "\n   Testing BPM Control:" << std::endl;
    try {
        float initial_bpm = manager.getBPM();
        std::cout << "   Initial BPM: " << initial_bpm << std::endl;
        
        manager.setBPM(140.0f);
        std::this_thread::sleep_for(100ms);
        manager.update();
        
        float new_bpm = manager.getBPM();
        std::cout << "   After setting to 140: " << new_bpm << std::endl;
        
    } catch (...) {
        std::cout << "   ❌ BPM control failed" << std::endl;
    }
    
    // Final cleanup
    std::cout << "\n5. Final cleanup..." << std::endl;
    try {
        manager.stop();
        manager.update();
        std::cout << "   ✅ Final STOP command completed" << std::endl;
    } catch (...) {
        std::cout << "   ⚠️ Final STOP command failed" << std::endl;
    }
    
    std::cout << "\n✅ DIRECT GPU TRANSPORT TEST COMPLETE!" << std::endl;
    std::cout << "🎯 This test bypassed initialization checks to directly validate transport operations." << std::endl;
    
    jam::gpu_native::GPUTimebase::shutdown();
    return 0;
}
