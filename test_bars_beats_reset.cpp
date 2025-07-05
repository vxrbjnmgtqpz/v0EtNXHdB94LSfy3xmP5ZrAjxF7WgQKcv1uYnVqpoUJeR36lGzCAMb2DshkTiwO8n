#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include "JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"

int main() {
    std::cout << "🧪 Testing GPU Transport Bars/Beats Reset Bug Fix..." << std::endl;
    
    // Get the singleton instance and initialize
    auto& transport = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transport.initialize()) {
        std::cerr << "❌ Failed to initialize GPU Transport Manager" << std::endl;
        return 1;
    }
    
    // Get initial bars/beats state (should be 1.1.000)
    transport.update();  // Update to sync with GPU buffers
    auto initial_bars_beats = transport.getBarsBeatsInfo();
    std::cout << "📊 Initial state: " 
              << initial_bars_beats.bars << "." 
              << initial_bars_beats.beats << "." 
              << std::setfill('0') << std::setw(3) << initial_bars_beats.subdivisions << std::endl;
    
    // Start playback
    std::cout << "▶️ Starting playback..." << std::endl;
    transport.play();
    
    // Let it run for a bit to accumulate time
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    // Get bars/beats during playback
    transport.update();  // Update to sync with GPU buffers
    auto playing_bars_beats = transport.getBarsBeatsInfo();
    std::cout << "📊 During playback: " 
              << playing_bars_beats.bars << "." 
              << playing_bars_beats.beats << "." 
              << std::setfill('0') << std::setw(3) << playing_bars_beats.subdivisions << std::endl;
    
    // Stop playback
    std::cout << "⏹️ Stopping playback..." << std::endl;
    transport.stop();
    
    // Give a moment for the GPU buffer to update
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Get bars/beats after stop (should be reset to 1.1.000)
    transport.update();  // Update to sync with GPU buffers
    auto stopped_bars_beats = transport.getBarsBeatsInfo();
    std::cout << "📊 After stop: " 
              << stopped_bars_beats.bars << "." 
              << stopped_bars_beats.beats << "." 
              << std::setfill('0') << std::setw(3) << stopped_bars_beats.subdivisions << std::endl;
    
    // Test the fix: bars/beats should reset to 1.1.000 when stopped
    bool reset_successful = (stopped_bars_beats.bars == 1 && 
                           stopped_bars_beats.beats == 1 && 
                           stopped_bars_beats.subdivisions == 0);
    
    if (reset_successful) {
        std::cout << "✅ Bars/beats reset test PASSED! Values correctly reset to 1.1.000 on STOP." << std::endl;
    } else {
        std::cout << "❌ Bars/beats reset test FAILED! Values did not reset to 1.1.000 on STOP." << std::endl;
        std::cout << "   Expected: 1.1.000" << std::endl;
        std::cout << "   Actual: " << stopped_bars_beats.bars << "." 
                  << stopped_bars_beats.beats << "." 
                  << std::setfill('0') << std::setw(3) << stopped_bars_beats.subdivisions << std::endl;
    }
    
    // Test again with pause/stop to ensure consistency
    std::cout << "\n🧪 Testing pause/stop sequence..." << std::endl;
    
    // Start again
    transport.play();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Pause
    std::cout << "⏸️ Pausing..." << std::endl;
    transport.pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    transport.update();  // Update to sync with GPU buffers
    auto paused_bars_beats = transport.getBarsBeatsInfo();
    std::cout << "📊 During pause: " 
              << paused_bars_beats.bars << "." 
              << paused_bars_beats.beats << "." 
              << std::setfill('0') << std::setw(3) << paused_bars_beats.subdivisions << std::endl;
    
    // Stop from pause
    std::cout << "⏹️ Stopping from pause..." << std::endl;
    transport.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    transport.update();  // Update to sync with GPU buffers
    auto final_bars_beats = transport.getBarsBeatsInfo();
    std::cout << "📊 Final state: " 
              << final_bars_beats.bars << "." 
              << final_bars_beats.beats << "." 
              << std::setfill('0') << std::setw(3) << final_bars_beats.subdivisions << std::endl;
    
    bool final_reset_successful = (final_bars_beats.bars == 1 && 
                                 final_bars_beats.beats == 1 && 
                                 final_bars_beats.subdivisions == 0);
    
    if (final_reset_successful) {
        std::cout << "✅ Pause-to-stop reset test PASSED! Values correctly reset to 1.1.000." << std::endl;
    } else {
        std::cout << "❌ Pause-to-stop reset test FAILED!" << std::endl;
    }
    
    transport.shutdown();
    
    if (reset_successful && final_reset_successful) {
        std::cout << "\n🎉 All bars/beats reset tests PASSED! Bug fix verified." << std::endl;
        return 0;
    } else {
        std::cout << "\n💥 Some tests FAILED. Bug fix needs more work." << std::endl;
        return 1;
    }
}
