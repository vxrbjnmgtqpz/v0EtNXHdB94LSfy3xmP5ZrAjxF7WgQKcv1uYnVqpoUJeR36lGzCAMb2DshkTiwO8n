//
// test_gpu_bars_beats.cpp 
// Test GPU-native bars/beats calculation
//

#include "JAM_Framework_v2/include/gpu_native/gpu_timebase.h"
#include "JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "🧪 Testing GPU-Native Bars/Beats Calculation..." << std::endl;
    
    // Initialize GPU timebase
    if (!jam::gpu_native::GPUTimebase::initialize()) {
        std::cerr << "❌ Failed to initialize GPU timebase" << std::endl;
        return 1;
    }
    std::cout << "✅ GPU Timebase initialized" << std::endl;
    
    // Initialize GPU transport manager
    auto& transport = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transport.initialize()) {
        std::cerr << "❌ Failed to initialize GPU transport manager" << std::endl;
        return 1;
    }
    std::cout << "✅ GPU Transport Manager initialized" << std::endl;
    
    // Set time signature and BPM
    transport.setTimeSignature(4, 4);  // 4/4 time
    transport.setSubdivision(24);      // 24 ticks per quarter note (MIDI standard)
    transport.setBPM(120.0f);          // 120 BPM
    std::cout << "✅ Set 4/4 time signature, 24 PPQN, 120 BPM" << std::endl;
    
    // Start playing
    transport.play();
    std::cout << "▶️ Started playback" << std::endl;
    
    // Test for several seconds, showing bars/beats updates
    for (int i = 0; i < 20; ++i) {
        transport.update();
        
        // Get bars/beats info
        GPUBarsBeatsBuffer barsBeats = transport.getBarsBeatsInfo();
        float positionSeconds = transport.getPositionSeconds();
        
        // Format like a DAW
        printf("⏱️  Time: %06.3f sec | Bars/Beats: %03d.%02d.%03d | Total Beats: %.3f\n",
               positionSeconds,
               barsBeats.bars,
               barsBeats.beats, 
               barsBeats.subdivisions,
               barsBeats.total_beats);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    
    // Test pause/resume
    std::cout << "\n⏸️ Testing pause..." << std::endl;
    transport.pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    transport.update();
    GPUBarsBeatsBuffer pausedState = transport.getBarsBeatsInfo();
    printf("⏸️  Paused at: %03d.%02d.%03d\n", 
           pausedState.bars, pausedState.beats, pausedState.subdivisions);
    
    // Resume
    std::cout << "\n▶️ Resuming..." << std::endl;
    transport.play();
    
    // Test a few more updates
    for (int i = 0; i < 8; ++i) {
        transport.update();
        
        GPUBarsBeatsBuffer barsBeats = transport.getBarsBeatsInfo();
        float positionSeconds = transport.getPositionSeconds();
        
        printf("⏱️  Time: %06.3f sec | Bars/Beats: %03d.%02d.%03d\n",
               positionSeconds,
               barsBeats.bars,
               barsBeats.beats, 
               barsBeats.subdivisions);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    
    // Stop
    transport.stop();
    std::cout << "\n⏹️ Stopped transport" << std::endl;
    
    // Check stopped state
    transport.update();
    GPUBarsBeatsBuffer stoppedState = transport.getBarsBeatsInfo();
    printf("⏹️  Stopped at: %03d.%02d.%03d\n", 
           stoppedState.bars, stoppedState.beats, stoppedState.subdivisions);
    
    // Cleanup
    transport.shutdown();
    jam::gpu_native::GPUTimebase::shutdown();
    
    std::cout << "\n✅ GPU Bars/Beats test completed successfully!" << std::endl;
    return 0;
}
