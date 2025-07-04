#include <iostream>
#include <thread>
#include <chrono>
#include "../../include/jam_toast.h"

using namespace jam;

int main() {
    std::cout << "JAM Framework v2 - TOAST Protocol Test" << std::endl;
    
    // Create TOAST protocol instance
    TOASTv2Protocol toast;
    
    // Set up callbacks
    toast.set_midi_callback([](const TOASTFrame& frame) {
        std::cout << "Received MIDI frame: " << frame.payload.size() << " bytes" << std::endl;
    });
    
    toast.set_audio_callback([](const TOASTFrame& frame) {
        std::cout << "Received Audio frame: " << frame.payload.size() << " bytes" << std::endl;
    });
    
    toast.set_sync_callback([](const TOASTFrame& frame) {
        std::cout << "Received Sync frame" << std::endl;
    });
    
    // Initialize protocol
    if (!toast.initialize("239.255.77.77", 7777, 12345)) {
        std::cerr << "Failed to initialize TOAST protocol" << std::endl;
        return 1;
    }
    
    // Start processing
    if (!toast.start_processing()) {
        std::cerr << "Failed to start TOAST processing" << std::endl;
        return 1;
    }
    
    std::cout << "TOAST protocol started on 239.255.77.77:7777" << std::endl;
    std::cout << "Session ID: " << toast.get_session_id() << std::endl;
    
    // Send some test messages
    for (int i = 0; i < 5; ++i) {
        // Send MIDI note
        std::vector<uint8_t> midi_data = {0x90, 0x60, 0x7F}; // Note on
        toast.send_midi(midi_data, 
                       std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()
                       ).count(), 
                       true); // Use burst
        
        // Send sync
        toast.send_sync(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count());
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Display statistics
    auto stats = toast.get_stats();
    std::cout << "\nTOAST Statistics:" << std::endl;
    std::cout << "Frames sent: " << stats.frames_sent << std::endl;
    std::cout << "Frames received: " << stats.frames_received << std::endl;
    std::cout << "Burst packets sent: " << stats.burst_packets_sent << std::endl;
    std::cout << "Duplicate packets: " << stats.duplicate_packets_received << std::endl;
    
    // Keep running for a bit to receive messages
    std::cout << "\nListening for 10 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    
    toast.shutdown();
    std::cout << "TOAST protocol test completed" << std::endl;
    
    return 0;
}
