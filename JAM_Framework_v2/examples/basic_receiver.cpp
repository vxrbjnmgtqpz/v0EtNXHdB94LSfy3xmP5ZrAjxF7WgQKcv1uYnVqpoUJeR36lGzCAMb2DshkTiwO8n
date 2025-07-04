/**
 * JAM Framework v2: Basic UDP Receiver Example
 * 
 * Demonstrates UDP multicast message reception
 * NO TCP/HTTP - pure UDP only
 */

#include "jam_transport.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

using namespace jam;

std::atomic<bool> running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    running.store(false);
}

int main() {
    std::cout << "JAM Framework v2: UDP Receiver Example" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "UDP MULTICAST RECEIVER - NO TCP/HTTP" << std::endl;
    std::cout << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create UDP transport
    auto transport = UDPTransport::create("239.255.77.77", 7777);
    
    if (!transport) {
        std::cerr << "Failed to create UDP transport!" << std::endl;
        return 1;
    }
    
    std::cout << "Created UDP multicast transport:" << std::endl;
    std::cout << "  Group: " << transport->get_multicast_group() << std::endl;
    std::cout << "  Port: " << transport->get_port() << std::endl;
    std::cout << std::endl;
    
    // Set up message receiver callback
    std::atomic<uint32_t> message_count{0};
    
    transport->set_receive_callback([&](std::span<const uint8_t> data) {
        uint32_t msg_num = message_count.fetch_add(1) + 1;
        
        // Convert to string for display
        std::string message(reinterpret_cast<const char*>(data.data()), data.size());
        
        auto now = std::chrono::high_resolution_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        
        std::cout << "[" << timestamp << "] Message " << msg_num << " (" << data.size() << " bytes): ";
        
        // Show first 80 characters of message
        if (message.length() > 80) {
            std::cout << message.substr(0, 80) << "..." << std::endl;
        } else {
            std::cout << message << std::endl;
        }
    });
    
    // Start receiving
    if (!transport->start_receiving()) {
        std::cerr << "Failed to start UDP receiving!" << std::endl;
        return 1;
    }
    
    std::cout << "Started UDP multicast receiver." << std::endl;
    std::cout << "Listening for JSONL messages..." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;
    
    // Main loop
    auto last_stats_time = std::chrono::steady_clock::now();
    uint32_t last_message_count = 0;
    
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Print statistics every 5 seconds
        auto now = std::chrono::steady_clock::now();
        if (now - last_stats_time >= std::chrono::seconds(5)) {
            auto stats = transport->get_stats();
            uint32_t current_count = message_count.load();
            uint32_t new_messages = current_count - last_message_count;
            
            std::cout << "--- Statistics (last 5s) ---" << std::endl;
            std::cout << "  New messages: " << new_messages << std::endl;
            std::cout << "  Total packets: " << stats.packets_received << std::endl;
            std::cout << "  Total bytes: " << stats.bytes_received << std::endl;
            std::cout << "  Estimated loss: " << stats.estimated_packet_loss_percent << "%" << std::endl;
            std::cout << std::endl;
            
            last_stats_time = now;
            last_message_count = current_count;
        }
    }
    
    // Stop receiving
    transport->stop_receiving();
    
    // Final statistics
    auto final_stats = transport->get_stats();
    uint32_t final_count = message_count.load();
    
    std::cout << std::endl;
    std::cout << "Final Statistics:" << std::endl;
    std::cout << "  Messages processed: " << final_count << std::endl;
    std::cout << "  Packets received: " << final_stats.packets_received << std::endl;
    std::cout << "  Bytes received: " << final_stats.bytes_received << std::endl;
    std::cout << "  Estimated loss: " << final_stats.estimated_packet_loss_percent << "%" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Receiver shutdown complete." << std::endl;
    
    return 0;
}
