/**
 * JAM Framework v2: Basic UDP Sender Example
 * 
 * Demonstrates fire-and-forget UDP multicast messaging
 * NO TCP/HTTP - pure UDP only
 */

#include "jam_transport.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

using namespace jam;

int main() {
    std::cout << "JAM Framework v2: UDP Sender Example" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "FIRE-AND-FORGET UDP MULTICAST - NO TCP/HTTP" << std::endl;
    std::cout << std::endl;
    
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
    
    // Send example JSONL messages
    std::vector<std::string> example_messages = {
        R"({"type":"midi","note_on":{"note":60,"velocity":127,"channel":0},"timestamp":1642789234567890})",
        R"({"type":"audio","samples":[0.1,0.2,0.3,0.4],"channels":2,"sample_rate":48000,"timestamp":1642789234567891})",
        R"({"type":"video","width":640,"height":480,"pixels":[255,128,64,255],"format":"rgba8","timestamp":1642789234567892})",
        R"({"type":"control","command":"sync","session_id":"abc123","timestamp":1642789234567893})"
    };
    
    std::cout << "Sending JSONL messages via fire-and-forget UDP..." << std::endl;
    std::cout << std::endl;
    
    for (size_t i = 0; i < example_messages.size(); i++) {
        const auto& message = example_messages[i];
        
        std::cout << "Message " << (i + 1) << ": ";
        
        // Send single packet
        uint64_t send_time = transport->send_immediate(message.data(), message.size());
        
        if (send_time > 0) {
            std::cout << "Sent in " << send_time << "μs - " << message.substr(0, 50) << "..." << std::endl;
        } else {
            std::cout << "FAILED - " << message.substr(0, 50) << "..." << std::endl;
        }
        
        // Wait between messages
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << std::endl;
    std::cout << "Testing burst transmission (for reliability)..." << std::endl;
    
    std::string burst_message = R"({"type":"midi","note_on":{"note":72,"velocity":100,"channel":1},"timestamp":1642789234567894})";
    
    // Send burst of 3 duplicate packets
    uint64_t burst_time = transport->send_burst(burst_message.data(), burst_message.size(), 3, 50);
    
    std::cout << "Burst sent (3 packets) in " << burst_time << "μs - " << burst_message.substr(0, 50) << "..." << std::endl;
    std::cout << std::endl;
    
    // Display statistics
    auto stats = transport->get_stats();
    std::cout << "Transport Statistics:" << std::endl;
    std::cout << "  Packets sent: " << stats.packets_sent << std::endl;
    std::cout << "  Bytes sent: " << stats.bytes_sent << std::endl;
    std::cout << "  Send failures: " << stats.send_failures << std::endl;
    std::cout << "  Avg send time: " << stats.avg_send_time_us << "μs" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Example complete!" << std::endl;
    std::cout << "Run the receiver example to see these messages arrive." << std::endl;
    
    return 0;
}
