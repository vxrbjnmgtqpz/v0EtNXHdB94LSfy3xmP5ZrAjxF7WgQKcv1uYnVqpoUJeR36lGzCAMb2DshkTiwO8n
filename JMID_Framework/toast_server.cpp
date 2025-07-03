#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace TOAST;

int main() {
    std::cout << "🖥️  TOAST TCP Server - Waiting for TOASTer connections" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    ConnectionManager server;
    ClockDriftArbiter arbiter;
    arbiter.initialize("master-server", true);
    
    // Message statistics
    int messagesReceived = 0;
    int messagesSent = 0;
    
    // Set up message handler
    server.setMessageHandler([&](std::unique_ptr<TransportMessage> message) {
        messagesReceived++;
        
        std::cout << "📥 [" << messagesReceived << "] Received " 
                  << (message->getType() == MessageType::MIDI ? "MIDI" : 
                      message->getType() == MessageType::HEARTBEAT ? "HEARTBEAT" : "OTHER")
                  << " message" << std::endl;
        
        if (message->getType() == MessageType::MIDI) {
            std::cout << "🎵 MIDI Data: " << message->getPayload().substr(0, 80) << std::endl;
        }
        
        // Send acknowledgment
        auto ack = std::make_unique<TransportMessage>(
            MessageType::METADATA,
            "{\"type\":\"ack\",\"messageId\":" + std::to_string(message->getSequenceNumber()) + 
            ",\"timestamp\":" + std::to_string(arbiter.getCurrentMasterTime()) + "}",
            arbiter.getCurrentMasterTime(),
            ++messagesSent
        );
        
        server.sendMessage(std::move(ack));
        std::cout << "📤 Sent acknowledgment #" << messagesSent << std::endl;
    });
    
    if (server.startServer(8080)) {
        std::cout << "✅ TOAST server started on port 8080" << std::endl;
        std::cout << "⏳ Press Ctrl+C to stop server..." << std::endl;
        
        // Keep server running and show statistics
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            auto clients = server.getConnectedClients();
            std::cout << "📊 Status: " << clients.size() << " clients, " 
                      << messagesReceived << " received, " << messagesSent << " sent" << std::endl;
        }
    } else {
        std::cout << "❌ Failed to start TCP server on port 8080" << std::endl;
        return 1;
    }
    
    return 0;
}
