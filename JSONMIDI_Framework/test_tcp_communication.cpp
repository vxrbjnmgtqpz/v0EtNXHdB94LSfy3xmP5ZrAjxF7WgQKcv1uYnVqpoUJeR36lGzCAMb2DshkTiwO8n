#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace TOAST;

void runServer() {
    std::cout << "🖥️  Starting TOAST TCP Server Test" << std::endl;
    
    ConnectionManager server;
    ClockDriftArbiter arbiter;
    arbiter.initialize("server", true);  // Allow master role
    
    // Set up message handler
    server.setMessageHandler([&](std::unique_ptr<TransportMessage> message) {
        std::cout << "📥 Server received message: " << message->getPayload().substr(0, 50) << "..." << std::endl;
        
        // Echo back a response
        auto response = std::make_unique<TransportMessage>(
            MessageType::MIDI,
            "{\"type\":\"response\",\"echo\":\"Server received your message\"}",
            arbiter.getCurrentMasterTime(),
            message->getSequenceNumber() + 1000
        );
        
        server.sendMessage(std::move(response));
        std::cout << "📤 Server sent response" << std::endl;
    });
    
    if (server.startServer(8081)) {
        std::cout << "✅ TCP server started on port 8081" << std::endl;
        std::cout << "⏳ Waiting for client connections..." << std::endl;
        
        // Keep server running for 30 seconds
        for (int i = 0; i < 30; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto clients = server.getConnectedClients();
            if (!clients.empty()) {
                std::cout << "🔗 Clients connected: " << clients.size() << std::endl;
                
                // Send a heartbeat message
                auto heartbeat = std::make_unique<TransportMessage>(
                    MessageType::HEARTBEAT,
                    "{\"type\":\"heartbeat\",\"timestamp\":" + std::to_string(arbiter.getCurrentMasterTime()) + "}",
                    arbiter.getCurrentMasterTime(),
                    i
                );
                
                server.sendMessage(std::move(heartbeat));
            }
        }
    } else {
        std::cout << "❌ Failed to start TCP server" << std::endl;
    }
    
    std::cout << "🛑 Server test complete" << std::endl;
}

void runClient() {
    std::cout << "💻 Starting TOAST TCP Client Test" << std::endl;
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    ConnectionManager client;
    ClockDriftArbiter arbiter;
    arbiter.initialize("client", false);  // Don't allow master role
    
    // Set up message handler
    client.setMessageHandler([](std::unique_ptr<TransportMessage> message) {
        std::cout << "📥 Client received message: " << message->getPayload().substr(0, 50) << "..." << std::endl;
    });
    
    if (client.connectToServer("127.0.0.1", 8081)) {
        std::cout << "✅ TCP client connected to server" << std::endl;
        
        // Send test messages
        for (int i = 0; i < 5; ++i) {
            auto message = std::make_unique<TransportMessage>(
                MessageType::MIDI,
                "{\"type\":\"noteOn\",\"note\":" + std::to_string(60 + i) + ",\"velocity\":100,\"channel\":1}",
                arbiter.getCurrentMasterTime(),
                i
            );
            
            client.sendMessage(std::move(message));
            std::cout << "📤 Client sent MIDI message " << i + 1 << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        // Wait for responses
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
    } else {
        std::cout << "❌ Failed to connect to TCP server" << std::endl;
    }
    
    std::cout << "🛑 Client test complete" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "client") {
        runClient();
    } else if (argc > 1 && std::string(argv[1]) == "server") {
        runServer();
    } else {
        std::cout << "🚀 TOAST TCP Communication Test" << std::endl;
        std::cout << "===============================\n" << std::endl;
        
        std::cout << "Starting server and client in separate threads..." << std::endl;
        
        // Run both in same process for testing
        std::thread serverThread(runServer);
        std::thread clientThread(runClient);
        
        serverThread.join();
        clientThread.join();
        
        std::cout << "\n🎉 TCP Communication Test Complete!" << std::endl;
    }
    
    return 0;
}
