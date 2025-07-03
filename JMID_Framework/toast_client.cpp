#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace TOAST;

int main(int argc, char* argv[]) {
    std::string serverIP = "127.0.0.1";
    if (argc > 1) {
        serverIP = argv[1];
    }
    
    std::cout << "💻 TOAST TCP Client - Connecting to TOASTer Server" << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << "🔗 Connecting to: " << serverIP << ":8080" << std::endl;
    
    ConnectionManager client;
    ClockDriftArbiter arbiter;
    arbiter.initialize("client-device", false);
    
    // Message statistics
    int messagesSent = 0;
    int messagesReceived = 0;
    
    // Set up message handler
    client.setMessageHandler([&](std::unique_ptr<TransportMessage> message) {
        messagesReceived++;
        std::cout << "📥 [" << messagesReceived << "] Received response: " 
                  << message->getPayload().substr(0, 50) << "..." << std::endl;
    });
    
    if (client.connectToServer(serverIP, 8080)) {
        std::cout << "✅ Connected to TOAST server!" << std::endl;
        std::cout << "🎵 Sending test MIDI sequence..." << std::endl;
        
        // Send a sequence of MIDI messages simulating real usage
        const char* notes[] = {"C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"};
        int noteValues[] = {60, 62, 64, 65, 67, 69, 71, 72};
        
        for (int i = 0; i < 8; ++i) {
            // Note On
            auto noteOn = std::make_unique<TransportMessage>(
                MessageType::MIDI,
                "{\"type\":\"noteOn\",\"note\":" + std::to_string(noteValues[i]) + 
                ",\"velocity\":100,\"channel\":1,\"noteName\":\"" + notes[i] + "\"}",
                arbiter.getCurrentMasterTime(),
                ++messagesSent
            );
            
            client.sendMessage(std::move(noteOn));
            std::cout << "📤 [" << messagesSent << "] Sent noteOn: " << notes[i] 
                      << " (" << noteValues[i] << ")" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Note Off
            auto noteOff = std::make_unique<TransportMessage>(
                MessageType::MIDI,
                "{\"type\":\"noteOff\",\"note\":" + std::to_string(noteValues[i]) + 
                ",\"velocity\":0,\"channel\":1,\"noteName\":\"" + notes[i] + "\"}",
                arbiter.getCurrentMasterTime(),
                ++messagesSent
            );
            
            client.sendMessage(std::move(noteOff));
            std::cout << "📤 [" << messagesSent << "] Sent noteOff: " << notes[i] 
                      << " (" << noteValues[i] << ")" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        // Send a few heartbeats
        for (int i = 0; i < 3; ++i) {
            auto heartbeat = std::make_unique<TransportMessage>(
                MessageType::HEARTBEAT,
                "{\"type\":\"heartbeat\",\"clientId\":\"test-client\",\"timestamp\":" + 
                std::to_string(arbiter.getCurrentMasterTime()) + "}",
                arbiter.getCurrentMasterTime(),
                ++messagesSent
            );
            
            client.sendMessage(std::move(heartbeat));
            std::cout << "💓 [" << messagesSent << "] Sent heartbeat" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // Wait for final responses
        std::cout << "⏳ Waiting for final responses..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        std::cout << "\n📊 Final Statistics:" << std::endl;
        std::cout << "   Messages sent: " << messagesSent << std::endl;
        std::cout << "   Responses received: " << messagesReceived << std::endl;
        std::cout << "   Success rate: " << (messagesReceived * 100 / messagesSent) << "%" << std::endl;
        
    } else {
        std::cout << "❌ Failed to connect to server at " << serverIP << ":8080" << std::endl;
        std::cout << "💡 Make sure the TOAST server is running first" << std::endl;
        return 1;
    }
    
    std::cout << "🛑 Client test complete" << std::endl;
    return 0;
}
