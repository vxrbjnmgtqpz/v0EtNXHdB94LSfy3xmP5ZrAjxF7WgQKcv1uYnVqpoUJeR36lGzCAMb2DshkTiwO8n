#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace TOAST;

int main() {
    std::cout << "🧪 Testing ClockDriftArbiter Implementation..." << std::endl;
    
    // Create two arbiters to simulate network peers
    ClockDriftArbiter arbiter1;
    ClockDriftArbiter arbiter2;
    
    // Set up callbacks
    arbiter1.onRoleChanged = [](ClockRole role, const std::string& masterId) {
        std::cout << "Arbiter1 role changed to: " 
                  << (role == ClockRole::MASTER ? "MASTER" : 
                      role == ClockRole::SLAVE ? "SLAVE" : "CANDIDATE") 
                  << " (master: " << masterId << ")" << std::endl;
    };
    
    arbiter2.onRoleChanged = [](ClockRole role, const std::string& masterId) {
        std::cout << "Arbiter2 role changed to: " 
                  << (role == ClockRole::MASTER ? "MASTER" : 
                      role == ClockRole::SLAVE ? "SLAVE" : "CANDIDATE") 
                  << " (master: " << masterId << ")" << std::endl;
    };
    
    arbiter1.onMasterElected = [](const std::string& masterId) {
        std::cout << "✅ Master elected: " << masterId << std::endl;
    };
    
    // Initialize arbiters
    if (!arbiter1.initialize("peer1", true)) {
        std::cerr << "❌ Failed to initialize arbiter1" << std::endl;
        return 1;
    }
    
    if (!arbiter2.initialize("peer2", true)) {
        std::cerr << "❌ Failed to initialize arbiter2" << std::endl;
        return 1;
    }
    
    // Add each other as peers
    arbiter1.addPeer("peer2", "127.0.0.1", 8001);
    arbiter2.addPeer("peer1", "127.0.0.1", 8000);
    
    std::cout << "\n🗳️ Starting master election..." << std::endl;
    arbiter1.startMasterElection();
    
    // Wait for election to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Test timing functions
    std::cout << "\n⏰ Testing timing functions..." << std::endl;
    uint64_t masterTime1 = arbiter1.getCurrentMasterTime();
    uint64_t masterTime2 = arbiter2.getCurrentMasterTime();
    uint64_t localTime1 = arbiter1.getLocalTimestamp();
    uint64_t localTime2 = arbiter2.getLocalTimestamp();
    
    std::cout << "Arbiter1 - Master time: " << masterTime1 << ", Local time: " << localTime1 << std::endl;
    std::cout << "Arbiter2 - Master time: " << masterTime2 << ", Local time: " << localTime2 << std::endl;
    
    // Test peer statistics
    std::cout << "\n📊 Peer statistics:" << std::endl;
    auto peers1 = arbiter1.getConnectedPeers();
    auto peers2 = arbiter2.getConnectedPeers();
    
    std::cout << "Arbiter1 connected peers: ";
    for (const auto& peer : peers1) {
        std::cout << peer << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Arbiter2 connected peers: ";
    for (const auto& peer : peers2) {
        std::cout << peer << " ";
    }
    std::cout << std::endl;
    
    // Test buffer recommendations
    std::cout << "\n🔧 Buffer recommendations:" << std::endl;
    std::cout << "Arbiter1 recommended buffer: " << arbiter1.getRecommendedBufferSize() << "μs" << std::endl;
    std::cout << "Arbiter2 recommended buffer: " << arbiter2.getRecommendedBufferSize() << "μs" << std::endl;
    
    // Test role assignment
    std::cout << "\n👑 Role assignments:" << std::endl;
    std::cout << "Arbiter1 role: " << (arbiter1.getCurrentRole() == ClockRole::MASTER ? "MASTER" : 
                                        arbiter1.getCurrentRole() == ClockRole::SLAVE ? "SLAVE" : "CANDIDATE") << std::endl;
    std::cout << "Arbiter2 role: " << (arbiter2.getCurrentRole() == ClockRole::MASTER ? "MASTER" : 
                                        arbiter2.getCurrentRole() == ClockRole::SLAVE ? "SLAVE" : "CANDIDATE") << std::endl;
    
    // Test manual master override
    std::cout << "\n🔄 Testing manual master override..." << std::endl;
    arbiter2.forceMasterRole();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "After override - Arbiter2 role: " << 
                 (arbiter2.getCurrentRole() == ClockRole::MASTER ? "MASTER" : 
                  arbiter2.getCurrentRole() == ClockRole::SLAVE ? "SLAVE" : "CANDIDATE") << std::endl;
    
    // Let it run for a bit to see synchronization in action
    std::cout << "\n🔄 Running synchronization for 3 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Check sync quality
    std::cout << "\n📈 Synchronization quality:" << std::endl;
    std::cout << "Arbiter1 sync quality: " << arbiter1.getSyncQuality() << std::endl;
    std::cout << "Arbiter2 sync quality: " << arbiter2.getSyncQuality() << std::endl;
    
    // Clean shutdown
    std::cout << "\n🛑 Shutting down..." << std::endl;
    arbiter1.shutdown();
    arbiter2.shutdown();
    
    std::cout << "✅ ClockDriftArbiter test completed successfully!" << std::endl;
    return 0;
}
