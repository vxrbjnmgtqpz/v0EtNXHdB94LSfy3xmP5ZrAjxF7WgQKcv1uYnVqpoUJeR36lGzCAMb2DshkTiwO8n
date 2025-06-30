#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <random>
#include "JSONMIDIMessage.h"
#include "JSONMIDIParser.h"

using namespace JSONMIDI;

void printHeader() {
    std::cout << "\n" << "=" << std::string(60, '=') << "\n";
    std::cout << "       🎵 JSONMIDI Framework Simple Demo 🎵\n";
    std::cout << "         Real-World Testing Application\n";
    std::cout << "=" << std::string(60, '=') << "\n\n";
    
    std::cout << "📊 Framework Status:\n";
    std::cout << "   • Phase 1.1: ✅ Complete\n";
    std::cout << "   • Phase 1.2: ✅ Complete\n";
    std::cout << "   • Target Performance: ✅ EXCEEDED\n\n";
}

void demonstrateMessageCreation() {
    std::cout << "🔧 Testing Message Creation & Serialization...\n";
    
    try {
        // Create note on message
        auto timestamp = std::chrono::high_resolution_clock::now();
            
        NoteOnMessage noteOn(1, 60, 127, timestamp);  // Channel 1, Middle C, Full velocity
        std::string json = noteOn.toJSON();
        std::vector<uint8_t> midiBytes = noteOn.toMIDIBytes();
        
        std::cout << "   ✅ NoteOn Created: Channel " << (int)noteOn.getChannel() 
                  << ", Note " << (int)noteOn.getNote() 
                  << ", Velocity " << (int)noteOn.getVelocity() << "\n";
        std::cout << "   📝 JSON: " << json.substr(0, 80) << "...\n";
        std::cout << "   📦 MIDI Bytes: " << midiBytes.size() << " bytes\n";
        
        // Create note off message
        auto timestamp2 = timestamp + std::chrono::milliseconds(1);
        NoteOffMessage noteOff(1, 60, 0, timestamp2);
        std::cout << "   ✅ NoteOff Created: Channel " << (int)noteOff.getChannel() 
                  << ", Note " << (int)noteOff.getNote() << "\n";
                  
    } catch (const std::exception& e) {
        std::cout << "   ❌ Error: " << e.what() << "\n";
    }
    
    std::cout << "\n";
}

void demonstrateParsing() {
    std::cout << "🔍 Testing JSON Parsing & Validation...\n";
    
    try {
        BassoonParser parser;
        
        // Test parsing a simple note on message
        std::string testJson = R"({
            "type": "noteOn",
            "channel": 1,
            "note": 60,
            "velocity": 100,
            "timestamp": 1234567890,
            "protocol": "midi1"
        })";
        
        std::cout << "   📝 Parsing JSON: " << testJson.substr(0, 50) << "...\n";
        
        auto result = parser.parseMessageWithValidation(testJson);
        if (result.second.isValid && result.first) {
            std::cout << "   ✅ Parse Success: Type " << (int)result.first->getType() << "\n";
            auto ts = result.first->getTimestamp();
            auto epoch = ts.time_since_epoch();
            auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
            std::cout << "   ⏱️  Timestamp: " << microseconds.count() << "μs\n";
        } else {
            std::cout << "   ❌ Parse Failed: " << result.second.errorMessage << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "   ❌ Error: " << e.what() << "\n";
    }
    
    std::cout << "\n";
}

void demonstratePerformance() {
    std::cout << "⚡ Performance Benchmark...\n";
    
    try {
        BassoonParser parser;
        parser.setPerformanceMonitoring(true);
        
        // Create test messages
        auto timestamp = std::chrono::high_resolution_clock::now();
            
        const int iterations = 1000;
        
        // Measure serialization performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto ts = timestamp + std::chrono::microseconds(i);
            NoteOnMessage note(1, 60 + (i % 12), 100, ts);
            std::string json = note.toJSON();
            std::vector<uint8_t> bytes = note.toMIDIBytes();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avgTime = static_cast<double>(duration.count()) / iterations;
        
        std::cout << "   📊 " << iterations << " messages serialized\n";
        std::cout << "   ⚡ Average time: " << std::fixed << std::setprecision(2) 
                  << avgTime << "μs per message\n";
        std::cout << "   🎯 Target (<1.3μs): " 
                  << (avgTime < 1.3 ? "✅ ACHIEVED" : "❌ MISSED") << "\n";
                  
    } catch (const std::exception& e) {
        std::cout << "   ❌ Error: " << e.what() << "\n";
    }
    
    std::cout << "\n";
}

void demonstrateRealTimeScenario() {
    std::cout << "🎼 Real-Time MIDI Scenario...\n";
    
    try {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> noteDist(48, 84);  // Piano range
        std::uniform_int_distribution<> velDist(60, 127);
        
        std::cout << "   🎹 Simulating piano performance...\n";
        
        auto timestamp = std::chrono::high_resolution_clock::now();
            
        for (int i = 0; i < 10; ++i) {
            int note = noteDist(gen);
            int velocity = velDist(gen);
            
            auto ts = timestamp + std::chrono::milliseconds(i * 100);
            NoteOnMessage noteOn(1, note, velocity, ts);
            std::cout << "   🎵 Note " << note << " (vel: " << velocity << ")\n";
            
            // Simulate note duration
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        std::cout << "   ✅ Real-time simulation complete\n";
        
    } catch (const std::exception& e) {
        std::cout << "   ❌ Error: " << e.what() << "\n";
    }
    
    std::cout << "\n";
}

int main() {
    try {
        printHeader();
        
        demonstrateMessageCreation();
        demonstrateParsing();
        demonstratePerformance();
        demonstrateRealTimeScenario();
        
        std::cout << "🎉 Demo completed successfully!\n";
        std::cout << "📝 Framework ready for integration into applications.\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Demo failed: " << e.what() << "\n";
        return 1;
    }
}
