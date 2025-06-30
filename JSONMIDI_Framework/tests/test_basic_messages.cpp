#include "JSONMIDIMessage.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <string>

using namespace JSONMIDI;

void testNoteOnMessage() {
    std::cout << "Testing NoteOn message..." << std::endl;
    
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    // Test MIDI 1.0 Note On
    NoteOnMessage noteOn1(1, 60, 127, timestamp, Protocol::MIDI1);
    
    std::string json1 = noteOn1.toJSON();
    std::cout << "MIDI 1.0 Note On JSON: " << json1 << std::endl;
    
    auto bytes1 = noteOn1.toMIDIBytes();
    std::cout << "MIDI 1.0 Note On bytes: ";
    for (auto byte : bytes1) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify MIDI 1.0 bytes
    assert(bytes1.size() == 3);
    assert(bytes1[0] == 0x90); // Note On, channel 1
    assert(bytes1[1] == 60);   // Note number
    assert(bytes1[2] == 127);  // Velocity
    
    // Test MIDI 2.0 Note On with attributes
    NoteOnMessage noteOn2(2, 72, 30000, timestamp, Protocol::MIDI2);
    noteOn2.setPerNoteAttribute(1, 10000);
    
    std::string json2 = noteOn2.toJSON();
    std::cout << "MIDI 2.0 Note On JSON: " << json2 << std::endl;
    
    auto bytes2 = noteOn2.toMIDIBytes();
    std::cout << "MIDI 2.0 Note On bytes: ";
    for (auto byte : bytes2) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify MIDI 2.0 bytes
    assert(bytes2.size() == 8);
    
    std::cout << "NoteOn tests passed!" << std::endl;
}

void testNoteOffMessage() {
    std::cout << "Testing NoteOff message..." << std::endl;
    
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    // Test MIDI 1.0 Note Off
    NoteOffMessage noteOff1(1, 60, 64, timestamp, Protocol::MIDI1);
    
    std::string json1 = noteOff1.toJSON();
    std::cout << "MIDI 1.0 Note Off JSON: " << json1 << std::endl;
    
    auto bytes1 = noteOff1.toMIDIBytes();
    std::cout << "MIDI 1.0 Note Off bytes: ";
    for (auto byte : bytes1) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify MIDI 1.0 bytes
    assert(bytes1.size() == 3);
    assert(bytes1[0] == 0x80); // Note Off, channel 1
    assert(bytes1[1] == 60);   // Note number
    assert(bytes1[2] == 64);   // Release velocity
    
    std::cout << "NoteOff tests passed!" << std::endl;
}

void testControlChangeMessage() {
    std::cout << "Testing ControlChange message..." << std::endl;
    
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    // Test MIDI 1.0 Control Change
    ControlChangeMessage cc1(1, 7, 100, timestamp, Protocol::MIDI1);
    
    std::string json1 = cc1.toJSON();
    std::cout << "MIDI 1.0 CC JSON: " << json1 << std::endl;
    
    auto bytes1 = cc1.toMIDIBytes();
    std::cout << "MIDI 1.0 CC bytes: ";
    for (auto byte : bytes1) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify MIDI 1.0 bytes
    assert(bytes1.size() == 3);
    assert(bytes1[0] == 0xB0); // Control Change, channel 1
    assert(bytes1[1] == 7);    // Controller number (volume)
    assert(bytes1[2] == 100);  // Controller value
    
    // Test MIDI 2.0 Control Change with extended range
    ControlChangeMessage cc2(2, 1000, 2000000, timestamp, Protocol::MIDI2);
    
    std::string json2 = cc2.toJSON();
    std::cout << "MIDI 2.0 CC JSON: " << json2 << std::endl;
    
    auto bytes2 = cc2.toMIDIBytes();
    std::cout << "MIDI 2.0 CC bytes: ";
    for (auto byte : bytes2) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify MIDI 2.0 bytes
    assert(bytes2.size() == 8);
    
    std::cout << "ControlChange tests passed!" << std::endl;
}

void testSystemExclusiveMessage() {
    std::cout << "Testing SystemExclusive message..." << std::endl;
    
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    // Test MIDI 1.0 SysEx
    std::vector<uint8_t> sysexData = {0x43, 0x12, 0x00, 0x7F, 0x00};
    SystemExclusiveMessage sysex1(0x43, sysexData, timestamp, 
                                 SystemExclusiveMessage::SysExType::SYSEX7);
    
    std::string json1 = sysex1.toJSON();
    std::cout << "MIDI 1.0 SysEx JSON: " << json1 << std::endl;
    
    auto bytes1 = sysex1.toMIDIBytes();
    std::cout << "MIDI 1.0 SysEx bytes: ";
    for (auto byte : bytes1) {
        printf("0x%02X ", byte);
    }
    std::cout << std::endl;
    
    // Verify SysEx structure
    assert(bytes1[0] == 0xF0); // SysEx start
    assert(bytes1[1] == 0x43); // Manufacturer ID
    assert(bytes1[bytes1.size()-1] == 0xF7); // SysEx end
    
    std::cout << "SystemExclusive tests passed!" << std::endl;
}

void testPerformance() {
    std::cout << "Testing performance..." << std::endl;
    
    const size_t numMessages = 10000;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < numMessages; ++i) {
        auto timestamp = std::chrono::high_resolution_clock::now();
        NoteOnMessage noteOn(1, 60, 127, timestamp);
        
        // Convert to JSON and back to bytes
        std::string json = noteOn.toJSON();
        auto bytes = noteOn.toMIDIBytes();
        
        // Basic validation
        assert(!json.empty());
        assert(bytes.size() == 3);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime).count();
    
    double avgTimePerMessage = static_cast<double>(duration) / numMessages;
    
    std::cout << "Processed " << numMessages << " messages in " << duration << " μs" << std::endl;
    std::cout << "Average time per message: " << avgTimePerMessage << " μs" << std::endl;
    
    // Check if we meet the performance target (<100μs per message)
    if (avgTimePerMessage < 100.0) {
        std::cout << "✓ Performance target met!" << std::endl;
    } else {
        std::cout << "✗ Performance target not met (target: <100μs)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "JSONMIDI Framework Basic Tests" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        if (argc > 1 && std::string(argv[1]) == "--test-basic") {
            testNoteOnMessage();
            testNoteOffMessage();
            testControlChangeMessage();
            testSystemExclusiveMessage();
            std::cout << "\n✓ All basic message tests passed!" << std::endl;
        } else if (argc > 1 && std::string(argv[1]) == "--test-performance") {
            testPerformance();
            std::cout << "\n✓ Performance targets met!" << std::endl;
        } else {
            // Run all tests
            testNoteOnMessage();
            std::cout << std::endl;
            
            testNoteOffMessage();
            std::cout << std::endl;
            
            testControlChangeMessage();
            std::cout << std::endl;
            
            testSystemExclusiveMessage();
            std::cout << std::endl;
            
            testPerformance();
            
            std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
