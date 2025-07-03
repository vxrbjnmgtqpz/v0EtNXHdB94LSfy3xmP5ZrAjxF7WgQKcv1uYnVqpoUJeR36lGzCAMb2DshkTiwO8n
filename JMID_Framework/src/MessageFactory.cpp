// MessageFactory implementation - Phase 1.2: Factory for creating MIDI messages
#include "JMIDParser.h"
#include <nlohmann/json.hpp>
#include <iostream>

namespace JMID {

std::unique_ptr<MIDIMessage> MessageFactory::createFromJSON(const std::string& json) {
    try {
        nlohmann::json j = nlohmann::json::parse(json);
        
        std::string type = j["type"];
        // Use current time as timestamp for simplicity
        auto timestamp = std::chrono::steady_clock::now();
        
        if (type == "note_on") {
            return std::make_unique<NoteOnMessage>(
                j["channel"].get<uint8_t>(),
                j["note"].get<uint8_t>(),
                j["velocity"].get<uint8_t>(),
                timestamp
            );
        } else if (type == "note_off") {
            return std::make_unique<NoteOffMessage>(
                j["channel"].get<uint8_t>(),
                j["note"].get<uint8_t>(),
                j["velocity"].get<uint8_t>(),
                timestamp
            );
        } else if (type == "control_change") {
            return std::make_unique<ControlChangeMessage>(
                j["channel"].get<uint8_t>(),
                j["controller"].get<uint8_t>(),
                j["value"].get<uint8_t>(),
                timestamp
            );
        } else if (type == "system_exclusive") {
            std::vector<uint8_t> data;
            for (const auto& element : j["data"]) {
                data.push_back(element.get<uint8_t>());
            }
            
            uint32_t manufacturerId = 0x00;
            if (j.contains("manufacturerId")) {
                manufacturerId = j["manufacturerId"].get<uint32_t>();
            }
            
            return std::make_unique<SystemExclusiveMessage>(manufacturerId, data, timestamp);
        }
        
        return nullptr;
        
    } catch (const std::exception& e) {
        // For debugging
        std::cerr << "MessageFactory error: " << e.what() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<MIDIMessage> MessageFactory::createFromMIDIBytes(
    const std::vector<uint8_t>& bytes, Timestamp timestamp) {
    
    if (bytes.empty()) {
        return nullptr;
    }
    
    uint8_t status = bytes[0];
    
    // Extract channel and message type
    uint8_t channel = status & 0x0F;
    uint8_t messageType = (status & 0xF0) >> 4;
    
    switch (messageType) {
        case 0x9: // Note On
            if (bytes.size() >= 3) {
                return std::make_unique<NoteOnMessage>(channel, bytes[1], bytes[2], timestamp);
            }
            break;
            
        case 0x8: // Note Off
            if (bytes.size() >= 3) {
                return std::make_unique<NoteOffMessage>(channel, bytes[1], bytes[2], timestamp);
            }
            break;
            
        case 0xB: // Control Change
            if (bytes.size() >= 3) {
                return std::make_unique<ControlChangeMessage>(channel, bytes[1], bytes[2], timestamp);
            }
            break;
            
        case 0xF: // System Exclusive (simplified)
            if (status == 0xF0 && bytes.size() > 1) {
                std::vector<uint8_t> data(bytes.begin() + 1, bytes.end());
                return std::make_unique<SystemExclusiveMessage>(0x00, data, timestamp);
            }
            break;
    }
    
    return nullptr;
}

} // namespace JMID
