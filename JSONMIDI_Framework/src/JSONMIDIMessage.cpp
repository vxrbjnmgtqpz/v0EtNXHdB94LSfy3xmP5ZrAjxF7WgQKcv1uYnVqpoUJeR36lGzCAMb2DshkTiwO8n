#include "JSONMIDIMessage.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <chrono>

namespace JSONMIDI {

//=============================================================================
// Utility functions
//=============================================================================

namespace {
    std::string formatTimestamp(Timestamp timestamp) {
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            timestamp.time_since_epoch()).count();
        return std::to_string(microseconds);
    }
    
    std::string vectorToHexString(const std::vector<uint8_t>& bytes) {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i > 0) ss << " ";
            ss << std::setw(2) << static_cast<unsigned>(bytes[i]);
        }
        return ss.str();
    }
    
    std::string protocolToString(Protocol protocol) {
        return protocol == Protocol::MIDI1 ? "midi1" : "midi2";
    }
}

//=============================================================================
// NoteOnMessage Implementation
//=============================================================================

void NoteOnMessage::validateNote(uint8_t note) {
    if (note > 127) {
        throw std::invalid_argument("MIDI note must be 0-127");
    }
}

void NoteOnMessage::validateVelocity(uint32_t velocity, Protocol protocol) {
    if (protocol == Protocol::MIDI1 && velocity > 127) {
        throw std::invalid_argument("MIDI 1.0 velocity must be 0-127");
    } else if (protocol == Protocol::MIDI2 && velocity > 65535) {
        throw std::invalid_argument("MIDI 2.0 velocity must be 0-65535");
    }
}

std::string NoteOnMessage::toJSON() const {
    std::stringstream json;
    json << "{";
    json << "\"type\":\"noteOn\",";
    json << "\"timestamp\":" << formatTimestamp(timestamp_) << ",";
    json << "\"protocol\":\"" << protocolToString(protocol_) << "\",";
    json << "\"channel\":" << static_cast<int>(channel_) << ",";
    json << "\"note\":" << static_cast<int>(note_) << ",";
    json << "\"velocity\":" << velocity_;
    
    if (protocol_ == Protocol::MIDI2 && group_ > 0) {
        json << ",\"group\":" << static_cast<int>(group_);
    }
    
    if (attributeType_.has_value()) {
        json << ",\"attributeType\":" << static_cast<int>(attributeType_.value());
        json << ",\"attributeValue\":" << attributeValue_.value();
    }
    
    // Add raw bytes for verification
    auto bytes = toMIDIBytes();
    json << ",\"rawBytes\":[";
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) json << ",";
        json << static_cast<int>(bytes[i]);
    }
    json << "]";
    
    json << "}";
    return json.str();
}

std::vector<uint8_t> NoteOnMessage::toMIDIBytes() const {
    std::vector<uint8_t> bytes;
    
    if (protocol_ == Protocol::MIDI1) {
        // MIDI 1.0: Status byte + Note + Velocity
        uint8_t status = static_cast<uint8_t>(MessageType::NOTE_ON) | (channel_ - 1);
        bytes.push_back(status);
        bytes.push_back(note_);
        bytes.push_back(static_cast<uint8_t>(velocity_ & 0x7F));
    } else {
        // MIDI 2.0: UMP format (64-bit for Note On with velocity)
        // First 32-bit word: [Group][Type][Channel][Note]
        uint32_t word1 = (group_ << 28) | (0x4 << 24) | ((channel_ - 1) << 16) | (note_ << 8);
        
        // Second 32-bit word: [Velocity][Attribute Type][Attribute Value]
        uint32_t word2 = (velocity_ << 16);
        if (attributeType_.has_value()) {
            word2 |= (attributeType_.value() << 8) | attributeValue_.value();
        }
        
        // Convert to bytes (big-endian)
        bytes.push_back((word1 >> 24) & 0xFF);
        bytes.push_back((word1 >> 16) & 0xFF);
        bytes.push_back((word1 >> 8) & 0xFF);
        bytes.push_back(word1 & 0xFF);
        bytes.push_back((word2 >> 24) & 0xFF);
        bytes.push_back((word2 >> 16) & 0xFF);
        bytes.push_back((word2 >> 8) & 0xFF);
        bytes.push_back(word2 & 0xFF);
    }
    
    return bytes;
}

size_t NoteOnMessage::getByteSize() const {
    return protocol_ == Protocol::MIDI1 ? 3 : 8;
}

//=============================================================================
// NoteOffMessage Implementation
//=============================================================================

void NoteOffMessage::validateNote(uint8_t note) {
    if (note > 127) {
        throw std::invalid_argument("MIDI note must be 0-127");
    }
}

void NoteOffMessage::validateVelocity(uint32_t velocity, Protocol protocol) {
    if (protocol == Protocol::MIDI1 && velocity > 127) {
        throw std::invalid_argument("MIDI 1.0 velocity must be 0-127");
    } else if (protocol == Protocol::MIDI2 && velocity > 65535) {
        throw std::invalid_argument("MIDI 2.0 velocity must be 0-65535");
    }
}

std::string NoteOffMessage::toJSON() const {
    std::stringstream json;
    json << "{";
    json << "\"type\":\"noteOff\",";
    json << "\"timestamp\":" << formatTimestamp(timestamp_) << ",";
    json << "\"protocol\":\"" << protocolToString(protocol_) << "\",";
    json << "\"channel\":" << static_cast<int>(channel_) << ",";
    json << "\"note\":" << static_cast<int>(note_) << ",";
    json << "\"velocity\":" << velocity_;
    
    if (protocol_ == Protocol::MIDI2 && group_ > 0) {
        json << ",\"group\":" << static_cast<int>(group_);
    }
    
    // Add raw bytes for verification
    auto bytes = toMIDIBytes();
    json << ",\"rawBytes\":[";
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) json << ",";
        json << static_cast<int>(bytes[i]);
    }
    json << "]";
    
    json << "}";
    return json.str();
}

std::vector<uint8_t> NoteOffMessage::toMIDIBytes() const {
    std::vector<uint8_t> bytes;
    
    if (protocol_ == Protocol::MIDI1) {
        // MIDI 1.0: Status byte + Note + Release Velocity
        uint8_t status = static_cast<uint8_t>(MessageType::NOTE_OFF) | (channel_ - 1);
        bytes.push_back(status);
        bytes.push_back(note_);
        bytes.push_back(static_cast<uint8_t>(velocity_ & 0x7F));
    } else {
        // MIDI 2.0: UMP format (64-bit for Note Off)
        uint32_t word1 = (group_ << 28) | (0x4 << 24) | ((channel_ - 1) << 16) | (note_ << 8) | 0x80;
        uint32_t word2 = velocity_ << 16;
        
        // Convert to bytes (big-endian)
        bytes.push_back((word1 >> 24) & 0xFF);
        bytes.push_back((word1 >> 16) & 0xFF);
        bytes.push_back((word1 >> 8) & 0xFF);
        bytes.push_back(word1 & 0xFF);
        bytes.push_back((word2 >> 24) & 0xFF);
        bytes.push_back((word2 >> 16) & 0xFF);
        bytes.push_back((word2 >> 8) & 0xFF);
        bytes.push_back(word2 & 0xFF);
    }
    
    return bytes;
}

size_t NoteOffMessage::getByteSize() const {
    return protocol_ == Protocol::MIDI1 ? 3 : 8;
}

//=============================================================================
// ControlChangeMessage Implementation
//=============================================================================

void ControlChangeMessage::validateController(uint32_t controller, Protocol protocol) {
    if (protocol == Protocol::MIDI1 && controller > 127) {
        throw std::invalid_argument("MIDI 1.0 controller must be 0-127");
    } else if (protocol == Protocol::MIDI2 && controller > 32767) {
        throw std::invalid_argument("MIDI 2.0 controller must be 0-32767");
    }
}

void ControlChangeMessage::validateValue(uint32_t value, Protocol protocol) {
    if (protocol == Protocol::MIDI1 && value > 127) {
        throw std::invalid_argument("MIDI 1.0 control value must be 0-127");
    } else if (protocol == Protocol::MIDI2 && value > 4294967295UL) {
        throw std::invalid_argument("MIDI 2.0 control value must be 0-4294967295");
    }
}

std::string ControlChangeMessage::toJSON() const {
    std::stringstream json;
    json << "{";
    json << "\"type\":\"controlChange\",";
    json << "\"timestamp\":" << formatTimestamp(timestamp_) << ",";
    json << "\"protocol\":\"" << protocolToString(protocol_) << "\",";
    json << "\"channel\":" << static_cast<int>(channel_) << ",";
    json << "\"controller\":" << controller_ << ",";
    json << "\"value\":" << value_;
    
    if (protocol_ == Protocol::MIDI2 && group_ > 0) {
        json << ",\"group\":" << static_cast<int>(group_);
    }
    
    // Add raw bytes for verification
    auto bytes = toMIDIBytes();
    json << ",\"rawBytes\":[";
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) json << ",";
        json << static_cast<int>(bytes[i]);
    }
    json << "]";
    
    json << "}";
    return json.str();
}

std::vector<uint8_t> ControlChangeMessage::toMIDIBytes() const {
    std::vector<uint8_t> bytes;
    
    if (protocol_ == Protocol::MIDI1) {
        // MIDI 1.0: Status byte + Controller + Value
        uint8_t status = static_cast<uint8_t>(MessageType::CONTROL_CHANGE) | (channel_ - 1);
        bytes.push_back(status);
        bytes.push_back(static_cast<uint8_t>(controller_ & 0x7F));
        bytes.push_back(static_cast<uint8_t>(value_ & 0x7F));
    } else {
        // MIDI 2.0: UMP format with extended controller and value ranges
        uint32_t word1 = (group_ << 28) | (0x4 << 24) | ((channel_ - 1) << 16) | 
                         (static_cast<uint8_t>(MessageType::CONTROL_CHANGE) << 8);
        uint32_t word2 = (controller_ << 16) | (value_ & 0xFFFF);
        
        // Convert to bytes (big-endian)
        bytes.push_back((word1 >> 24) & 0xFF);
        bytes.push_back((word1 >> 16) & 0xFF);
        bytes.push_back((word1 >> 8) & 0xFF);
        bytes.push_back(word1 & 0xFF);
        bytes.push_back((word2 >> 24) & 0xFF);
        bytes.push_back((word2 >> 16) & 0xFF);
        bytes.push_back((word2 >> 8) & 0xFF);
        bytes.push_back(word2 & 0xFF);
    }
    
    return bytes;
}

size_t ControlChangeMessage::getByteSize() const {
    return protocol_ == Protocol::MIDI1 ? 3 : 8;
}

//=============================================================================
// SystemExclusiveMessage Implementation
//=============================================================================

std::string SystemExclusiveMessage::toJSON() const {
    std::stringstream json;
    json << "{";
    json << "\"type\":\"systemExclusive\",";
    json << "\"timestamp\":" << formatTimestamp(timestamp_) << ",";
    json << "\"protocol\":\"" << protocolToString(protocol_) << "\",";
    json << "\"manufacturerId\":" << manufacturerId_ << ",";
    json << "\"sysexType\":\"" << (sysexType_ == SysExType::SYSEX7 ? "sysex7" : "sysex8") << "\",";
    
    // Data array
    json << "\"data\":[";
    for (size_t i = 0; i < data_.size(); ++i) {
        if (i > 0) json << ",";
        json << static_cast<int>(data_[i]);
    }
    json << "]";
    
    // Add raw bytes for verification
    auto bytes = toMIDIBytes();
    json << ",\"rawBytes\":[";
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) json << ",";
        json << static_cast<int>(bytes[i]);
    }
    json << "]";
    
    json << "}";
    return json.str();
}

std::vector<uint8_t> SystemExclusiveMessage::toMIDIBytes() const {
    std::vector<uint8_t> bytes;
    
    if (sysexType_ == SysExType::SYSEX7) {
        // Traditional MIDI 1.0 SysEx
        bytes.push_back(static_cast<uint8_t>(MessageType::SYSTEM_EXCLUSIVE));
        
        // Manufacturer ID (1-3 bytes)
        if (manufacturerId_ <= 0x7F) {
            // Single byte manufacturer ID
            bytes.push_back(static_cast<uint8_t>(manufacturerId_));
        } else {
            // Extended manufacturer ID (3 bytes: 00, ID_MSB, ID_LSB)
            bytes.push_back(0x00);
            bytes.push_back((manufacturerId_ >> 8) & 0x7F);
            bytes.push_back(manufacturerId_ & 0x7F);
        }
        
        // Data bytes (7-bit)
        for (uint8_t byte : data_) {
            bytes.push_back(byte & 0x7F);
        }
        
        // End SysEx
        bytes.push_back(static_cast<uint8_t>(MessageType::END_SYSEX));
    } else {
        // MIDI 2.0 SysEx8 - allows full 8-bit data
        // UMP format for SysEx8
        bytes.push_back(0x50); // SysEx8 start
        
        // Manufacturer ID
        if (manufacturerId_ <= 0xFF) {
            bytes.push_back(static_cast<uint8_t>(manufacturerId_));
        } else {
            bytes.push_back((manufacturerId_ >> 16) & 0xFF);
            bytes.push_back((manufacturerId_ >> 8) & 0xFF);
            bytes.push_back(manufacturerId_ & 0xFF);
        }
        
        // Data bytes (full 8-bit)
        for (uint8_t byte : data_) {
            bytes.push_back(byte);
        }
        
        bytes.push_back(0x57); // SysEx8 end
    }
    
    return bytes;
}

size_t SystemExclusiveMessage::getByteSize() const {
    size_t baseSize = 2; // Start and end bytes
    
    // Manufacturer ID size
    if (manufacturerId_ <= 0x7F) {
        baseSize += 1;
    } else {
        baseSize += 3;
    }
    
    // Data size
    baseSize += data_.size();
    
    return baseSize;
}

} // namespace JSONMIDI
