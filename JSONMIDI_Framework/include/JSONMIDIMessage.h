#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <chrono>

namespace JSONMIDI {

/**
 * MIDI Protocol Version
 */
enum class Protocol : uint8_t {
    MIDI1 = 1,
    MIDI2 = 2
};

/**
 * MIDI Message Types
 */
enum class MessageType : uint8_t {
    // Channel Voice Messages
    NOTE_ON = 0x90,
    NOTE_OFF = 0x80,
    POLYPHONIC_AFTERTOUCH = 0xA0,
    CONTROL_CHANGE = 0xB0,
    PROGRAM_CHANGE = 0xC0,
    CHANNEL_AFTERTOUCH = 0xD0,
    PITCH_BEND = 0xE0,
    
    // System Messages
    SYSTEM_EXCLUSIVE = 0xF0,
    TIME_CODE = 0xF1,
    SONG_POSITION = 0xF2,
    SONG_SELECT = 0xF3,
    TUNE_REQUEST = 0xF6,
    END_SYSEX = 0xF7,
    TIMING_CLOCK = 0xF8,
    START = 0xFA,
    CONTINUE = 0xFB,
    STOP = 0xFC,
    ACTIVE_SENSING = 0xFE,
    SYSTEM_RESET = 0xFF
};

/**
 * High-precision timestamp using microseconds
 */
using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::microseconds;

/**
 * Base class for all MIDI messages
 */
class MIDIMessage {
public:
    MIDIMessage(MessageType type, Timestamp timestamp, Protocol protocol = Protocol::MIDI1)
        : type_(type), timestamp_(timestamp), protocol_(protocol) {}
    
    virtual ~MIDIMessage() = default;
    
    MessageType getType() const { return type_; }
    Timestamp getTimestamp() const { return timestamp_; }
    Protocol getProtocol() const { return protocol_; }
    
    void setTimestamp(Timestamp timestamp) { timestamp_ = timestamp; }
    
    // Convert to JSON string
    virtual std::string toJSON() const = 0;
    
    // Convert to raw MIDI bytes
    virtual std::vector<uint8_t> toMIDIBytes() const = 0;
    
    // Get message size in bytes
    virtual size_t getByteSize() const = 0;

protected:
    MessageType type_;
    Timestamp timestamp_;
    Protocol protocol_;
};

/**
 * Channel Voice Message base class
 */
class ChannelMessage : public MIDIMessage {
public:
    ChannelMessage(MessageType type, uint8_t channel, Timestamp timestamp, 
                   Protocol protocol = Protocol::MIDI1, uint8_t group = 0)
        : MIDIMessage(type, timestamp, protocol), channel_(channel), group_(group) {
        if (channel < 1 || channel > 16) {
            throw std::invalid_argument("MIDI channel must be 1-16");
        }
    }
    
    uint8_t getChannel() const { return channel_; }
    uint8_t getGroup() const { return group_; }
    
    void setChannel(uint8_t channel) {
        if (channel < 1 || channel > 16) {
            throw std::invalid_argument("MIDI channel must be 1-16");
        }
        channel_ = channel;
    }

protected:
    uint8_t channel_;  // 1-16
    uint8_t group_;    // 0-15 (MIDI 2.0 groups)
};

/**
 * Note On message
 */
class NoteOnMessage : public ChannelMessage {
public:
    NoteOnMessage(uint8_t channel, uint8_t note, uint32_t velocity, 
                  Timestamp timestamp, Protocol protocol = Protocol::MIDI1)
        : ChannelMessage(MessageType::NOTE_ON, channel, timestamp, protocol)
        , note_(note), velocity_(velocity) {
        validateNote(note);
        validateVelocity(velocity, protocol);
    }
    
    uint8_t getNote() const { return note_; }
    uint32_t getVelocity() const { return velocity_; }
    
    // MIDI 2.0 per-note attributes
    std::optional<uint8_t> getAttributeType() const { return attributeType_; }
    std::optional<uint16_t> getAttributeValue() const { return attributeValue_; }
    
    void setPerNoteAttribute(uint8_t type, uint16_t value) {
        attributeType_ = type;
        attributeValue_ = value;
    }
    
    std::string toJSON() const override;
    std::vector<uint8_t> toMIDIBytes() const override;
    size_t getByteSize() const override;

private:
    uint8_t note_;                              // 0-127
    uint32_t velocity_;                         // 0-127 (MIDI1) or 0-65535 (MIDI2)
    std::optional<uint8_t> attributeType_;      // MIDI 2.0 only
    std::optional<uint16_t> attributeValue_;    // MIDI 2.0 only
    
    void validateNote(uint8_t note);
    void validateVelocity(uint32_t velocity, Protocol protocol);
};

/**
 * Note Off message
 */
class NoteOffMessage : public ChannelMessage {
public:
    NoteOffMessage(uint8_t channel, uint8_t note, uint32_t velocity, 
                   Timestamp timestamp, Protocol protocol = Protocol::MIDI1)
        : ChannelMessage(MessageType::NOTE_OFF, channel, timestamp, protocol)
        , note_(note), velocity_(velocity) {
        validateNote(note);
        validateVelocity(velocity, protocol);
    }
    
    uint8_t getNote() const { return note_; }
    uint32_t getVelocity() const { return velocity_; }
    
    std::string toJSON() const override;
    std::vector<uint8_t> toMIDIBytes() const override;
    size_t getByteSize() const override;

private:
    uint8_t note_;      // 0-127
    uint32_t velocity_; // 0-127 (MIDI1) or 0-65535 (MIDI2)
    
    void validateNote(uint8_t note);
    void validateVelocity(uint32_t velocity, Protocol protocol);
};

/**
 * Control Change message
 */
class ControlChangeMessage : public ChannelMessage {
public:
    ControlChangeMessage(uint8_t channel, uint32_t controller, uint32_t value,
                        Timestamp timestamp, Protocol protocol = Protocol::MIDI1)
        : ChannelMessage(MessageType::CONTROL_CHANGE, channel, timestamp, protocol)
        , controller_(controller), value_(value) {
        validateController(controller, protocol);
        validateValue(value, protocol);
    }
    
    uint32_t getController() const { return controller_; }
    uint32_t getValue() const { return value_; }
    
    std::string toJSON() const override;
    std::vector<uint8_t> toMIDIBytes() const override;
    size_t getByteSize() const override;

private:
    uint32_t controller_;   // 0-127 (MIDI1) or 0-32767 (MIDI2)
    uint32_t value_;        // 0-127 (MIDI1) or 0-4294967295 (MIDI2)
    
    void validateController(uint32_t controller, Protocol protocol);
    void validateValue(uint32_t value, Protocol protocol);
};

/**
 * System Exclusive message
 */
class SystemExclusiveMessage : public MIDIMessage {
public:
    enum class SysExType : uint8_t {
        SYSEX7 = 7,  // Traditional 7-bit SysEx
        SYSEX8 = 8   // MIDI 2.0 8-bit SysEx
    };
    
    SystemExclusiveMessage(uint32_t manufacturerId, const std::vector<uint8_t>& data,
                          Timestamp timestamp, SysExType sysexType = SysExType::SYSEX7)
        : MIDIMessage(MessageType::SYSTEM_EXCLUSIVE, timestamp, 
                      sysexType == SysExType::SYSEX8 ? Protocol::MIDI2 : Protocol::MIDI1)
        , manufacturerId_(manufacturerId), data_(data), sysexType_(sysexType) {}
    
    uint32_t getManufacturerId() const { return manufacturerId_; }
    const std::vector<uint8_t>& getData() const { return data_; }
    SysExType getSysExType() const { return sysexType_; }
    
    std::string toJSON() const override;
    std::vector<uint8_t> toMIDIBytes() const override;
    size_t getByteSize() const override;

private:
    uint32_t manufacturerId_;       // 1-3 bytes manufacturer ID
    std::vector<uint8_t> data_;     // SysEx data payload
    SysExType sysexType_;           // 7-bit or 8-bit SysEx
};

} // namespace JSONMIDI
