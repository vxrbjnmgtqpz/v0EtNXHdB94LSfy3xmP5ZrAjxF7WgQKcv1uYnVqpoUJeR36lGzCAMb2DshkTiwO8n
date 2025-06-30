#pragma once

#include "JSONMIDIMessage.h"
#include <memory>
#include <string>
#include <vector>

namespace JSONMIDI {

/**
 * Factory class for creating MIDI messages from various sources
 */
class MessageFactory {
public:
    /**
     * Create a MIDIMessage from a JSON string
     * @param json JSON string representation of the MIDI message
     * @return Unique pointer to the created message, or nullptr if creation failed
     */
    static std::unique_ptr<MIDIMessage> createFromJSON(const std::string& json);
    
    /**
     * Create a MIDIMessage from raw MIDI bytes
     * @param bytes Vector of MIDI bytes
     * @param timestamp Timestamp for the message
     * @return Unique pointer to the created message, or nullptr if creation failed
     */
    static std::unique_ptr<MIDIMessage> createFromMIDIBytes(
        const std::vector<uint8_t>& bytes, 
        Timestamp timestamp
    );
};

} // namespace JSONMIDI
