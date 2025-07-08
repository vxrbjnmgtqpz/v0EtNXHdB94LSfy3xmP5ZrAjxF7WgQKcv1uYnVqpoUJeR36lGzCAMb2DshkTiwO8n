#include "CompactJMIDFormat.h"
#include <sstream>
#include <regex>
#include <algorithm>
#include <iostream>

namespace JMID {

//=============================================================================
// Static member definitions
//=============================================================================

const std::unordered_map<CompactJMIDFormat::MessageType, std::string> 
CompactJMIDFormat::typeToString_ = {
    {MessageType::NoteOn, "n+"},
    {MessageType::NoteOff, "n-"},
    {MessageType::ControlChange, "cc"},
    {MessageType::ProgramChange, "pc"},
    {MessageType::PitchBend, "pb"},
    {MessageType::AfterTouch, "at"},
    {MessageType::SystemEx, "sx"},
    {MessageType::TimingClock, "tc"},
    {MessageType::Undefined, "??"}
};

const std::unordered_map<std::string, CompactJMIDFormat::MessageType> 
CompactJMIDFormat::stringToType_ = {
    {"n+", MessageType::NoteOn},
    {"n-", MessageType::NoteOff},
    {"cc", MessageType::ControlChange},
    {"pc", MessageType::ProgramChange},
    {"pb", MessageType::PitchBend},
    {"at", MessageType::AfterTouch},
    {"sx", MessageType::SystemEx},
    {"tc", MessageType::TimingClock},
    {"??", MessageType::Undefined}
};

//=============================================================================
// CompactJMIDFormat Implementation
//=============================================================================

std::string CompactJMIDFormat::encodeNoteOn(int channel, int note, int velocity, 
                                           uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidNote(note) || !isValidVelocity(velocity)) {
        return ""; // Invalid parameters
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"n+\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::NOTE << "\":" << note
        << ",\"" << FieldNames::VELOCITY << "\":" << velocity
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

std::string CompactJMIDFormat::encodeNoteOff(int channel, int note, int velocity,
                                            uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidNote(note) || !isValidVelocity(velocity)) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"n-\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::NOTE << "\":" << note
        << ",\"" << FieldNames::VELOCITY << "\":" << velocity
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

std::string CompactJMIDFormat::encodeControlChange(int channel, int control, int value,
                                                  uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidControl(control) || value < 0 || value > 127) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"cc\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::CONTROL << "\":" << control
        << ",\"" << FieldNames::VALUE << "\":" << value
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

std::string CompactJMIDFormat::encodeProgramChange(int channel, int program,
                                                  uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidProgram(program)) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"pc\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::PROGRAM << "\":" << program
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

std::string CompactJMIDFormat::encodePitchBend(int channel, int bendValue,
                                              uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidBendValue(bendValue)) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"pb\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::BEND << "\":" << bendValue
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

std::string CompactJMIDFormat::encodeAfterTouch(int channel, int note, int pressure,
                                               uint64_t timestamp, uint64_t sequence) {
    if (!isValidChannel(channel) || !isValidNote(note) || pressure < 0 || pressure > 127) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "{\"" << FieldNames::TYPE << "\":\"at\""
        << ",\"" << FieldNames::CHANNEL << "\":" << channel
        << ",\"" << FieldNames::NOTE << "\":" << note
        << ",\"" << FieldNames::PRESSURE << "\":" << pressure
        << ",\"" << FieldNames::TIMESTAMP << "\":" << timestamp
        << ",\"" << FieldNames::SEQUENCE << "\":" << sequence
        << "}";
    
    return oss.str();
}

CompactJMIDFormat::DecodedMessage CompactJMIDFormat::decode(const std::string& compactJson) {
    DecodedMessage result;
    
    try {
        auto fields = parseCompactJson(compactJson);
        
        // Extract type
        auto typeIt = fields.find(FieldNames::TYPE);
        if (typeIt == fields.end()) {
            return result; // Invalid - no type field
        }
        
        result.type = getTypeFromIdentifier(typeIt->second);
        if (result.type == MessageType::Undefined) {
            return result; // Invalid type
        }
        
        // Extract common fields
        auto channelIt = fields.find(FieldNames::CHANNEL);
        if (channelIt != fields.end()) {
            result.channel = std::stoi(channelIt->second);
        }
        
        auto timestampIt = fields.find(FieldNames::TIMESTAMP);
        if (timestampIt != fields.end()) {
            result.timestamp = std::stoull(timestampIt->second);
        }
        
        auto sequenceIt = fields.find(FieldNames::SEQUENCE);
        if (sequenceIt != fields.end()) {
            result.sequence = std::stoull(sequenceIt->second);
        }
        
        // Extract type-specific fields
        switch (result.type) {
            case MessageType::NoteOn:
            case MessageType::NoteOff: {
                auto noteIt = fields.find(FieldNames::NOTE);
                auto velocityIt = fields.find(FieldNames::VELOCITY);
                if (noteIt != fields.end()) result.note = std::stoi(noteIt->second);
                if (velocityIt != fields.end()) result.velocity = std::stoi(velocityIt->second);
                break;
            }
            
            case MessageType::ControlChange: {
                auto controlIt = fields.find(FieldNames::CONTROL);
                auto valueIt = fields.find(FieldNames::VALUE);
                if (controlIt != fields.end()) result.control = std::stoi(controlIt->second);
                if (valueIt != fields.end()) result.value = std::stoi(valueIt->second);
                break;
            }
            
            case MessageType::ProgramChange: {
                auto programIt = fields.find(FieldNames::PROGRAM);
                if (programIt != fields.end()) result.program = std::stoi(programIt->second);
                break;
            }
            
            case MessageType::PitchBend: {
                auto bendIt = fields.find(FieldNames::BEND);
                if (bendIt != fields.end()) result.bendValue = std::stoi(bendIt->second);
                break;
            }
            
            case MessageType::AfterTouch: {
                auto noteIt = fields.find(FieldNames::NOTE);
                auto pressureIt = fields.find(FieldNames::PRESSURE);
                if (noteIt != fields.end()) result.note = std::stoi(noteIt->second);
                if (pressureIt != fields.end()) result.pressure = std::stoi(pressureIt->second);
                break;
            }
            
            default:
                break;
        }
        
        result.valid = true;
        
    } catch (const std::exception& e) {
        // Parsing failed
        result.valid = false;
    }
    
    return result;
}

std::string CompactJMIDFormat::getTypeIdentifier(MessageType type) {
    auto it = typeToString_.find(type);
    return (it != typeToString_.end()) ? it->second : "??";
}

CompactJMIDFormat::MessageType CompactJMIDFormat::getTypeFromIdentifier(const std::string& identifier) {
    auto it = stringToType_.find(identifier);
    return (it != stringToType_.end()) ? it->second : MessageType::Undefined;
}

size_t CompactJMIDFormat::estimateCompactSize(MessageType type) {
    switch (type) {
        case MessageType::NoteOn:
        case MessageType::NoteOff:
            return 45; // {"t":"n+","c":1,"n":60,"v":100,"ts":1234567890,"seq":123}
            
        case MessageType::ControlChange:
            return 48; // {"t":"cc","c":1,"cc":7,"val":127,"ts":1234567890,"seq":123}
            
        case MessageType::ProgramChange:
            return 40; // {"t":"pc","c":1,"p":42,"ts":1234567890,"seq":123}
            
        case MessageType::PitchBend:
            return 43; // {"t":"pb","c":1,"b":8192,"ts":1234567890,"seq":123}
            
        case MessageType::AfterTouch:
            return 47; // {"t":"at","c":1,"n":60,"pr":100,"ts":1234567890,"seq":123}
            
        default:
            return AVG_COMPACT_SIZE;
    }
}

double CompactJMIDFormat::getCompressionRatio(const std::string& verboseJson, const std::string& compactJson) {
    if (verboseJson.empty() || compactJson.empty()) {
        return 0.0;
    }
    
    return static_cast<double>(compactJson.size()) / verboseJson.size();
}

bool CompactJMIDFormat::isValidCompactFormat(const std::string& json) {
    // Quick validation - check for required compact format structure
    return json.find("\"t\":") != std::string::npos &&
           json.find("\"ts\":") != std::string::npos &&
           json.find("\"seq\":") != std::string::npos;
}

bool CompactJMIDFormat::hasRequiredFields(const std::string& json, MessageType type) {
    auto fields = parseCompactJson(json);
    
    // Check common required fields
    if (fields.find(FieldNames::TYPE) == fields.end() ||
        fields.find(FieldNames::CHANNEL) == fields.end() ||
        fields.find(FieldNames::TIMESTAMP) == fields.end() ||
        fields.find(FieldNames::SEQUENCE) == fields.end()) {
        return false;
    }
    
    // Check type-specific required fields
    switch (type) {
        case MessageType::NoteOn:
        case MessageType::NoteOff:
            return fields.find(FieldNames::NOTE) != fields.end() &&
                   fields.find(FieldNames::VELOCITY) != fields.end();
            
        case MessageType::ControlChange:
            return fields.find(FieldNames::CONTROL) != fields.end() &&
                   fields.find(FieldNames::VALUE) != fields.end();
            
        case MessageType::ProgramChange:
            return fields.find(FieldNames::PROGRAM) != fields.end();
            
        case MessageType::PitchBend:
            return fields.find(FieldNames::BEND) != fields.end();
            
        case MessageType::AfterTouch:
            return fields.find(FieldNames::NOTE) != fields.end() &&
                   fields.find(FieldNames::PRESSURE) != fields.end();
            
        default:
            return true;
    }
}

std::string CompactJMIDFormat::fastEncode(MessageType type, int channel, int primary, int secondary,
                                         uint64_t timestamp, uint64_t sequence) {
    // Fast encoding without validation for performance-critical paths
    std::ostringstream oss;
    oss << "{\"t\":\"" << getTypeIdentifier(type) << "\",\"c\":" << channel;
    
    switch (type) {
        case MessageType::NoteOn:
        case MessageType::NoteOff:
            oss << ",\"n\":" << primary << ",\"v\":" << secondary;
            break;
            
        case MessageType::ControlChange:
            oss << ",\"cc\":" << primary << ",\"val\":" << secondary;
            break;
            
        case MessageType::ProgramChange:
            oss << ",\"p\":" << primary;
            break;
            
        case MessageType::PitchBend:
            oss << ",\"b\":" << primary;
            break;
            
        case MessageType::AfterTouch:
            oss << ",\"n\":" << primary << ",\"pr\":" << secondary;
            break;
            
        default:
            break;
    }
    
    oss << ",\"ts\":" << timestamp << ",\"seq\":" << sequence << "}";
    return oss.str();
}

bool CompactJMIDFormat::fastDecode(const std::string& json, MessageType& type, int& channel,
                                  int& primary, int& secondary, uint64_t& timestamp, uint64_t& sequence) {
    // Fast regex-based decoding for performance-critical paths
    std::regex typeRegex("\"t\":\"([^\"]+)\"");
    std::regex channelRegex("\"c\":(\\d+)");
    std::regex timestampRegex("\"ts\":(\\d+)");
    std::regex sequenceRegex("\"seq\":(\\d+)");
    
    std::smatch match;
    
    // Extract type
    if (!std::regex_search(json, match, typeRegex)) return false;
    type = getTypeFromIdentifier(match[1].str());
    if (type == MessageType::Undefined) return false;
    
    // Extract common fields
    if (!std::regex_search(json, match, channelRegex)) return false;
    channel = std::stoi(match[1].str());
    
    if (!std::regex_search(json, match, timestampRegex)) return false;
    timestamp = std::stoull(match[1].str());
    
    if (!std::regex_search(json, match, sequenceRegex)) return false;
    sequence = std::stoull(match[1].str());
    
    // Extract type-specific fields
    switch (type) {
        case MessageType::NoteOn:
        case MessageType::NoteOff: {
            std::regex noteRegex("\"n\":(\\d+)");
            std::regex velocityRegex("\"v\":(\\d+)");
            if (!std::regex_search(json, match, noteRegex)) return false;
            primary = std::stoi(match[1].str());
            if (!std::regex_search(json, match, velocityRegex)) return false;
            secondary = std::stoi(match[1].str());
            break;
        }
        
        case MessageType::ControlChange: {
            std::regex controlRegex("\"cc\":(\\d+)");
            std::regex valueRegex("\"val\":(\\d+)");
            if (!std::regex_search(json, match, controlRegex)) return false;
            primary = std::stoi(match[1].str());
            if (!std::regex_search(json, match, valueRegex)) return false;
            secondary = std::stoi(match[1].str());
            break;
        }
        
        case MessageType::ProgramChange: {
            std::regex programRegex("\"p\":(\\d+)");
            if (!std::regex_search(json, match, programRegex)) return false;
            primary = std::stoi(match[1].str());
            secondary = 0;
            break;
        }
        
        case MessageType::PitchBend: {
            std::regex bendRegex("\"b\":(\\d+)");
            if (!std::regex_search(json, match, bendRegex)) return false;
            primary = std::stoi(match[1].str());
            secondary = 0;
            break;
        }
        
        default:
            return false;
    }
    
    return true;
}

//=============================================================================
// Private helper methods
//=============================================================================

std::string CompactJMIDFormat::buildCompactJson(const std::unordered_map<std::string, std::string>& fields) {
    std::ostringstream oss;
    oss << "{";
    
    bool first = true;
    for (const auto& [key, value] : fields) {
        if (!first) oss << ",";
        oss << "\"" << key << "\":" << value;
        first = false;
    }
    
    oss << "}";
    return oss.str();
}

std::unordered_map<std::string, std::string> CompactJMIDFormat::parseCompactJson(const std::string& json) {
    std::unordered_map<std::string, std::string> fields;
    
    // Simple regex-based JSON parsing (for compact format only)
    std::regex fieldRegex("\"([^\"]+)\":(\"([^\"]*)\"|(-?\\d+))");
    std::sregex_iterator iter(json.begin(), json.end(), fieldRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        const std::smatch& match = *iter;
        std::string key = match[1].str();
        std::string value = match[3].str().empty() ? match[4].str() : match[3].str();
        fields[key] = value;
    }
    
    return fields;
}

bool CompactJMIDFormat::isValidChannel(int channel) {
    return channel >= 1 && channel <= 16;
}

bool CompactJMIDFormat::isValidNote(int note) {
    return note >= 0 && note <= 127;
}

bool CompactJMIDFormat::isValidVelocity(int velocity) {
    return velocity >= 0 && velocity <= 127;
}

bool CompactJMIDFormat::isValidControl(int control) {
    return control >= 0 && control <= 127;
}

bool CompactJMIDFormat::isValidProgram(int program) {
    return program >= 0 && program <= 127;
}

bool CompactJMIDFormat::isValidBendValue(int bendValue) {
    return bendValue >= 0 && bendValue <= 16383;
}

//=============================================================================
// CompactJMIDBuilder Implementation
//=============================================================================

CompactJMIDBuilder& CompactJMIDBuilder::noteOn(int channel, int note, int velocity) {
    type_ = CompactJMIDFormat::MessageType::NoteOn;
    channel_ = channel;
    note_ = note;
    velocity_ = velocity;
    return *this;
}

CompactJMIDBuilder& CompactJMIDBuilder::noteOff(int channel, int note, int velocity) {
    type_ = CompactJMIDFormat::MessageType::NoteOff;
    channel_ = channel;
    note_ = note;
    velocity_ = velocity;
    return *this;
}

CompactJMIDBuilder& CompactJMIDBuilder::controlChange(int channel, int control, int value) {
    type_ = CompactJMIDFormat::MessageType::ControlChange;
    channel_ = channel;
    control_ = control;
    value_ = value;
    return *this;
}

CompactJMIDBuilder& CompactJMIDBuilder::programChange(int channel, int program) {
    type_ = CompactJMIDFormat::MessageType::ProgramChange;
    channel_ = channel;
    program_ = program;
    return *this;
}

CompactJMIDBuilder& CompactJMIDBuilder::pitchBend(int channel, int bendValue) {
    type_ = CompactJMIDFormat::MessageType::PitchBend;
    channel_ = channel;
    bendValue_ = bendValue;
    return *this;
}

CompactJMIDBuilder& CompactJMIDBuilder::afterTouch(int channel, int note, int pressure) {
    type_ = CompactJMIDFormat::MessageType::AfterTouch;
    channel_ = channel;
    note_ = note;
    pressure_ = pressure;
    return *this;
}

std::string CompactJMIDBuilder::build() {
    switch (type_) {
        case CompactJMIDFormat::MessageType::NoteOn:
            return CompactJMIDFormat::encodeNoteOn(channel_, note_, velocity_, timestamp_, sequence_);
            
        case CompactJMIDFormat::MessageType::NoteOff:
            return CompactJMIDFormat::encodeNoteOff(channel_, note_, velocity_, timestamp_, sequence_);
            
        case CompactJMIDFormat::MessageType::ControlChange:
            return CompactJMIDFormat::encodeControlChange(channel_, control_, value_, timestamp_, sequence_);
            
        case CompactJMIDFormat::MessageType::ProgramChange:
            return CompactJMIDFormat::encodeProgramChange(channel_, program_, timestamp_, sequence_);
            
        case CompactJMIDFormat::MessageType::PitchBend:
            return CompactJMIDFormat::encodePitchBend(channel_, bendValue_, timestamp_, sequence_);
            
        case CompactJMIDFormat::MessageType::AfterTouch:
            return CompactJMIDFormat::encodeAfterTouch(channel_, note_, pressure_, timestamp_, sequence_);
            
        default:
            return "";
    }
}

size_t CompactJMIDBuilder::estimateSize() const {
    return CompactJMIDFormat::estimateCompactSize(type_);
}

std::string CompactJMIDBuilder::quickNoteOn(int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    return CompactJMIDFormat::fastEncode(CompactJMIDFormat::MessageType::NoteOn, channel, note, velocity, timestamp, sequence);
}

std::string CompactJMIDBuilder::quickNoteOff(int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    return CompactJMIDFormat::fastEncode(CompactJMIDFormat::MessageType::NoteOff, channel, note, velocity, timestamp, sequence);
}

std::string CompactJMIDBuilder::quickCC(int channel, int control, int value, uint64_t timestamp, uint64_t sequence) {
    return CompactJMIDFormat::fastEncode(CompactJMIDFormat::MessageType::ControlChange, channel, control, value, timestamp, sequence);
}

//=============================================================================
// FormatStats Implementation
//=============================================================================

void FormatStats::recordMessage(const std::string& compactJson, const std::string& verboseJson) {
    totalMessages_++;
    size_t compactSize = compactJson.size();
    totalCompactBytes_ += compactSize;
    
    if (!verboseJson.empty()) {
        totalVerboseBytes_ += verboseJson.size();
    }
    
    minSize_ = std::min(minSize_, compactSize);
    maxSize_ = std::max(maxSize_, compactSize);
}

double FormatStats::getAverageSize() const {
    return totalMessages_ > 0 ? static_cast<double>(totalCompactBytes_) / totalMessages_ : 0.0;
}

double FormatStats::getCompressionRatio() const {
    return totalVerboseBytes_ > 0 ? static_cast<double>(totalCompactBytes_) / totalVerboseBytes_ : 0.0;
}

void FormatStats::reset() {
    totalMessages_ = 0;
    totalCompactBytes_ = 0;
    totalVerboseBytes_ = 0;
    minSize_ = SIZE_MAX;
    maxSize_ = 0;
}

FormatStats::Summary FormatStats::getSummary() const {
    Summary summary;
    summary.messageCount = totalMessages_;
    summary.averageCompactSize = getAverageSize();
    summary.compressionRatio = getCompressionRatio();
    summary.minSize = (minSize_ == SIZE_MAX) ? 0 : minSize_;
    summary.maxSize = maxSize_;
    summary.totalBytesSaved = totalVerboseBytes_ - totalCompactBytes_;
    return summary;
}

} // namespace JMID 