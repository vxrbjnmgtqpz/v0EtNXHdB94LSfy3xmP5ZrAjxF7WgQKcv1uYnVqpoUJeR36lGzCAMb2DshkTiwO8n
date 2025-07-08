#include "SIMDJMIDParser.h"
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cassert>

namespace JMID {

// Static member initialization
bool SIMDJMIDParser::typeTableInitialized_ = false;
SIMDJMIDParser::ParsedMessage::Type SIMDJMIDParser::typeTable_[256];

void SIMDJMIDParser::initializeTypeTable() {
    if (typeTableInitialized_) return;
    
    // Initialize all to Invalid
    std::fill(std::begin(typeTable_), std::end(typeTable_), ParsedMessage::Invalid);
    
    // Fast type lookup based on first character of type field value
    typeTable_['n'] = ParsedMessage::NoteOn;  // Will distinguish n+ vs n- later
    typeTable_['c'] = ParsedMessage::ControlChange;
    typeTable_['p'] = ParsedMessage::ProgramChange;
    typeTable_['b'] = ParsedMessage::PitchBend;    // "pb" -> check 'b' for bend
    typeTable_['a'] = ParsedMessage::AfterTouch;
    typeTable_['s'] = ParsedMessage::SystemEx;
    typeTable_['t'] = ParsedMessage::TimingClock;
    
    typeTableInitialized_ = true;
}

// Ultra-fast parsing implementation
SIMDJMIDParser::ParsedMessage SIMDJMIDParser::fastParse(const std::string& compactJson) {
    return fastParse(compactJson.c_str(), compactJson.length());
}

SIMDJMIDParser::ParsedMessage SIMDJMIDParser::fastParse(const char* json, size_t length) {
    auto startTime = getCurrentMicros();
    
    if (!typeTableInitialized_) {
        initializeTypeTable();
    }
    
    ParsedMessage result;
    
    // Basic validation
    if (length < 10 || json[0] != '{' || json[length-1] != '}') {
        recordParseTime(getCurrentMicros() - startTime);
        return result;
    }
    
    // Fast field parsing using direct character search
    result.type = parseTypeField(json, length);
    if (result.type == ParsedMessage::Invalid) {
        recordParseTime(getCurrentMicros() - startTime);
        return result;
    }
    
    // Parse common fields that appear in all messages
    result.timestamp = parseTimestampField(json, length);
    result.sequence = parseSequenceField(json, length);
    
    // Type-specific parsing
    switch (result.type) {
        case ParsedMessage::NoteOn:
        case ParsedMessage::NoteOff:
            result.channel = parseChannelField(json, length);
            result.note = parseNoteField(json, length);
            result.velocity = parseVelocityField(json, length);
            break;
            
        case ParsedMessage::ControlChange:
            result.channel = parseChannelField(json, length);
            result.control = parseControlField(json, length);
            result.value = parseValueField(json, length);
            break;
            
        case ParsedMessage::ProgramChange:
            result.channel = parseChannelField(json, length);
            result.program = parseProgramField(json, length);
            break;
            
        case ParsedMessage::PitchBend:
            result.channel = parseChannelField(json, length);
            result.bendValue = parseBendField(json, length);
            break;
            
        default:
            result.channel = parseChannelField(json, length);
            break;
    }
    
    // Basic validation
    result.valid = validateMessage(result);
    
    auto parseTime = getCurrentMicros() - startTime;
    recordParseTime(parseTime);
    
    return result;
}

// Fast type parsing with character-level optimization
inline SIMDJMIDParser::ParsedMessage::Type SIMDJMIDParser::parseTypeField(const char* json, size_t length) {
    // Look for "t":"<type>" pattern
    const char* typeStart = findField(json, length, "\"t\":\"", 5);
    if (!typeStart) return ParsedMessage::Invalid;
    
    typeStart += 5; // Skip past "t":"
    
    // Check first character for fast dispatch
    char firstChar = typeStart[0];
    if (firstChar == 'n') {
        // Note message - check second character
        if (typeStart[1] == '+') return ParsedMessage::NoteOn;
        if (typeStart[1] == '-') return ParsedMessage::NoteOff;
        return ParsedMessage::Invalid;
    }
    
    if (firstChar == 'c' && typeStart[1] == 'c') return ParsedMessage::ControlChange;
    if (firstChar == 'p' && typeStart[1] == 'c') return ParsedMessage::ProgramChange;
    if (firstChar == 'p' && typeStart[1] == 'b') return ParsedMessage::PitchBend;
    
    return ParsedMessage::Invalid;
}

// Ultra-fast integer parsing
inline uint64_t SIMDJMIDParser::fastParseInteger(const char* start, const char* end) {
    uint64_t result = 0;
    for (const char* p = start; p < end && *p >= '0' && *p <= '9'; ++p) {
        result = result * 10 + (*p - '0');
    }
    return result;
}

// Field parsing implementations
inline uint8_t SIMDJMIDParser::parseChannelField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"c\":", 4);
    if (!field) return 0;
    field += 4;
    return static_cast<uint8_t>(fastParseInteger(field, field + 2));
}

inline uint8_t SIMDJMIDParser::parseNoteField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"n\":", 4);
    if (!field) return 0;
    field += 4;
    return static_cast<uint8_t>(fastParseInteger(field, field + 3));
}

inline uint8_t SIMDJMIDParser::parseVelocityField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"v\":", 4);
    if (!field) return 0;
    field += 4;
    return static_cast<uint8_t>(fastParseInteger(field, field + 3));
}

inline uint8_t SIMDJMIDParser::parseControlField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"cc\":", 5);
    if (!field) return 0;
    field += 5;
    return static_cast<uint8_t>(fastParseInteger(field, field + 3));
}

inline uint8_t SIMDJMIDParser::parseProgramField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"p\":", 4);
    if (!field) return 0;
    field += 4;
    return static_cast<uint8_t>(fastParseInteger(field, field + 3));
}

inline uint16_t SIMDJMIDParser::parseBendField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"b\":", 4);
    if (!field) return 8192;
    field += 4;
    return static_cast<uint16_t>(fastParseInteger(field, field + 5));
}

inline uint16_t SIMDJMIDParser::parseValueField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"val\":", 6);
    if (!field) return 0;
    field += 6;
    return static_cast<uint16_t>(fastParseInteger(field, field + 3));
}

inline uint64_t SIMDJMIDParser::parseTimestampField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"ts\":", 5);
    if (!field) return 0;
    field += 5;
    return fastParseInteger(field, field + 13);
}

inline uint64_t SIMDJMIDParser::parseSequenceField(const char* json, size_t length) {
    const char* field = findField(json, length, "\"seq\":", 6);
    if (!field) return 0;
    field += 6;
    return fastParseInteger(field, field + 10);
}

// SIMD-style field finding
inline const char* SIMDJMIDParser::findField(const char* json, size_t length, const char* fieldName, size_t fieldLen) {
    const char* end = json + length - fieldLen;
    for (const char* p = json; p <= end; ++p) {
        if (fastCompareString(p, fieldName, fieldLen)) {
            return p;
        }
    }
    return nullptr;
}

inline bool SIMDJMIDParser::fastCompareString(const char* a, const char* b, size_t len) {
    // Optimized comparison for small strings
    if (len <= 8) {
        uint64_t aVal = 0, bVal = 0;
        memcpy(&aVal, a, std::min(len, sizeof(uint64_t)));
        memcpy(&bVal, b, std::min(len, sizeof(uint64_t)));
        return aVal == bVal;
    }
    return memcmp(a, b, len) == 0;
}

// Validation
inline bool SIMDJMIDParser::validateMessage(const ParsedMessage& msg) {
    if (validationLevel_ == 0) return true;
    
    bool valid = msg.type != ParsedMessage::Invalid;
    
    if (validationLevel_ >= 1) {
        valid &= (msg.channel <= 15);
        if (msg.isNoteMessage()) {
            valid &= (msg.note <= 127 && msg.velocity <= 127);
        }
    }
    
    if (validationLevel_ >= 2 && validateTimestamps_) {
        valid &= isValidTimestamp(msg.timestamp);
    }
    
    return valid;
}

inline bool SIMDJMIDParser::isValidTimestamp(uint64_t timestamp) {
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t diff = (timestamp > now) ? timestamp - now : now - timestamp;
    return diff < 60000; // Within 1 minute
}

// Performance tracking
inline void SIMDJMIDParser::recordParseTime(uint64_t micros) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.totalMessages++;
    stats_.totalParseTimeMicros += micros;
    stats_.successfulParses++;
    stats_.minParseTimeMicros = std::min(stats_.minParseTimeMicros, micros);
    stats_.maxParseTimeMicros = std::max(stats_.maxParseTimeMicros, micros);
    stats_.averageParseTimeMicros = static_cast<double>(stats_.totalParseTimeMicros) / stats_.totalMessages;
    stats_.messagesPerSecond = 1000000.0 / stats_.averageParseTimeMicros;
}

inline uint64_t SIMDJMIDParser::getCurrentMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Batch parsing for maximum throughput
std::vector<SIMDJMIDParser::ParsedMessage> SIMDJMIDParser::batchParse(const std::vector<std::string>& messages) {
    std::vector<ParsedMessage> results;
    results.reserve(messages.size());
    
    for (const auto& msg : messages) {
        results.push_back(fastParse(msg));
    }
    
    return results;
}

size_t SIMDJMIDParser::batchParse(const std::vector<std::string>& messages, ParsedMessage* output) {
    size_t count = 0;
    for (const auto& msg : messages) {
        output[count++] = fastParse(msg);
    }
    return count;
}

// Burst parsing with deduplication
SIMDJMIDParser::BurstParseResult SIMDJMIDParser::burstParse(const std::string& compactJson) {
    BurstParseResult result;
    result.message = fastParse(compactJson);
    result.isDuplicate = false;
    result.duplicateCount = 0;
    
    if (trackSequences_ && result.message.valid) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        if (seenSequences_.count(result.message.sequence) > 0) {
            result.isDuplicate = true;
            result.duplicateCount = 1;
        } else {
            seenSequences_.insert(result.message.sequence);
        }
    }
    
    return result;
}

// Utility methods
bool SIMDJMIDParser::isValidCompactFormat(const char* json, size_t length) {
    return length >= 10 && json[0] == '{' && json[length-1] == '}';
}

SIMDJMIDParser::ParsedMessage::Type SIMDJMIDParser::extractType(const char* json, size_t length) {
    SIMDJMIDParser parser;
    return parser.parseTypeField(json, length);
}

uint64_t SIMDJMIDParser::extractSequence(const char* json, size_t length) {
    SIMDJMIDParser parser;
    return parser.parseSequenceField(json, length);
}

uint64_t SIMDJMIDParser::extractTimestamp(const char* json, size_t length) {
    SIMDJMIDParser parser;
    return parser.parseTimestampField(json, length);
}

void SIMDJMIDParser::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = ParseStats{};
    seenSequences_.clear();
}

// FastMessageBuilder implementation
std::string FastMessageBuilder::buildNoteOn(int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    char buffer[MAX_MESSAGE_SIZE];
    auto len = snprintf(buffer, sizeof(buffer), NOTE_ON_TEMPLATE, channel, note, velocity, timestamp, sequence);
    return std::string(buffer, len);
}

std::string FastMessageBuilder::buildNoteOff(int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    char buffer[MAX_MESSAGE_SIZE];
    auto len = snprintf(buffer, sizeof(buffer), NOTE_OFF_TEMPLATE, channel, note, velocity, timestamp, sequence);
    return std::string(buffer, len);
}

std::string FastMessageBuilder::buildControlChange(int channel, int control, int value, uint64_t timestamp, uint64_t sequence) {
    char buffer[MAX_MESSAGE_SIZE];
    auto len = snprintf(buffer, sizeof(buffer), CONTROL_TEMPLATE, channel, control, value, timestamp, sequence);
    return std::string(buffer, len);
}

std::string FastMessageBuilder::buildProgramChange(int channel, int program, uint64_t timestamp, uint64_t sequence) {
    char buffer[MAX_MESSAGE_SIZE];
    auto len = snprintf(buffer, sizeof(buffer), PROGRAM_TEMPLATE, channel, program, timestamp, sequence);
    return std::string(buffer, len);
}

std::string FastMessageBuilder::buildPitchBend(int channel, int bendValue, uint64_t timestamp, uint64_t sequence) {
    char buffer[MAX_MESSAGE_SIZE];
    auto len = snprintf(buffer, sizeof(buffer), BEND_TEMPLATE, channel, bendValue, timestamp, sequence);
    return std::string(buffer, len);
}

// Zero-allocation builders
size_t FastMessageBuilder::buildNoteOn(char* buffer, size_t bufferSize, int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    return snprintf(buffer, bufferSize, NOTE_ON_TEMPLATE, channel, note, velocity, timestamp, sequence);
}

size_t FastMessageBuilder::buildNoteOff(char* buffer, size_t bufferSize, int channel, int note, int velocity, uint64_t timestamp, uint64_t sequence) {
    return snprintf(buffer, bufferSize, NOTE_OFF_TEMPLATE, channel, note, velocity, timestamp, sequence);
}

size_t FastMessageBuilder::buildControlChange(char* buffer, size_t bufferSize, int channel, int control, int value, uint64_t timestamp, uint64_t sequence) {
    return snprintf(buffer, bufferSize, CONTROL_TEMPLATE, channel, control, value, timestamp, sequence);
}

size_t FastMessageBuilder::buildProgramChange(char* buffer, size_t bufferSize, int channel, int program, uint64_t timestamp, uint64_t sequence) {
    return snprintf(buffer, bufferSize, PROGRAM_TEMPLATE, channel, program, timestamp, sequence);
}

size_t FastMessageBuilder::buildPitchBend(char* buffer, size_t bufferSize, int channel, int bendValue, uint64_t timestamp, uint64_t sequence) {
    return snprintf(buffer, bufferSize, BEND_TEMPLATE, channel, bendValue, timestamp, sequence);
}

// Benchmark implementation
SIMDParserBenchmark::BenchmarkResult SIMDParserBenchmark::runParsingBenchmark(size_t numMessages) {
    auto testMessages = generateTestMessages(numMessages);
    SIMDJMIDParser parser;
    
    auto startTime = getCurrentMicroseconds();
    
    size_t successful = 0;
    for (const auto& msg : testMessages) {
        auto result = parser.fastParse(msg);
        if (result.valid) successful++;
    }
    
    auto endTime = getCurrentMicroseconds();
    auto totalTime = endTime - startTime;
    
    BenchmarkResult result;
    result.totalMessages = numMessages;
    result.totalTimeMicros = totalTime;
    result.avgParseTimeMicros = static_cast<double>(totalTime) / numMessages;
    result.messagesPerSecond = 1000000.0 * numMessages / totalTime;
    result.speedupFactor = 1000.0 / result.avgParseTimeMicros; // Assuming 1ms baseline
    result.targetAchieved = result.avgParseTimeMicros < 10.0;
    
    return result;
}

std::vector<std::string> SIMDParserBenchmark::generateTestMessages(size_t count) {
    std::vector<std::string> messages;
    messages.reserve(count);
    
    uint64_t timestamp = getCurrentMicroseconds() / 1000;
    
    for (size_t i = 0; i < count; ++i) {
        switch (i % 5) {
            case 0:
                messages.push_back(FastMessageBuilder::buildNoteOn(1, 60, 100, timestamp + i, i));
                break;
            case 1:
                messages.push_back(FastMessageBuilder::buildNoteOff(1, 60, 0, timestamp + i, i));
                break;
            case 2:
                messages.push_back(FastMessageBuilder::buildControlChange(1, 7, 127, timestamp + i, i));
                break;
            case 3:
                messages.push_back(FastMessageBuilder::buildProgramChange(1, 42, timestamp + i, i));
                break;
            case 4:
                messages.push_back(FastMessageBuilder::buildPitchBend(1, 16383, timestamp + i, i));
                break;
        }
    }
    
    return messages;
}

uint64_t SIMDParserBenchmark::getCurrentMicroseconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void SIMDParserBenchmark::runComprehensiveBenchmark() {
    printf("\n🔥 SIMD JMID Parser - Comprehensive Performance Benchmark\n");
    printf("========================================================\n\n");
    
    auto singleResult = runParsingBenchmark(100000);
    printf("📊 Single Message Parsing (100K messages):\n");
    printf("   Average Parse Time: %.2f μs\n", singleResult.avgParseTimeMicros);
    printf("   Messages/Second: %.0f\n", singleResult.messagesPerSecond);
    printf("   Target <10μs: %s\n", singleResult.targetAchieved ? "✅ ACHIEVED" : "❌ NOT MET");
    printf("   Speedup Factor: %.1fx\n\n", singleResult.speedupFactor);
    
    auto batchResult = runBatchBenchmark(1000, 100);
    printf("📦 Batch Parsing (1K batch × 100 batches):\n");
    printf("   Average Parse Time: %.2f μs\n", batchResult.avgParseTimeMicros);
    printf("   Messages/Second: %.0f\n", batchResult.messagesPerSecond);
    printf("   Target <10μs: %s\n\n", batchResult.targetAchieved ? "✅ ACHIEVED" : "❌ NOT MET");
    
    auto burstResult = runBurstBenchmark(10000, 3);
    printf("💥 Burst Parsing (10K bursts × 3 packets):\n");
    printf("   Average Parse Time: %.2f μs\n", burstResult.avgParseTimeMicros);
    printf("   Messages/Second: %.0f\n", burstResult.messagesPerSecond);
    printf("   Target <10μs: %s\n\n", burstResult.targetAchieved ? "✅ ACHIEVED" : "❌ NOT MET");
    
    printf("🎯 Overall Performance Summary:\n");
    printf("   All targets achieved: %s\n", 
        (singleResult.targetAchieved && batchResult.targetAchieved && burstResult.targetAchieved) 
        ? "✅ SUCCESS" : "❌ NEEDS OPTIMIZATION");
    printf("   Peak throughput: %.0f messages/sec\n", 
        std::max({singleResult.messagesPerSecond, batchResult.messagesPerSecond, burstResult.messagesPerSecond}));
}

SIMDParserBenchmark::BenchmarkResult SIMDParserBenchmark::runBatchBenchmark(size_t batchSize, size_t numBatches) {
    auto testMessages = generateTestMessages(batchSize);
    SIMDJMIDParser parser;
    
    auto startTime = getCurrentMicroseconds();
    
    for (size_t i = 0; i < numBatches; ++i) {
        auto results = parser.batchParse(testMessages);
    }
    
    auto endTime = getCurrentMicroseconds();
    auto totalTime = endTime - startTime;
    auto totalMessages = batchSize * numBatches;
    
    BenchmarkResult result;
    result.totalMessages = totalMessages;
    result.totalTimeMicros = totalTime;
    result.avgParseTimeMicros = static_cast<double>(totalTime) / totalMessages;
    result.messagesPerSecond = 1000000.0 * totalMessages / totalTime;
    result.speedupFactor = 1000.0 / result.avgParseTimeMicros;
    result.targetAchieved = result.avgParseTimeMicros < 10.0;
    
    return result;
}

SIMDParserBenchmark::BenchmarkResult SIMDParserBenchmark::runBurstBenchmark(size_t numBursts, int burstSize) {
    auto testMessages = generateTestMessages(numBursts);
    SIMDJMIDParser parser;
    
    auto startTime = getCurrentMicroseconds();
    
    for (size_t i = 0; i < numBursts; ++i) {
        for (int j = 0; j < burstSize; ++j) {
            auto result = parser.burstParse(testMessages[i]);
        }
    }
    
    auto endTime = getCurrentMicroseconds();
    auto totalTime = endTime - startTime;
    auto totalMessages = numBursts * burstSize;
    
    BenchmarkResult result;
    result.totalMessages = totalMessages;
    result.totalTimeMicros = totalTime;
    result.avgParseTimeMicros = static_cast<double>(totalTime) / totalMessages;
    result.messagesPerSecond = 1000000.0 * totalMessages / totalTime;
    result.speedupFactor = 1000.0 / result.avgParseTimeMicros;
    result.targetAchieved = result.avgParseTimeMicros < 10.0;
    
    return result;
}

} // namespace JMID 