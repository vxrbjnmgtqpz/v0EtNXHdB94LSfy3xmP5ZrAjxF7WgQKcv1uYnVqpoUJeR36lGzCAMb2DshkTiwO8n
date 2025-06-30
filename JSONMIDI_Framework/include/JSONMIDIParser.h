#pragma once

#include "JSONMIDIMessage.h"
#include "LockFreeQueue.h"
#include <memory>
#include <functional>
#include <queue>
#include <mutex>
#include <atomic>
#include <map>
#include <limits>
#include <nlohmann/json.hpp>

namespace JSONMIDI {

/**
 * JSON Schema Validation Result
 */
struct ValidationResult {
    bool isValid;
    std::string errorMessage;
    std::string errorPath;
    
    ValidationResult(bool valid = true, const std::string& error = "", const std::string& path = "")
        : isValid(valid), errorMessage(error), errorPath(path) {}
};

/**
 * Fast JSON parser optimized for MIDI streaming
 * Based on the Bassoon.js concept for ultra-low latency parsing
 */
class BassoonParser {
public:
    BassoonParser();
    ~BassoonParser();
    
    // Parse JSON string to MIDI message
    std::unique_ptr<MIDIMessage> parseMessage(const std::string& json);
    
    // Parse JSON string with validation
    std::pair<std::unique_ptr<MIDIMessage>, ValidationResult> 
    parseMessageWithValidation(const std::string& json);
    
    // Streaming parse for incremental JSON processing
    void feedData(const char* data, size_t length);
    bool hasCompleteMessage() const;
    std::unique_ptr<MIDIMessage> extractMessage();
    
    // Performance metrics
    void resetPerformanceCounters();
    double getAverageParseTime() const;  // microseconds
    uint64_t getTotalMessagesProcessed() const;
    
    // Configuration
    void setValidationEnabled(bool enabled) { validationEnabled_ = enabled; }
    void setPerformanceMonitoring(bool enabled) { performanceMonitoring_ = enabled; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    std::atomic<bool> validationEnabled_{true};
    std::atomic<bool> performanceMonitoring_{true};
};

/**
 * JSON Schema Validator for JSONMIDI messages
 */
class SchemaValidator {
public:
    SchemaValidator();
    ~SchemaValidator();
    
    // Load schema from file
    bool loadSchema(const std::string& schemaPath);
    
    // Load schema from JSON string
    bool loadSchemaFromString(const std::string& schemaJson);
    
    // Validate JSON against loaded schema
    ValidationResult validate(const std::string& json) const;
    
    // Validate message object
    ValidationResult validateMessage(const MIDIMessage& message) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Message Factory for creating MIDI messages from JSON
 */
class MessageFactory {
public:
    // Create message from JSON object
    static std::unique_ptr<MIDIMessage> createFromJSON(const std::string& json);
    
    // Create message from raw MIDI bytes
    static std::unique_ptr<MIDIMessage> createFromMIDIBytes(
        const std::vector<uint8_t>& bytes, Timestamp timestamp = Timestamp{});
    
    // Create message from parsed JSON data
    static std::unique_ptr<MIDIMessage> createFromParsedData(
        const std::string& type,
        const nlohmann::json& data);

private:
    // Helper methods for specific message types
    static std::unique_ptr<NoteOnMessage> createNoteOn(
        const nlohmann::json& data);
    static std::unique_ptr<NoteOffMessage> createNoteOff(
        const nlohmann::json& data);
    static std::unique_ptr<ControlChangeMessage> createControlChange(
        const nlohmann::json& data);
    static std::unique_ptr<SystemExclusiveMessage> createSysEx(
        const nlohmann::json& data);
};

/**
 * Performance profiler for parsing operations
 */
class PerformanceProfiler {
public:
    PerformanceProfiler();
    
    // Timing measurement
    class Timer {
    public:
        Timer(PerformanceProfiler& profiler, const std::string& operation);
        ~Timer();
    private:
        PerformanceProfiler& profiler_;
        std::string operation_;
        std::chrono::high_resolution_clock::time_point start_;
    };
    
    void recordTiming(const std::string& operation, Duration duration);
    
    // Statistics
    double getAverageTime(const std::string& operation) const;
    double getMaxTime(const std::string& operation) const;
    double getMinTime(const std::string& operation) const;
    uint64_t getOperationCount(const std::string& operation) const;
    
    void reset();
    std::string generateReport() const;

private:
    struct Stats {
        double totalTime = 0.0;
        double maxTime = 0.0;
        double minTime = std::numeric_limits<double>::max();
        uint64_t count = 0;
    };
    
    mutable std::mutex mutex_;
    std::map<std::string, Stats> stats_;
};

// Macro for easy timing measurement
#define PROFILE_OPERATION(profiler, operation) \
    PerformanceProfiler::Timer timer(profiler, operation)

} // namespace JSONMIDI
