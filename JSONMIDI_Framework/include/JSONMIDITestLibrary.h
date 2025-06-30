#pragma once

#include "JSONMIDIMessage.h"
#include <vector>
#include <string>
#include <map>

namespace JSONMIDI {
namespace TestLibrary {

/**
 * Test Message Categories
 */
enum class TestCategory {
    BASIC_CHANNEL_MESSAGES,
    EXTENDED_MIDI2_MESSAGES,
    SYSTEM_MESSAGES,
    EDGE_CASES,
    PERFORMANCE_STRESS,
    PROTOCOL_VALIDATION
};

/**
 * Test Message Definition
 */
struct TestMessage {
    std::string name;
    std::string description;
    TestCategory category;
    Protocol protocol;
    std::string expectedJSON;
    std::vector<uint8_t> expectedMIDIBytes;
    std::unique_ptr<MIDIMessage> messageObject;
    
    // Validation criteria
    bool shouldValidate = true;
    double expectedParseTime = 100.0; // microseconds
    std::string expectedError = "";
};

/**
 * Test Message Library
 */
class MessageLibrary {
public:
    MessageLibrary();
    
    // Get test messages by category
    std::vector<TestMessage> getTestMessages(TestCategory category) const;
    std::vector<TestMessage> getAllTestMessages() const;
    
    // Get specific test messages
    TestMessage getNoteOnTest() const;
    TestMessage getNoteOffTest() const;
    TestMessage getControlChangeTest() const;
    TestMessage getPitchBendTest() const;
    TestMessage getSysExTest() const;
    
    // MIDI 2.0 specific tests
    TestMessage getMIDI2NoteOnTest() const;
    TestMessage getMIDI2ControlChangeTest() const;
    TestMessage getMIDI2SysEx8Test() const;
    
    // Edge case tests
    TestMessage getMaxVelocityTest() const;
    TestMessage getZeroVelocityTest() const;
    TestMessage getInvalidChannelTest() const;
    TestMessage getCorruptedJSONTest() const;
    
    // Performance stress tests
    std::vector<TestMessage> generateStressTestMessages(size_t count) const;
    std::vector<TestMessage> generateLatencyTestMessages() const;
    
    // Round-trip validation
    bool validateRoundTrip(const TestMessage& test) const;
    
    // JSON examples for documentation
    std::map<std::string, std::string> getJSONExamples() const;

private:
    void initializeBasicChannelMessages();
    void initializeMIDI2Messages();
    void initializeSystemMessages();
    void initializeEdgeCases();
    void initializePerformanceTests();
    
    std::map<TestCategory, std::vector<TestMessage>> testMessages_;
};

/**
 * Validation Test Suite
 */
class ValidationTestSuite {
public:
    struct TestResult {
        std::string testName;
        bool passed;
        std::string errorMessage;
        double parseTime;
        double roundTripTime;
    };
    
    struct TestSuiteResult {
        std::vector<TestResult> results;
        size_t totalTests;
        size_t passedTests;
        size_t failedTests;
        double averageParseTime;
        double maxParseTime;
        bool allTestsPassed;
    };
    
    // Run validation tests
    TestSuiteResult runAllTests() const;
    TestSuiteResult runCategoryTests(TestCategory category) const;
    
    // Performance benchmarking
    TestSuiteResult runPerformanceBenchmark(size_t iterations = 1000) const;
    
    // Memory usage tests
    struct MemoryTestResult {
        size_t peakMemoryUsage;
        size_t averageMemoryUsage;
        bool memoryLeakDetected;
    };
    
    MemoryTestResult runMemoryTest(size_t messageCount = 10000) const;
    
    // Generate test report
    std::string generateTestReport(const TestSuiteResult& result) const;
    
private:
    MessageLibrary library_;
    
    TestResult runSingleTest(const TestMessage& test) const;
    double measureParseTime(const std::string& json) const;
    double measureRoundTripTime(const TestMessage& test) const;
};

/**
 * Performance Benchmark Suite
 */
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        std::string benchmarkName;
        double averageTime;
        double minTime;
        double maxTime;
        double standardDeviation;
        uint64_t operationsPerSecond;
        size_t iterations;
    };
    
    // Core parsing benchmarks
    BenchmarkResult benchmarkJSONParsing(size_t iterations = 10000) const;
    BenchmarkResult benchmarkMIDIByteConversion(size_t iterations = 10000) const;
    BenchmarkResult benchmarkRoundTripConversion(size_t iterations = 10000) const;
    
    // Memory efficiency benchmarks
    BenchmarkResult benchmarkMemoryAllocation(size_t iterations = 10000) const;
    BenchmarkResult benchmarkLockFreeQueue(size_t iterations = 100000) const;
    
    // Real-world scenario benchmarks
    BenchmarkResult benchmarkStreamingParse(Duration testDuration) const;
    BenchmarkResult benchmarkHighThroughput(size_t messagesPerSecond, Duration testDuration) const;
    
    // Target validation (from roadmap)
    struct TargetValidation {
        bool parseTimeUnder100us;
        bool throughputAbove10kMps;
        bool memoryUsageAcceptable;
        std::string report;
    };
    
    TargetValidation validatePerformanceTargets() const;
    
private:
    MessageLibrary library_;
    
    std::vector<double> measureOperationTimes(
        std::function<void()> operation, size_t iterations) const;
    
    BenchmarkResult calculateStatistics(
        const std::string& name, const std::vector<double>& times) const;
};

} // namespace TestLibrary
} // namespace JSONMIDI
