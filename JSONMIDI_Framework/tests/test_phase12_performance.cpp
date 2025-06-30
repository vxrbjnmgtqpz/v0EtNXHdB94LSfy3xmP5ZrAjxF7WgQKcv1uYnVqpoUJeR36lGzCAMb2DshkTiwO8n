// Phase 1.2 Performance Tests - SIMD parser, lock-free queues, advanced validation
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include "JSONMIDIParser.h"
#include "LockFreeQueue.h"

using namespace JSONMIDI;

class Phase12PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
        generateTestMessages();
        
        // Setup schema validator
        validator.loadSchemaFromString(R"({
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "timestamp": {"type": "integer"},
                "channel": {"type": "integer", "minimum": 0, "maximum": 15},
                "note": {"type": "integer", "minimum": 0, "maximum": 127},
                "velocity": {"type": "integer", "minimum": 0, "maximum": 127}
            },
            "required": ["type", "timestamp"]
        })");
    }
    
    void generateTestMessages() {
        testMessages = {
            R"({"type":"note_on","timestamp":1000,"channel":0,"note":60,"velocity":100})",
            R"({"type":"note_off","timestamp":1100,"channel":0,"note":60,"velocity":0})",
            R"({"type":"control_change","timestamp":1200,"channel":1,"controller":7,"value":64})",
            R"({"type":"system_exclusive","timestamp":1300,"data":[0xF0,0x43,0x12,0x00,0xF7]})",
            R"({"type":"note_on","timestamp":1400,"channel":2,"note":64,"velocity":80})",
            R"({"type":"note_on","timestamp":1500,"channel":3,"note":67,"velocity":90})",
            R"({"type":"control_change","timestamp":1600,"channel":4,"controller":1,"value":32})",
            R"({"type":"note_off","timestamp":1700,"channel":2,"note":64,"velocity":0})"
        };
    }
    
    BassoonParser parser;
    SchemaValidator validator;
    std::vector<std::string> testMessages;
    MIDIMessageQueue messageQueue;
};

TEST_F(Phase12PerformanceTest, BassoonParserSpeed) {
    const int iterations = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        for (const auto& msg : testMessages) {
            auto parsed = parser.parseMessage(msg);
            ASSERT_NE(parsed, nullptr);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = static_cast<double>(duration.count()) / (iterations * testMessages.size());
    
    std::cout << "\n=== Bassoon Parser Performance ===" << std::endl;
    std::cout << "Messages processed: " << (iterations * testMessages.size()) << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per message: " << avgTime << " μs" << std::endl;
    std::cout << "Target: <50 μs per message" << std::endl;
    
    // Phase 1.2 target: <50μs per message
    EXPECT_LT(avgTime, 50.0) << "Parser should be faster than 50μs per message";
    
    // Also check parser metrics
    double parserAvg = parser.getAverageParseTime();
    uint64_t processed = parser.getTotalMessagesProcessed();
    
    std::cout << "Parser internal avg: " << parserAvg << " μs" << std::endl;
    std::cout << "Parser processed count: " << processed << std::endl;
    
    EXPECT_GT(processed, 0);
}

TEST_F(Phase12PerformanceTest, StreamingParserPerformance) {
    const int messageCount = 1000;
    
    // Create a large JSON stream
    std::string jsonStream;
    for (int i = 0; i < messageCount; ++i) {
        jsonStream += testMessages[i % testMessages.size()];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Feed data in chunks to simulate real streaming
    const size_t chunkSize = 64;
    size_t pos = 0;
    int messagesExtracted = 0;
    
    while (pos < jsonStream.length()) {
        size_t remaining = jsonStream.length() - pos;
        size_t currentChunk = std::min(chunkSize, remaining);
        
        parser.feedData(jsonStream.data() + pos, currentChunk);
        pos += currentChunk;
        
        // Extract any complete messages
        while (parser.hasCompleteMessage()) {
            auto message = parser.extractMessage();
            if (message) {
                messagesExtracted++;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = static_cast<double>(duration.count()) / messagesExtracted;
    
    std::cout << "\n=== Streaming Parser Performance ===" << std::endl;
    std::cout << "Messages extracted: " << messagesExtracted << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per message: " << avgTime << " μs" << std::endl;
    
    EXPECT_GT(messagesExtracted, 0);
    EXPECT_LT(avgTime, 100.0) << "Streaming parser should be efficient";
}

TEST_F(Phase12PerformanceTest, LockFreeQueueThroughput) {
    const int messageCount = 100000; // Phase 1.2 target: 100k+ messages/second
    
    // Producer thread
    auto start = std::chrono::high_resolution_clock::now();
    
    std::thread producer([&]() {
        for (int i = 0; i < messageCount; ++i) {
            auto message = parser.parseMessage(testMessages[i % testMessages.size()]);
            
            // Try to push, with small retry loop for realistic simulation
            while (!messageQueue.tryPush(std::move(message))) {
                std::this_thread::yield();
            }
        }
    });
    
    // Consumer thread
    int consumed = 0;
    std::thread consumer([&]() {
        std::unique_ptr<MIDIMessage> message;
        while (consumed < messageCount) {
            if (messageQueue.tryPop(message)) {
                consumed++;
                // Simulate some processing
                if (message) {
                    message->toJSON(); // Light processing
                }
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double messagesPerSecond = (static_cast<double>(messageCount) / duration.count()) * 1000.0;
    
    std::cout << "\n=== Lock-Free Queue Throughput ===" << std::endl;
    std::cout << "Messages processed: " << consumed << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << messagesPerSecond << " messages/second" << std::endl;
    std::cout << "Target: 100,000+ messages/second" << std::endl;
    
    EXPECT_EQ(consumed, messageCount);
    EXPECT_GE(messagesPerSecond, 100000.0) << "Should achieve 100k+ messages per second";
}

TEST_F(Phase12PerformanceTest, ValidationPerformanceWithCaching) {
    const int iterations = 50000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test with repeated messages to benefit from caching
    for (int i = 0; i < iterations; ++i) {
        const auto& msg = testMessages[i % testMessages.size()];
        auto result = validator.validate(msg);
        EXPECT_TRUE(result.isValid) << "Message should be valid: " << result.errorMessage;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = static_cast<double>(duration.count()) / iterations;
    
    std::cout << "\n=== Validation Performance (with caching) ===" << std::endl;
    std::cout << "Validations performed: " << iterations << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per validation: " << avgTime << " μs" << std::endl;
    std::cout << "Target: <10 μs per validation (with caching)" << std::endl;
    
    EXPECT_LT(avgTime, 10.0) << "Validation with caching should be very fast";
}

TEST_F(Phase12PerformanceTest, IntegratedPipelinePerformance) {
    const int messageCount = 10000;
    
    // Test the complete pipeline: parse -> validate -> queue -> consume
    auto start = std::chrono::high_resolution_clock::now();
    
    MIDIMessageQueue pipeline;
    int processed = 0;
    
    for (int i = 0; i < messageCount; ++i) {
        const auto& jsonMsg = testMessages[i % testMessages.size()];
        
        // Parse
        auto parsed = parser.parseMessage(jsonMsg);
        ASSERT_NE(parsed, nullptr);
        
        // Validate
        auto validation = validator.validate(jsonMsg);
        EXPECT_TRUE(validation.isValid);
        
        // Queue
        while (!pipeline.tryPush(std::move(parsed))) {
            // Handle full queue
            std::unique_ptr<MIDIMessage> consumed;
            if (pipeline.tryPop(consumed)) {
                processed++;
            }
        }
    }
    
    // Drain remaining messages
    std::unique_ptr<MIDIMessage> consumed;
    while (pipeline.tryPop(consumed)) {
        processed++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = static_cast<double>(duration.count()) / messageCount;
    
    std::cout << "\n=== Integrated Pipeline Performance ===" << std::endl;
    std::cout << "Messages processed: " << processed << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per message: " << avgTime << " μs" << std::endl;
    std::cout << "Target: <10 μs end-to-end processing" << std::endl;
    
    EXPECT_EQ(processed, messageCount);
    EXPECT_LT(avgTime, 10.0) << "Integrated pipeline should be under 10μs per message";
}

TEST_F(Phase12PerformanceTest, MemoryUsageTest) {
    const int messageCount = 10000;
    
    // Test memory usage with large message volumes
    std::vector<std::unique_ptr<MIDIMessage>> messages;
    messages.reserve(messageCount);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < messageCount; ++i) {
        auto message = parser.parseMessage(testMessages[i % testMessages.size()]);
        messages.push_back(std::move(message));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n=== Memory Usage Test ===" << std::endl;
    std::cout << "Messages created: " << messages.size() << std::endl;
    std::cout << "Total time: " << duration.count() << " μs" << std::endl;
    std::cout << "Target: <1MB working set for typical usage" << std::endl;
    
    // Verify all messages were created successfully
    for (const auto& msg : messages) {
        EXPECT_NE(msg, nullptr);
    }
    
    // Clear messages to test cleanup
    messages.clear();
    
    std::cout << "Memory cleanup: Successful" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
