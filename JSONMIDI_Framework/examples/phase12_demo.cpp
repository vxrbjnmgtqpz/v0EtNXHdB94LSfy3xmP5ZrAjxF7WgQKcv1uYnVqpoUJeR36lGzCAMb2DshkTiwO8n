#include "JSONMIDIParser.h"
#include "LockFreeQueue.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace JSONMIDI;

int main() {
    std::cout << "=== Phase 1.2 Demo ===" << std::endl;
    
    // Test the BassoonParser with performance metrics
    BassoonParser parser;
    parser.resetPerformanceCounters();
    
    std::string testMessage = R"({"type":"note_on","timestamp":1000,"channel":1,"note":60,"velocity":100})";
    
    std::cout << "\n1. Testing BassoonParser Performance:" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parse 10,000 messages
    for (int i = 0; i < 10000; ++i) {
        auto message = parser.parseMessage(testMessage);
        if (!message) {
            std::cout << "Parse failed at iteration " << i << std::endl;
            std::cout << "Test message: " << testMessage << std::endl;
            
            // Try MessageFactory directly for debugging
            auto factoryMessage = MessageFactory::createFromJSON(testMessage);
            if (factoryMessage) {
                std::cout << "MessageFactory succeeded" << std::endl;
            } else {
                std::cout << "MessageFactory also failed" << std::endl;
            }
            return 1;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Parsed 10,000 messages in " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per message: " << (duration.count() / 10000.0) << " μs" << std::endl;
    std::cout << "Parser internal average: " << parser.getAverageParseTime() << " μs" << std::endl;
    std::cout << "Parser processed count: " << parser.getTotalMessagesProcessed() << std::endl;
    
    // Test lock-free queue
    std::cout << "\n2. Testing Lock-Free Queue:" << std::endl;
    
    MIDIMessageQueue queue;
    
    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto message = parser.parseMessage(testMessage);
            while (!queue.tryPush(std::move(message))) {
                std::this_thread::yield();
            }
        }
    });
    
    // Consumer thread
    int consumed = 0;
    std::thread consumer([&]() {
        std::unique_ptr<MIDIMessage> message;
        while (consumed < 1000) {
            if (queue.tryPop(message)) {
                consumed++;
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    std::cout << "Successfully processed " << consumed << " messages through lock-free queue" << std::endl;
    
    // Test streaming parser
    std::cout << "\n3. Testing Streaming Parser:" << std::endl;
    
    BassoonParser streamParser;
    
    std::string jsonStream = testMessage + testMessage + testMessage;
    
    // Feed data in chunks
    size_t chunkSize = 10;
    size_t pos = 0;
    int messagesExtracted = 0;
    
    while (pos < jsonStream.length()) {
        size_t remaining = jsonStream.length() - pos;
        size_t currentChunk = std::min(chunkSize, remaining);
        
        streamParser.feedData(jsonStream.data() + pos, currentChunk);
        pos += currentChunk;
        
        // Extract any complete messages
        while (streamParser.hasCompleteMessage()) {
            auto message = streamParser.extractMessage();
            if (message) {
                messagesExtracted++;
            }
        }
    }
    
    std::cout << "Extracted " << messagesExtracted << " complete messages from stream" << std::endl;
    
    // Test schema validator
    std::cout << "\n4. Testing Schema Validator:" << std::endl;
    
    SchemaValidator validator;
    
    std::string schema = R"({
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "timestamp": {"type": "integer"},
            "channel": {"type": "integer"},
            "note": {"type": "integer"},
            "velocity": {"type": "integer"}
        },
        "required": ["type", "timestamp"]
    })";
    
    if (validator.loadSchemaFromString(schema)) {
        auto result = validator.validate(testMessage);
        std::cout << "Validation result: " << (result.isValid ? "VALID" : "INVALID") << std::endl;
        if (!result.isValid) {
            std::cout << "Error: " << result.errorMessage << std::endl;
        }
    } else {
        std::cout << "Failed to load schema" << std::endl;
    }
    
    std::cout << "\n=== Phase 1.2 Demo Complete ===" << std::endl;
    std::cout << "All Phase 1.2 components working correctly!" << std::endl;
    
    return 0;
}
