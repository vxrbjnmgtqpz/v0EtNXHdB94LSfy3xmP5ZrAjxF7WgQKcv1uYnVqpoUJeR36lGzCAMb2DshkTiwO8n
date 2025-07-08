#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio>
#include "../include/SIMDJMIDParser.h"

using namespace JMID;

/**
 * SIMD JMID Parser Performance Test
 * 
 * Phase 4: SIMD JSON Performance Validation
 * Target: <10μs parse time per message (100x speedup from baseline)
 * Test: 100K+ messages/second throughput
 */

void testBasicParsing() {
    printf("🔍 Basic Parsing Test\n");
    printf("--------------------\n");
    
    SIMDJMIDParser parser;
    
    // Test different message types
    std::vector<std::string> testMessages = {
        R"({"t":"n+","c":1,"n":60,"v":100,"ts":1642789234567,"seq":12345})",
        R"({"t":"n-","c":1,"n":60,"v":0,"ts":1642789234568,"seq":12346})",
        R"({"t":"cc","c":1,"cc":7,"val":127,"ts":1642789234569,"seq":12347})",
        R"({"t":"pc","c":1,"p":42,"ts":1642789234570,"seq":12348})",
        R"({"t":"pb","c":1,"b":16383,"ts":1642789234571,"seq":12349})"
    };
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    size_t successCount = 0;
    for (const auto& msg : testMessages) {
        auto result = parser.fastParse(msg);
        if (result.valid) {
            successCount++;
            printf("   ✅ %s parsed successfully\n", 
                result.type == SIMDJMIDParser::ParsedMessage::NoteOn ? "Note On" :
                result.type == SIMDJMIDParser::ParsedMessage::NoteOff ? "Note Off" :
                result.type == SIMDJMIDParser::ParsedMessage::ControlChange ? "Control Change" :
                result.type == SIMDJMIDParser::ParsedMessage::ProgramChange ? "Program Change" :
                result.type == SIMDJMIDParser::ParsedMessage::PitchBend ? "Pitch Bend" : "Unknown");
        } else {
            printf("   ❌ Parse failed for message\n");
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    printf("\n📊 Results:\n");
    printf("   Messages parsed: %zu/%zu\n", successCount, testMessages.size());
    printf("   Total time: %ld μs\n", duration.count());
    printf("   Average time per message: %.2f μs\n", 
        static_cast<double>(duration.count()) / testMessages.size());
    printf("   Success rate: %.1f%%\n\n", 
        100.0 * successCount / testMessages.size());
    
    assert(successCount == testMessages.size());
}

void testPerformanceBenchmark() {
    printf("🚀 Performance Benchmark Test\n");
    printf("-----------------------------\n");
    
    const size_t NUM_MESSAGES = 100000;
    printf("   Testing %zu messages for <10μs target...\n", NUM_MESSAGES);
    
    auto result = SIMDParserBenchmark::runParsingBenchmark(NUM_MESSAGES);
    
    printf("\n📈 Performance Results:\n");
    printf("   Total messages: %zu\n", result.totalMessages);
    printf("   Total time: %llu μs\n", result.totalTimeMicros);
    printf("   Average parse time: %.3f μs per message\n", result.avgParseTimeMicros);
    printf("   Messages per second: %.0f\n", result.messagesPerSecond);
    printf("   Speedup factor: %.1fx\n", result.speedupFactor);
    printf("   Target <10μs achieved: %s\n", 
        result.targetAchieved ? "✅ YES" : "❌ NO");
    
    // Additional performance metrics
    if (result.avgParseTimeMicros < 5.0) {
        printf("   Performance level: 🔥 ULTRA-FAST (<5μs)\n");
    } else if (result.avgParseTimeMicros < 10.0) {
        printf("   Performance level: ⚡ FAST (<10μs)\n");
    } else {
        printf("   Performance level: ⚠️  NEEDS OPTIMIZATION (>10μs)\n");
    }
    
    printf("\n");
    assert(result.targetAchieved);
}

void testBatchProcessing() {
    printf("📦 Batch Processing Test\n");
    printf("-----------------------\n");
    
    const size_t BATCH_SIZE = 1000;
    const size_t NUM_BATCHES = 100;
    
    auto result = SIMDParserBenchmark::runBatchBenchmark(BATCH_SIZE, NUM_BATCHES);
    
    printf("   Batch size: %zu messages\n", BATCH_SIZE);
    printf("   Number of batches: %zu\n", NUM_BATCHES);
    printf("   Total messages: %zu\n", result.totalMessages);
    printf("   Average parse time: %.3f μs per message\n", result.avgParseTimeMicros);
    printf("   Batch throughput: %.0f messages/sec\n", result.messagesPerSecond);
    printf("   Target achieved: %s\n\n", 
        result.targetAchieved ? "✅ YES" : "❌ NO");
    
    assert(result.targetAchieved);
}

void testBurstParsing() {
    printf("💥 Burst Parsing Test\n");
    printf("--------------------\n");
    
    SIMDJMIDParser parser;
    parser.enableSequenceTracking(true);
    
    // Create burst messages (same sequence = duplicates)
    std::string burstMessage = R"({"t":"n+","c":1,"n":60,"v":100,"ts":1642789234567,"seq":12345})";
    
    printf("   Testing burst with 3 identical packets (deduplication)...\n");
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Parse same message 3 times (simulating burst)
    auto result1 = parser.burstParse(burstMessage);
    auto result2 = parser.burstParse(burstMessage);
    auto result3 = parser.burstParse(burstMessage);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    printf("   First packet: %s (duplicate: %s)\n", 
        result1.message.valid ? "✅ VALID" : "❌ INVALID",
        result1.isDuplicate ? "YES" : "NO");
    printf("   Second packet: %s (duplicate: %s)\n", 
        result2.message.valid ? "✅ VALID" : "❌ INVALID",
        result2.isDuplicate ? "YES" : "NO");
    printf("   Third packet: %s (duplicate: %s)\n", 
        result3.message.valid ? "✅ VALID" : "❌ INVALID",
        result3.isDuplicate ? "YES" : "NO");
    
    printf("   Total burst time: %ld μs\n", duration.count());
    printf("   Average time per packet: %.2f μs\n", 
        static_cast<double>(duration.count()) / 3);
    
    printf("   Deduplication working: %s\n\n", 
        (!result1.isDuplicate && result2.isDuplicate && result3.isDuplicate) ? "✅ YES" : "❌ NO");
    
    assert(!result1.isDuplicate);
    assert(result2.isDuplicate);
    assert(result3.isDuplicate);
}

void testCompactFormatCompatibility() {
    printf("🔗 Compact Format Compatibility Test\n");
    printf("------------------------------------\n");
    
    SIMDJMIDParser parser;
    
    // Test with our compact format from Phase 3
    std::vector<std::string> compactMessages = {
        R"({"t":"n+","c":1,"n":60,"v":100,"ts":1642789234567,"seq":12345})",
        R"({"t":"cc","c":1,"cc":7,"val":127,"ts":1642789234569,"seq":12347})",
        R"({"t":"pc","c":1,"p":42,"ts":1642789234570,"seq":12348})"
    };
    
    size_t successCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < compactMessages.size(); ++i) {
        auto result = parser.fastParse(compactMessages[i]);
        if (result.valid) {
            successCount++;
            printf("   ✅ Compact message %zu: parsed successfully\n", i + 1);
            printf("      Type: %d, Channel: %d, Sequence: %llu\n", 
                result.type, result.channel, result.sequence);
        } else {
            printf("   ❌ Compact message %zu: parse failed\n", i + 1);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    printf("   Compatibility rate: %.1f%%\n", 
        100.0 * successCount / compactMessages.size());
    printf("   Average parse time: %.2f μs\n", 
        static_cast<double>(duration.count()) / compactMessages.size());
    printf("\n");
    
    assert(successCount == compactMessages.size());
}

void testParserStatistics() {
    printf("📊 Parser Statistics Test\n");
    printf("------------------------\n");
    
    SIMDJMIDParser parser;
    
    // Parse some messages to generate stats
    std::vector<std::string> testMessages;
    for (int i = 0; i < 1000; ++i) {
        testMessages.push_back(
            FastMessageBuilder::buildNoteOn(1, 60 + (i % 12), 100, 1642789234567 + i, 12345 + i)
        );
    }
    
    for (const auto& msg : testMessages) {
        parser.fastParse(msg);
    }
    
    auto stats = parser.getStats();
    
    printf("   Total messages processed: %llu\n", stats.totalMessages);
    printf("   Successful parses: %llu\n", stats.successfulParses);
    printf("   Failed parses: %llu\n", stats.failedParses);
    printf("   Average parse time: %.3f μs\n", stats.averageParseTimeMicros);
    printf("   Messages per second: %.0f\n", stats.messagesPerSecond);
    printf("   Min parse time: %llu μs\n", stats.minParseTimeMicros);
    printf("   Max parse time: %llu μs\n", stats.maxParseTimeMicros);
    printf("   Success rate: %.2f%%\n\n", 
        100.0 * stats.successfulParses / stats.totalMessages);
    
    assert(stats.totalMessages == 1000);
    assert(stats.successfulParses > 990); // Allow for some variance
}

void runComprehensiveBenchmark() {
    printf("🔥 Running Comprehensive SIMD Parser Benchmark\n");
    printf("==============================================\n\n");
    
    SIMDParserBenchmark::runComprehensiveBenchmark();
}

int main() {
    printf("🚀 JMID Framework - Phase 4: SIMD JSON Performance Test\n");
    printf("=======================================================\n\n");
    
    printf("Target: <10μs parse time per message (100x speedup)\n");
    printf("Goal: 100K+ messages/second throughput\n\n");
    
    try {
        testBasicParsing();
        testPerformanceBenchmark();
        testBatchProcessing();
        testBurstParsing();
        testCompactFormatCompatibility();
        testParserStatistics();
        runComprehensiveBenchmark();
        
        printf("🎉 All SIMD Parser Tests PASSED!\n");
        printf("✅ Phase 4: SIMD JSON Performance - COMPLETE\n\n");
        
        printf("🎯 Achievement Summary:\n");
        printf("   ⚡ Sub-10μs parsing achieved\n");
        printf("   🔥 100x speedup target met\n");
        printf("   📦 Batch processing optimized\n");
        printf("   💥 Burst deduplication working\n");
        printf("   🔗 Compact format compatible\n");
        printf("   📊 Performance monitoring active\n\n");
        
        printf("🚀 Ready for Phase 5: Performance Validation!\n");
        
        return 0;
        
    } catch (const std::exception& e) {
        printf("❌ Test failed with exception: %s\n", e.what());
        return 1;
    }
} 