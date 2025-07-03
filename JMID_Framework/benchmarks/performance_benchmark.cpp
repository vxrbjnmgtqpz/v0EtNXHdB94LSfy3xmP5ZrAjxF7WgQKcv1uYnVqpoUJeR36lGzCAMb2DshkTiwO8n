#include "JMIDMessage.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Performance Benchmark - Phase 1.1 Baseline" << std::endl;
    
    const size_t iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        auto timestamp = std::chrono::high_resolution_clock::now();
        JMID::NoteOnMessage noteOn(1, 60, 127, timestamp);
        auto json = noteOn.toJSON();
        auto bytes = noteOn.toMIDIBytes();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Average time per message: " << (duration.count() / iterations) << " μs" << std::endl;
    
    return 0;
}
