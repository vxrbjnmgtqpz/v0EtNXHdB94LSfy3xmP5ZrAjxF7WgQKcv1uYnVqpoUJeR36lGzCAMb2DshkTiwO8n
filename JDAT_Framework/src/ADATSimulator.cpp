#include "ADATSimulator.h"
#include <algorithm>
#include <numeric>

namespace jdat {

ADATSimulator::ADATSimulator(uint8_t redundancy_level)
    : redundancy_level_(redundancy_level) {
    
    if (redundancy_level_ > 2) {
        redundancy_level_ = 2; // Maximum 2 redundancy streams
    }
}

std::vector<std::vector<float>> ADATSimulator::splitToStreams(const std::vector<float>& mono_audio) {
    std::vector<std::vector<float>> streams(4);
    
    // Calculate stream sizes
    size_t half_size = mono_audio.size() / 2;
    
    // Reserve space for efficiency
    streams[0].reserve(half_size + 1); // Even samples
    streams[1].reserve(half_size + 1); // Odd samples
    streams[2].reserve(half_size + 1); // Redundancy A
    streams[3].reserve(half_size + 1); // Redundancy B
    
    // Split into even/odd sample streams (ADAT-inspired interleaving)
    for (size_t i = 0; i < mono_audio.size(); ++i) {
        if (i % 2 == 0) {
            streams[0].push_back(mono_audio[i]); // Even samples
        } else {
            streams[1].push_back(mono_audio[i]); // Odd samples
        }
    }
    
    // Generate redundancy streams based on level
    if (redundancy_level_ >= 1) {
        generateRedundancyA(streams[0], streams[1], streams[2]);
    }
    
    if (redundancy_level_ >= 2) {
        generateRedundancyB(streams[0], streams[1], streams[3]);
    }
    
    return streams;
}

std::vector<float> ADATSimulator::reconstructFromStreams(const std::vector<std::vector<float>>& streams) {
    if (streams.size() != 4) {
        throw std::invalid_argument("Expected exactly 4 streams for ADAT reconstruction");
    }
    
    const auto& even_stream = streams[0];
    const auto& odd_stream = streams[1];
    const auto& redundancy_a = streams[2];
    const auto& redundancy_b = streams[3];
    
    // Determine which streams are available (non-empty)
    bool has_even = !even_stream.empty();
    bool has_odd = !odd_stream.empty();
    bool has_redundancy_a = !redundancy_a.empty();
    bool has_redundancy_b = !redundancy_b.empty();
    
    // Reconstruct based on available streams
    if (has_even && has_odd) {
        // Best case: we have both primary streams
        return interleaveStreams(even_stream, odd_stream);
    }
    
    if (has_even && has_redundancy_a) {
        // Reconstruct odd samples from redundancy A
        auto reconstructed_odd = reconstructOddFromRedundancyA(even_stream, redundancy_a);
        return interleaveStreams(even_stream, reconstructed_odd);
    }
    
    if (has_odd && has_redundancy_a) {
        // Reconstruct even samples from redundancy A
        auto reconstructed_even = reconstructEvenFromRedundancyA(odd_stream, redundancy_a);
        return interleaveStreams(reconstructed_even, odd_stream);
    }
    
    if (has_redundancy_a && has_redundancy_b) {
        // Reconstruct both streams from redundancy data
        auto reconstructed_streams = reconstructFromRedundancyOnly(redundancy_a, redundancy_b);
        return interleaveStreams(reconstructed_streams.first, reconstructed_streams.second);
    }
    
    // Fallback: use whatever single stream we have
    if (has_even) {
        return expandSingleStream(even_stream);
    }
    
    if (has_odd) {
        return expandSingleStream(odd_stream);
    }
    
    if (has_redundancy_a) {
        return expandSingleStream(redundancy_a);
    }
    
    if (has_redundancy_b) {
        return expandSingleStream(redundancy_b);
    }
    
    // Complete failure - return silence
    return std::vector<float>(960, 0.0f); // 10ms at 96kHz
}

void ADATSimulator::generateRedundancyA(const std::vector<float>& even_samples,
                                       const std::vector<float>& odd_samples,
                                       std::vector<float>& redundancy_a) {
    // Redundancy A: XOR-like parity with weighted combination
    size_t max_size = std::max(even_samples.size(), odd_samples.size());
    redundancy_a.clear();
    redundancy_a.reserve(max_size);
    
    for (size_t i = 0; i < max_size; ++i) {
        float even_val = (i < even_samples.size()) ? even_samples[i] : 0.0f;
        float odd_val = (i < odd_samples.size()) ? odd_samples[i] : 0.0f;
        
        // Generate parity data using weighted difference
        // This allows reconstruction of either stream if the other is available
        float parity = (even_val + odd_val) * 0.5f + (even_val - odd_val) * 0.3f;
        redundancy_a.push_back(parity);
    }
}

void ADATSimulator::generateRedundancyB(const std::vector<float>& even_samples,
                                       const std::vector<float>& odd_samples,
                                       std::vector<float>& redundancy_b) {
    // Redundancy B: Alternative parity with phase shift
    size_t max_size = std::max(even_samples.size(), odd_samples.size());
    redundancy_b.clear();
    redundancy_b.reserve(max_size);
    
    for (size_t i = 0; i < max_size; ++i) {
        float even_val = (i < even_samples.size()) ? even_samples[i] : 0.0f;
        float odd_val = (i < odd_samples.size()) ? odd_samples[i] : 0.0f;
        
        // Generate alternative parity data using different weighting
        // This provides additional redundancy for critical reconstruction
        float parity = (even_val - odd_val) * 0.7f + (even_val + odd_val) * 0.1f;
        redundancy_b.push_back(parity);
    }
}

std::vector<float> ADATSimulator::interleaveStreams(const std::vector<float>& even_samples,
                                                   const std::vector<float>& odd_samples) {
    std::vector<float> result;
    size_t total_size = even_samples.size() + odd_samples.size();
    result.reserve(total_size);
    
    size_t max_pairs = std::max(even_samples.size(), odd_samples.size());
    
    for (size_t i = 0; i < max_pairs; ++i) {
        // Add even sample
        if (i < even_samples.size()) {
            result.push_back(even_samples[i]);
        }
        
        // Add odd sample
        if (i < odd_samples.size()) {
            result.push_back(odd_samples[i]);
        }
    }
    
    return result;
}

std::vector<float> ADATSimulator::reconstructOddFromRedundancyA(const std::vector<float>& even_samples,
                                                               const std::vector<float>& redundancy_a) {
    std::vector<float> odd_samples;
    size_t size = std::min(even_samples.size(), redundancy_a.size());
    odd_samples.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        // Reverse the redundancy A generation formula
        // redundancy_a[i] = (even + odd) * 0.5 + (even - odd) * 0.3
        // Solving for odd: odd = (redundancy_a - even * 0.8) / 0.2
        float reconstructed_odd = (redundancy_a[i] - even_samples[i] * 0.8f) / 0.2f;
        
        // Clamp to reasonable audio range
        reconstructed_odd = std::max(-1.0f, std::min(1.0f, reconstructed_odd));
        
        odd_samples.push_back(reconstructed_odd);
    }
    
    return odd_samples;
}

std::vector<float> ADATSimulator::reconstructEvenFromRedundancyA(const std::vector<float>& odd_samples,
                                                                const std::vector<float>& redundancy_a) {
    std::vector<float> even_samples;
    size_t size = std::min(odd_samples.size(), redundancy_a.size());
    even_samples.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        // Reverse the redundancy A generation formula
        // redundancy_a[i] = (even + odd) * 0.5 + (even - odd) * 0.3
        // Solving for even: even = (redundancy_a + odd * 0.2) / 0.8
        float reconstructed_even = (redundancy_a[i] + odd_samples[i] * 0.2f) / 0.8f;
        
        // Clamp to reasonable audio range
        reconstructed_even = std::max(-1.0f, std::min(1.0f, reconstructed_even));
        
        even_samples.push_back(reconstructed_even);
    }
    
    return even_samples;
}

std::pair<std::vector<float>, std::vector<float>> ADATSimulator::reconstructFromRedundancyOnly(
    const std::vector<float>& redundancy_a,
    const std::vector<float>& redundancy_b) {
    
    std::vector<float> even_samples, odd_samples;
    size_t size = std::min(redundancy_a.size(), redundancy_b.size());
    
    even_samples.reserve(size);
    odd_samples.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        // Solve system of equations:
        // redundancy_a[i] = (even + odd) * 0.5 + (even - odd) * 0.3 = even * 0.8 + odd * 0.2
        // redundancy_b[i] = (even - odd) * 0.7 + (even + odd) * 0.1 = even * 0.8 - odd * 0.6
        
        float ra = redundancy_a[i];
        float rb = redundancy_b[i];
        
        // Solve: 
        // 0.8*even + 0.2*odd = ra
        // 0.8*even - 0.6*odd = rb
        // Subtracting: 0.8*odd = ra - rb, so odd = (ra - rb) / 0.8
        // Substituting back: even = (ra - 0.2*odd) / 0.8
        
        float odd = (ra - rb) / 0.8f;
        float even = (ra - 0.2f * odd) / 0.8f;
        
        // Clamp to reasonable audio range
        even = std::max(-1.0f, std::min(1.0f, even));
        odd = std::max(-1.0f, std::min(1.0f, odd));
        
        even_samples.push_back(even);
        odd_samples.push_back(odd);
    }
    
    return {even_samples, odd_samples};
}

std::vector<float> ADATSimulator::expandSingleStream(const std::vector<float>& single_stream) {
    // When we only have one stream, duplicate samples to create full audio
    // This is a fallback and will have reduced quality
    std::vector<float> result;
    result.reserve(single_stream.size() * 2);
    
    for (float sample : single_stream) {
        result.push_back(sample);  // Even position
        result.push_back(sample);  // Odd position (same value)
    }
    
    return result;
}

} // namespace jdat
