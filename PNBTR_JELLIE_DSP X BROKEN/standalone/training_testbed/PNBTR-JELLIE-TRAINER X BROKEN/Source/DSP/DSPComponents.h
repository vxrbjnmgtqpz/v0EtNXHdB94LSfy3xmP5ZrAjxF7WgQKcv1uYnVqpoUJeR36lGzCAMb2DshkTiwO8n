/*
  ==============================================================================

    DSPComponents.h
    Created: Example DSP Components for ECS System

    Demonstrates Unity/Unreal-style DSP components:
    - GainComponent: Volume control
    - JELLIEEncoderComponent: Audio encoding  
    - PNBTRDecoderComponent: Audio decoding/reconstruction
    - FilterComponent: Basic audio filtering
    - CompressorComponent: Dynamic range compression

    All components are hot-swappable without audio interruption.

  ==============================================================================
*/

#pragma once

#include "DSPEntitySystem.h"
#include <atomic>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>

//==============================================================================
/**
 * Gain Component - Basic volume control
 * Like Unity's AudioSource volume or Unreal's Audio Component gain
 */
class GainComponent : public DSPComponent {
public:
    GainComponent(EntityID entity, float initialGain = 1.0f) 
        : DSPComponent(entity), gain(initialGain) {}
    
    void processAudio(AudioBlock& input, AudioBlock& output) override {
        float currentGain = gain.load();
        
        if (currentGain == 1.0f) {
            // No gain change - just copy
            output.copyFrom(input);
        } else if (currentGain == 0.0f) {
            // Muted - silence
            output.clearToSilence();
        } else {
            // Apply gain
            output.copyFrom(input);
            output.applyGain(currentGain);
        }
        
        updatePerformanceStats();
    }
    
    void setParameter(const std::string& name, float value) override {
        if (name == "gain" || name == "volume") {
            gain.store(std::max(0.0f, value)); // Clamp to non-negative
        }
    }
    
    float getParameter(const std::string& name) const override {
        if (name == "gain" || name == "volume") {
            return gain.load();
        }
        return 0.0f;
    }
    
    std::string getName() const override { return "GainComponent"; }

private:
    std::atomic<float> gain{1.0f};
    
    void updatePerformanceStats() {
        processCallCount.fetch_add(1);
        // Simple performance tracking
    }
};

//==============================================================================
/**
 * JELLIE Encoder Component - Audio compression/encoding
 * Simulates JELLIE encoding process with network-aware compression
 */
class JELLIEEncoderComponent : public DSPComponent {
public:
    JELLIEEncoderComponent(EntityID entity) : DSPComponent(entity) {}
    
    void initialize(double sampleRate, size_t maxBufferSize) override {
        this->sampleRate = sampleRate;
        
        // Initialize JELLIE encoder state
        compressionRatio.store(4.0f); // 4:1 compression by default
        adaptiveMode.store(true);
        
        // Allocate internal buffers
        encoderBuffer.resize(maxBufferSize * 2);
        
        std::cout << "[JELLIEEncoder] Initialized at " << sampleRate << "Hz" << std::endl;
    }
    
    void processAudio(AudioBlock& input, AudioBlock& output) override {
        float ratio = compressionRatio.load();
        bool adaptive = adaptiveMode.load();
        
        output.copyFrom(input);
        
        // Simulate JELLIE encoding process
        // In real implementation, this would:
        // 1. Analyze audio content
        // 2. Apply perceptual compression
        // 3. Prepare for network transmission
        
        if (adaptive) {
            // Adaptive compression based on content
            simulateAdaptiveEncoding(output, ratio);
        } else {
            // Fixed compression ratio
            simulateFixedEncoding(output, ratio);
        }
        
        updatePerformanceStats();
    }
    
    void setParameter(const std::string& name, float value) override {
        if (name == "compression_ratio") {
            compressionRatio.store(std::max(1.0f, std::min(value, 16.0f)));
        } else if (name == "adaptive_mode") {
            adaptiveMode.store(value > 0.5f);
        } else if (name == "network_quality") {
            // Adjust compression based on network conditions
            float networkRatio = 2.0f + (1.0f - value) * 6.0f; // 2:1 to 8:1 based on quality
            compressionRatio.store(networkRatio);
        }
    }
    
    float getParameter(const std::string& name) const override {
        if (name == "compression_ratio") {
            return compressionRatio.load();
        } else if (name == "adaptive_mode") {
            return adaptiveMode.load() ? 1.0f : 0.0f;
        }
        return 0.0f;
    }
    
    std::string getName() const override { return "JELLIEEncoderComponent"; }
    size_t getLatencyFrames() const override { return 64; } // Encoding latency

private:
    std::atomic<float> compressionRatio{4.0f};
    std::atomic<bool> adaptiveMode{true};
    
    double sampleRate = 48000.0;
    std::vector<float> encoderBuffer;
    
    void simulateAdaptiveEncoding(AudioBlock& block, float baseRatio) {
        // Simulate adaptive compression based on audio content
        
        // Calculate RMS level for dynamic adjustment
        float rmsLevel = 0.0f;
        for (size_t ch = 0; ch < block.numChannels; ++ch) {
            if (block.channels[ch]) {
                for (size_t frame = 0; frame < block.numFrames; ++frame) {
                    float sample = block.channels[ch][frame];
                    rmsLevel += sample * sample;
                }
            }
        }
        rmsLevel = std::sqrt(rmsLevel / (block.numFrames * block.numChannels));
        
        // Adjust compression based on content
        float dynamicRatio = baseRatio;
        if (rmsLevel > 0.5f) {
            dynamicRatio *= 0.8f; // Less compression for loud signals
        } else if (rmsLevel < 0.1f) {
            dynamicRatio *= 1.2f; // More compression for quiet signals
        }
        
        // Apply simulated compression artifacts
        float compressionGain = 1.0f / dynamicRatio;
        block.applyGain(compressionGain);
    }
    
    void simulateFixedEncoding(AudioBlock& block, float ratio) {
        // Simple fixed-ratio compression simulation
        float compressionGain = 1.0f / ratio;
        block.applyGain(compressionGain);
    }
    
    void updatePerformanceStats() {
        processCallCount.fetch_add(1);
    }
};

//==============================================================================
/**
 * PNBTR Decoder Component - Audio reconstruction/decoding
 * Simulates PNBTR neural network decoding with quality enhancement
 */
class PNBTRDecoderComponent : public DSPComponent {
public:
    PNBTRDecoderComponent(EntityID entity) : DSPComponent(entity) {}
    
    void initialize(double sampleRate, size_t maxBufferSize) override {
        this->sampleRate = sampleRate;
        
        // Initialize PNBTR decoder state
        enhancementLevel.store(0.7f);
        noiseReduction.store(true);
        
        // Allocate internal processing buffers
        decoderBuffer.resize(maxBufferSize * 2);
        historyBuffer.resize(512); // For temporal processing
        
        std::cout << "[PNBTRDecoder] Initialized at " << sampleRate << "Hz" << std::endl;
    }
    
    void processAudio(AudioBlock& input, AudioBlock& output) override {
        float enhancement = enhancementLevel.load();
        bool noiseReduct = noiseReduction.load();
        
        output.copyFrom(input);
        
        // Simulate PNBTR neural network processing
        // In real implementation, this would:
        // 1. Reconstruct missing audio data
        // 2. Enhance audio quality  
        // 3. Reduce compression artifacts
        // 4. Apply perceptual improvements
        
        if (noiseReduct) {
            simulateNoiseReduction(output);
        }
        
        if (enhancement > 0.0f) {
            simulateQualityEnhancement(output, enhancement);
        }
        
        updatePerformanceStats();
    }
    
    void setParameter(const std::string& name, float value) override {
        if (name == "enhancement_level") {
            enhancementLevel.store(std::max(0.0f, std::min(value, 1.0f)));
        } else if (name == "noise_reduction") {
            noiseReduction.store(value > 0.5f);
        } else if (name == "network_quality") {
            // Adjust enhancement based on network conditions
            float qualityLevel = value; // 0.0 = poor network, 1.0 = good network
            enhancementLevel.store(1.0f - qualityLevel); // More enhancement for poor quality
        }
    }
    
    float getParameter(const std::string& name) const override {
        if (name == "enhancement_level") {
            return enhancementLevel.load();
        } else if (name == "noise_reduction") {
            return noiseReduction.load() ? 1.0f : 0.0f;
        }
        return 0.0f;
    }
    
    std::string getName() const override { return "PNBTRDecoderComponent"; }
    size_t getLatencyFrames() const override { return 128; } // Neural network latency
    
    float getCPUUsage() const override {
        // Simulate neural network CPU usage
        return 0.15f + (enhancementLevel.load() * 0.25f); // 15-40% CPU usage
    }

private:
    std::atomic<float> enhancementLevel{0.7f};
    std::atomic<bool> noiseReduction{true};
    
    double sampleRate = 48000.0;
    std::vector<float> decoderBuffer;
    std::vector<float> historyBuffer;
    
    void simulateNoiseReduction(AudioBlock& block) {
        // Simple noise gate simulation
        const float noiseThreshold = 0.01f;
        
        for (size_t ch = 0; ch < block.numChannels; ++ch) {
            if (block.channels[ch]) {
                for (size_t frame = 0; frame < block.numFrames; ++frame) {
                    float& sample = block.channels[ch][frame];
                    if (std::abs(sample) < noiseThreshold) {
                        sample *= 0.1f; // Reduce noise floor
                    }
                }
            }
        }
    }
    
    void simulateQualityEnhancement(AudioBlock& block, float enhancement) {
        // Simulate neural network quality enhancement
        
        for (size_t ch = 0; ch < block.numChannels; ++ch) {
            if (block.channels[ch]) {
                for (size_t frame = 0; frame < block.numFrames; ++frame) {
                    float& sample = block.channels[ch][frame];
                    
                    // Simulate harmonic enhancement
                    float enhanced = sample;
                    if (std::abs(sample) > 0.1f) {
                        enhanced += sample * sample * sample * enhancement * 0.1f; // Add subtle harmonics
                    }
                    
                    // Soft clipping to prevent overload
                    sample = std::tanh(enhanced);
                }
            }
        }
    }
    
    void updatePerformanceStats() {
        processCallCount.fetch_add(1);
        
        // Update average processing time
        float cpuUsage = getCPUUsage();
        averageProcessTime_us.store(cpuUsage * 100.0f); // Simulate processing time
    }
};

//==============================================================================
/**
 * Filter Component - Basic audio filtering
 * Demonstrates modularity and parameter control
 */
class FilterComponent : public DSPComponent {
public:
    enum FilterType {
        LowPass,
        HighPass,
        BandPass,
        Notch
    };
    
    FilterComponent(EntityID entity, FilterType type = LowPass) 
        : DSPComponent(entity), filterType(type) {}
    
    void initialize(double sampleRate, size_t maxBufferSize) override {
        this->sampleRate = sampleRate;
        
        // Initialize filter coefficients
        cutoffFrequency.store(1000.0f);
        resonance.store(0.7f);
        updateFilterCoefficients();
        
        // Clear filter state
        for (auto& state : filterState) {
            state.fill(0.0f);
        }
    }
    
    void processAudio(AudioBlock& input, AudioBlock& output) override {
        output.copyFrom(input);
        
        float cutoff = cutoffFrequency.load();
        float q = resonance.load();
        
        // Simple biquad filter implementation
        for (size_t ch = 0; ch < output.numChannels && ch < MAX_AUDIO_CHANNELS; ++ch) {
            if (output.channels[ch]) {
                processChannel(output.channels[ch], output.numFrames, ch);
            }
        }
        
        updatePerformanceStats();
    }
    
    void setParameter(const std::string& name, float value) override {
        if (name == "cutoff" || name == "frequency") {
            cutoffFrequency.store(std::max(20.0f, std::min(value, 20000.0f)));
            updateFilterCoefficients();
        } else if (name == "resonance" || name == "q") {
            resonance.store(std::max(0.1f, std::min(value, 10.0f)));
            updateFilterCoefficients();
        }
    }
    
    float getParameter(const std::string& name) const override {
        if (name == "cutoff" || name == "frequency") {
            return cutoffFrequency.load();
        } else if (name == "resonance" || name == "q") {
            return resonance.load();
        }
        return 0.0f;
    }
    
    std::string getName() const override { 
        switch (filterType) {
            case LowPass: return "LowPassFilterComponent";
            case HighPass: return "HighPassFilterComponent";
            case BandPass: return "BandPassFilterComponent";
            case Notch: return "NotchFilterComponent";
            default: return "FilterComponent";
        }
    }

private:
    FilterType filterType;
    std::atomic<float> cutoffFrequency{1000.0f};
    std::atomic<float> resonance{0.7f};
    
    double sampleRate = 48000.0;
    
    // Biquad filter coefficients
    std::atomic<float> a0{1.0f}, a1{0.0f}, a2{0.0f}, b1{0.0f}, b2{0.0f};
    
    // Filter state per channel (x1, x2, y1, y2)
    std::array<std::array<float, 4>, MAX_AUDIO_CHANNELS> filterState;
    
    void updateFilterCoefficients() {
        float freq = cutoffFrequency.load();
        float q = resonance.load();
        
        // Calculate biquad coefficients based on filter type
        float omega = 2.0f * M_PI * freq / sampleRate;
        float sin_omega = std::sin(omega);
        float cos_omega = std::cos(omega);
        float alpha = sin_omega / (2.0f * q);
        
        float b0, b1_coeff, b2_coeff, a0_coeff, a1_coeff, a2_coeff;
        
        switch (filterType) {
            case LowPass:
                b0 = (1.0f - cos_omega) / 2.0f;
                b1_coeff = 1.0f - cos_omega;
                b2_coeff = (1.0f - cos_omega) / 2.0f;
                a0_coeff = 1.0f + alpha;
                a1_coeff = -2.0f * cos_omega;
                a2_coeff = 1.0f - alpha;
                break;
                
            case HighPass:
                b0 = (1.0f + cos_omega) / 2.0f;
                b1_coeff = -(1.0f + cos_omega);
                b2_coeff = (1.0f + cos_omega) / 2.0f;
                a0_coeff = 1.0f + alpha;
                a1_coeff = -2.0f * cos_omega;
                a2_coeff = 1.0f - alpha;
                break;
                
            default:
                // Pass through for unsupported types
                b0 = 1.0f; b1_coeff = 0.0f; b2_coeff = 0.0f;
                a0_coeff = 1.0f; a1_coeff = 0.0f; a2_coeff = 0.0f;
                break;
        }
        
        // Normalize coefficients
        a0.store(b0 / a0_coeff);
        a1.store(b1_coeff / a0_coeff);
        a2.store(b2_coeff / a0_coeff);
        b1.store(a1_coeff / a0_coeff);
        b2.store(a2_coeff / a0_coeff);
    }
    
    void processChannel(float* samples, size_t numFrames, size_t channel) {
        auto& state = filterState[channel];
        
        float coeff_a0 = a0.load();
        float coeff_a1 = a1.load();
        float coeff_a2 = a2.load();
        float coeff_b1 = b1.load();
        float coeff_b2 = b2.load();
        
        for (size_t i = 0; i < numFrames; ++i) {
            float input = samples[i];
            
            // Biquad filter processing
            float output = coeff_a0 * input + coeff_a1 * state[0] + coeff_a2 * state[1]
                         - coeff_b1 * state[2] - coeff_b2 * state[3];
            
            // Update state
            state[1] = state[0];  // x2 = x1
            state[0] = input;     // x1 = input
            state[3] = state[2];  // y2 = y1
            state[2] = output;    // y1 = output
            
            samples[i] = output;
        }
    }
    
    void updatePerformanceStats() {
        processCallCount.fetch_add(1);
    }
}; 