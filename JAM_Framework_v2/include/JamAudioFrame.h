#pragma once

#include <cstdint>
#include <atomic>

namespace JAMNet {

// Maximum buffer size for audio frames
static constexpr uint32_t JAM_MAX_BUFFER_SIZE = 512;

/**
 * @brief Universal audio frame format for cross-platform compatibility
 * 
 * This structure provides a platform-agnostic representation of audio data
 * with precise timing information and prediction metadata.
 */
struct JamAudioFrame {
    // Audio data (SIMD-aligned for performance)
    alignas(16) float samples[JAM_MAX_BUFFER_SIZE];
    uint32_t numSamples;
    uint32_t sampleRate;
    uint8_t channels;
    uint8_t reserved[3];  // Padding for alignment
    
    // Timing information (critical for latency doctrine compliance)
    uint64_t timestamp_gpu_ns;      ///< GPU clock nanoseconds
    uint64_t timestamp_system_ns;   ///< Correlated system time
    float calibration_offset_ms;    ///< Current drift correction
    
    // Quality and prediction flags
    uint32_t flags;
    static constexpr uint32_t FLAG_CLEAN         = 0x00;  ///< No processing applied
    static constexpr uint32_t FLAG_PREDICTED     = 0x01;  ///< PNBTR filled
    static constexpr uint32_t FLAG_INTERPOLATED  = 0x02;  ///< Gap filled
    static constexpr uint32_t FLAG_SILENCE       = 0x04;  ///< Intentional silence
    static constexpr uint32_t FLAG_DROPOUT       = 0x08;  ///< Network loss
    static constexpr uint32_t FLAG_OVERRUN       = 0x10;  ///< Buffer overrun
    
    // Prediction data (PNBTR)
    float prediction_confidence;    ///< 0.0-1.0
    uint32_t prediction_samples;    ///< How many samples are predicted
    
    // Network metadata (for TOAST)
    uint32_t sequence_number;
    uint32_t source_node_id;
    uint16_t checksum;
    uint16_t reserved2;  // Padding
    
    JamAudioFrame() {
        clear();
    }
    
    void clear() {
        numSamples = 0;
        sampleRate = 48000;
        channels = 2;
        timestamp_gpu_ns = 0;
        timestamp_system_ns = 0;
        calibration_offset_ms = 0.0f;
        flags = FLAG_CLEAN;
        prediction_confidence = 0.0f;
        prediction_samples = 0;
        sequence_number = 0;
        source_node_id = 0;
        checksum = 0;
        // Zero the audio buffer
        for (uint32_t i = 0; i < JAM_MAX_BUFFER_SIZE; ++i) {
            samples[i] = 0.0f;
        }
    }
    
    bool isPredicted() const { return (flags & FLAG_PREDICTED) != 0; }
    bool hasDropout() const { return (flags & FLAG_DROPOUT) != 0; }
    bool isSilence() const { return (flags & FLAG_SILENCE) != 0; }
};

/**
 * @brief Thread-safe ring buffer for audio frames
 * 
 * Lock-free implementation optimized for real-time audio processing.
 */
class SharedAudioBuffer {
private:
    static constexpr size_t RING_BUFFER_SIZE = 8192;  // Power of 2 for fast modulo
    JamAudioFrame frames_[RING_BUFFER_SIZE];
    std::atomic<uint32_t> writeIndex_{0};
    std::atomic<uint32_t> readIndex_{0};
    
public:
    bool pushFrame(const JamAudioFrame& frame);
    bool popFrame(JamAudioFrame& frame);
    bool isEmpty() const;
    bool isFull() const;
    uint32_t getAvailableFrames() const;
    void flush();
    
    // Get read/write indices for debugging
    uint32_t getWriteIndex() const { return writeIndex_.load(); }
    uint32_t getReadIndex() const { return readIndex_.load(); }
};

} // namespace JAMNet
