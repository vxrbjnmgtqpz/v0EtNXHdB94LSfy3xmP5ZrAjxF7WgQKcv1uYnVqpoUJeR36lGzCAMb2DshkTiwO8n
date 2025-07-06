#pragma once

#include <cstdint>
#include <memory>
#include <functional>
#include <string>

namespace JAMNet {

/**
 * @brief Abstract interface for cross-platform audio output backends
 * 
 * Provides unified interface for JACK (Linux/macOS) and Core Audio (macOS)
 * with GPU-native timing integration.
 */
class AudioOutputBackend {
public:
    struct AudioConfig {
        uint32_t sampleRate = 48000;
        uint32_t bufferSize = 64;
        uint32_t channels = 2;
        std::string deviceName = "default";
        bool enableLowLatency = true;
        bool preferGPUMemory = true;
    };
    
    enum class BackendType {
        JACK,
        CORE_AUDIO,
        ALSA,
        DUMMY
    };
    
    virtual ~AudioOutputBackend() = default;
    
    // Lifecycle management
    virtual bool initialize(const AudioConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual BackendType getType() const = 0;
    virtual std::string getName() const = 0;
    
    // Audio processing
    virtual void pushAudio(const float* data, uint32_t numFrames, 
                          uint64_t timestampNs) = 0;
    virtual void setProcessCallback(std::function<void(float*, uint32_t, uint64_t)> callback) = 0;
    
    // Timing and sync
    virtual uint64_t getCurrentTimeNs() = 0;
    virtual float getActualLatencyMs() const = 0;
    virtual bool supportsGPUMemory() const = 0;
    
    // GPU integration
    virtual bool enableGPUMemoryMode(void* sharedBuffer, size_t bufferSize) = 0;
    virtual void setExternalClock(std::function<uint64_t()> clockCallback) = 0;
    
    // Factory method
    static std::unique_ptr<AudioOutputBackend> create(BackendType preferredType = BackendType::JACK);
    static std::vector<BackendType> getAvailableBackends();
};

} // namespace JAMNet
