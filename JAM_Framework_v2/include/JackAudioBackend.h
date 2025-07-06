#pragma once

#include "AudioOutputBackend.h"

#ifdef JAM_ENABLE_JACK
#include <jack/jack.h>
#include <vector>
#include <mutex>

namespace JAMNet {

/**
 * @brief JACK Audio Connection Kit backend implementation
 * 
 * Provides GPU-native JACK integration with external clock injection
 * and memory-mapped GPU buffer support for zero-overhead audio routing.
 */
class JackAudioBackend : public AudioOutputBackend {
private:
    jack_client_t* client_;
    std::vector<jack_port_t*> outputPorts_;
    std::function<void(float*, uint32_t, uint64_t)> processCallback_;
    
    // GPU integration
    void* gpuSharedMemory_;
    size_t gpuBufferSize_;
    bool useGPUMemory_;
    
    // External clock for GPU timing
    std::function<uint64_t()> externalClockCallback_;
    
    // Timing and configuration
    AudioConfig config_;
    uint64_t lastGPUTimestamp_;
    float calibrationOffset_;
    bool initialized_;
    std::mutex callbackMutex_;
    
    // JACK callbacks
    static int jackProcessCallback(jack_nframes_t nframes, void* arg);
    static void jackShutdownCallback(void* arg);
    static int jackSampleRateCallback(jack_nframes_t nframes, void* arg);
    static int jackBufferSizeCallback(jack_nframes_t nframes, void* arg);
    
    // Internal methods
    bool createPorts();
    void destroyPorts();
    uint64_t getJackTimeWithGPUSync();
    
public:
    JackAudioBackend();
    virtual ~JackAudioBackend();
    
    // AudioOutputBackend interface implementation
    bool initialize(const AudioConfig& config) override;
    void shutdown() override;
    BackendType getType() const override { return BackendType::JACK; }
    std::string getName() const override;
    
    void pushAudio(const float* data, uint32_t numFrames, uint64_t timestampNs) override;
    void setProcessCallback(std::function<void(float*, uint32_t, uint64_t)> callback) override;
    
    uint64_t getCurrentTimeNs() override;
    float getActualLatencyMs() const override;
    bool supportsGPUMemory() const override { return useGPUMemory_; }
    
    // GPU integration methods
    bool enableGPUMemoryMode(void* sharedBuffer, size_t bufferSize) override;
    void setExternalClock(std::function<uint64_t()> clockCallback) override;
    
    // JACK-specific methods
    jack_client_t* getJackClient() const { return client_; }
    jack_nframes_t getCurrentBufferSize() const;
    jack_nframes_t getCurrentSampleRate() const;
    bool connectToSystemPorts();
    std::vector<std::string> getAvailablePorts() const;
    
    // Static utility methods
    static bool isJackRunning();
    static std::string getJackServerName();
};

} // namespace JAMNet

#endif // JAM_ENABLE_JACK
