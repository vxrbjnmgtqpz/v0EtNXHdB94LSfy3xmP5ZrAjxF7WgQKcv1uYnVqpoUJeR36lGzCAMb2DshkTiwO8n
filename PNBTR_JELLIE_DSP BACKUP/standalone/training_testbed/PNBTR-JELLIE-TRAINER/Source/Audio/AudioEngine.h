/*
  ==============================================================================

    AudioEngine.h
    Created: Game Engine-style Real-Time Audio I/O

    Implements live audio processing pipeline:
    Microphone → JELLIE Encode → Network Simulation → PNBTR Decode → Speakers
    
    Features:
    - JUCE AudioDeviceManager for hardware I/O
    - Double-buffered audio data exchange
    - Real-time safe processing
    - Game engine-style resource management

  ==============================================================================
*/

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <array>

// Forward declarations
class PNBTRTrainer;
class AudioScheduler;

//==============================================================================
/**
 * Game Engine-Style Audio Engine
 * 
 * Manages real-time audio I/O and processing pipeline:
 * - Hardware audio device management
 * - Live microphone input capture
 * - Real-time DSP processing chain
 * - Speaker output with monitoring
 */
class AudioEngine
{
public:
    AudioEngine();
    ~AudioEngine();

    //==============================================================================
    // Game Engine-style lifecycle
    bool initialize(double sampleRate = 48000.0, int bufferSize = 512);
    void shutdown();
    bool isInitialized() const { return initialized.load(); }

    //==============================================================================
    // Audio device management
    struct AudioDeviceInfo {
        std::string inputDeviceName;
        std::string outputDeviceName;
        double sampleRate;
        int bufferSize;
        int inputChannels;
        int outputChannels;
        bool isActive;
    };

    AudioDeviceInfo getDeviceInfo() const;
    std::vector<std::string> getAvailableInputDevices() const;
    std::vector<std::string> getAvailableOutputDevices() const;
    
    bool setInputDevice(const std::string& deviceName);
    bool setOutputDevice(const std::string& deviceName);

    //==============================================================================
    // Real-time audio processing
    void startProcessing();
    void stopProcessing();
    bool isProcessing() const { return processing.load(); }

    //==============================================================================
    // Audio data access (thread-safe double buffering)
    struct AudioBuffer {
        std::vector<float> data;
        size_t channels;
        size_t frames;
        uint64_t timestamp_us;
        bool ready = false;
    };

    // Get latest input audio (for oscilloscope display)
    bool getLatestInputBuffer(AudioBuffer& buffer) const;
    
    // Get latest output audio (for monitoring)
    bool getLatestOutputBuffer(AudioBuffer& buffer) const;

    //==============================================================================
    // Recording system (like DAW)
    void startRecording();
    void stopRecording();
    bool isRecording() const { return recording.load(); }
    
    // Export recorded audio to WAV file
    bool exportRecording(const std::string& filename) const;
    
    // Get recorded data for waveform display
    size_t getRecordedFrames() const { return recordedFrames.load(); }
    bool getRecordedAudio(float* buffer, size_t startFrame, size_t numFrames) const;

    //==============================================================================
    // Performance monitoring
    struct AudioStats {
        double cpuUsage;
        uint32_t underruns;
        uint32_t totalCallbacks;
        float averageCallbackTime_ms;
        float peakCallbackTime_ms;
        uint64_t totalSamplesProcessed;
    };

    AudioStats getStats() const { return stats; }

    //==============================================================================
    // DSP integration
    void setTrainer(PNBTRTrainer* trainer) { this->trainer = trainer; }
    void setScheduler(AudioScheduler* scheduler) { this->scheduler = scheduler; }
    
    //==============================================================================
    // Audio processing pipeline (public for testing)
    void processAudioCallback(const float* inputData, float* outputData, 
                             int numChannels, int numFrames);

private:
    //==============================================================================
    // Double-buffered audio data (game engine pattern)
    static constexpr size_t MAX_BUFFER_SIZE = 2048;
    static constexpr size_t MAX_CHANNELS = 8;
    
    // Input buffers (double buffered)
    std::array<AudioBuffer, 2> inputBuffers;
    std::atomic<int> activeInputBuffer{0};
    
    // Output buffers (double buffered) 
    std::array<AudioBuffer, 2> outputBuffers;
    std::atomic<int> activeOutputBuffer{0};

    //==============================================================================
    // Recording system
    static constexpr size_t MAX_RECORDING_FRAMES = 48000 * 60 * 10; // 10 minutes @ 48kHz
    std::vector<float> recordingBuffer;
    std::atomic<size_t> recordedFrames{0};
    std::atomic<bool> recording{false};


    
    void processInputBuffer(const float* input, int numFrames, int numChannels);
    void processOutputBuffer(float* output, int numFrames, int numChannels);
    
    void updateRecording(const float* input, int numFrames, int numChannels);
    void updateBufferStats(int numFrames);
    void applyNetworkEffects(float* audioData, int numFrames, float packetLoss, float jitter);

    //==============================================================================
    // Audio device implementation
    class AudioCallback; // Forward declaration for implementation
    std::unique_ptr<AudioCallback> audioCallback;
    
    // Device manager (will be implemented with JUCE when build issues resolved)
    void* deviceManager; // Placeholder - will be AudioDeviceManager*
    
    //==============================================================================
    // State and configuration
    std::atomic<bool> initialized{false};
    std::atomic<bool> processing{false};
    
    double currentSampleRate = 48000.0;
    int currentBufferSize = 512;
    int currentInputChannels = 1;
    int currentOutputChannels = 2;

    //==============================================================================
    // Performance monitoring
    mutable AudioStats stats;
    std::atomic<uint64_t> callbackStartTime{0};

    //==============================================================================
    // DSP integration
    PNBTRTrainer* trainer = nullptr;
    AudioScheduler* scheduler = nullptr;

    //==============================================================================
    // Platform-specific optimizations
    void optimizeForRealTime();
    
    // Non-copyable
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
}; 