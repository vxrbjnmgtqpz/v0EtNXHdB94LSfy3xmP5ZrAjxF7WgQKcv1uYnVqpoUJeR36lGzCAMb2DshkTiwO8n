//
//
#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <vector>
#include <atomic>
#include <cstddef>
#include <memory>

// Forward declarations
class MetalBridge;
class TrainingMetrics;
class PacketLossSimulator;
class AudioScheduler;

//==============================================================================
class PNBTRTrainer : public juce::AudioProcessor
{
public:
    PNBTRTrainer();
    ~PNBTRTrainer() override;

    // Recording state (atomic, public for GUI control)
    std::atomic<bool> recordingActive{false};

    // Double buffers for oscilloscope/waveform (thread-safe swap)
    std::vector<float> oscInputBufferA, oscInputBufferB;
    std::vector<float> oscOutputBufferA, oscOutputBufferB;
    std::atomic<bool> oscBufferToggle{false};

    // Lock-free circular buffer for recording (JELLIE track)
    std::vector<float> recordBuffer;
    std::atomic<size_t> recordWritePos{0};

    // Thread-safe oscilloscope buffer update (call in processBlock)
    void updateOscilloscopeBuffers(const juce::AudioBuffer<float>& buffer);
    // Thread-safe recording buffer update (call in processBlock)
    void updateRecordingBuffer(const juce::AudioBuffer<float>& buffer);
    // GUI thread: get latest oscilloscope input/output buffer (thread-safe)
    void getLatestOscInput(float* dest, int numSamples);
    void getLatestOscOutput(float* dest, int numSamples);
    // GUI thread: get latest recorded buffer (thread-safe)
    void getRecordedBuffer(float* dest, int numSamples, size_t offset);
    
    // ADDED: Real-time GPU processing data access for waveform visualization
    void getInputBuffer(float* dest, int numSamples);
    void getReconstructedBuffer(float* dest, int numSamples);
    void getSpectralBuffer(float* dest, int numSamples);
    void getNetworkBuffer(float* dest, int numSamples);
    
    // ADDED: GPU processing performance metrics for dashboard
    struct GPUProcessingMetrics {
        float totalLatency_us = 0.0f;
        float gpuLatency_us = 0.0f;
        float averageLatency_us = 0.0f;
        float peakLatency_us = 0.0f;
        float qualityLevel = 1.0f;
        uint32_t samplesProcessed = 0;
        uint32_t fftSize = 1024;
        bool spectralProcessingEnabled = true;
        bool neuralProcessingEnabled = true;
    };
    
    GPUProcessingMetrics getGPUMetrics() const;
    bool isTrainingActive() const { return recordingActive.load(); }

    //==============================================================================
    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    //==============================================================================
    // AudioProcessor metadata
    const juce::String getName() const override { return "PNBTR+JELLIE Trainer"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    // Program management
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int index) override {}
    const juce::String getProgramName(int index) override { return "Default"; }
    void changeProgramName(int index, const juce::String& newName) override {}

    //==============================================================================
    // State management
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    //==============================================================================
    // AudioProcessorEditor interface
    bool hasEditor() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }

    //==============================================================================
    // Training controls
    void setPacketLossPercentage(float percentage);
    void setJitterAmount(float jitterMs);
    void setGain(float gainDb);

    // Metrics access
    TrainingMetrics* getMetrics() const { return metrics.get(); }

    // GPU buffer access for oscilloscope
    void getOutputBuffer(float* destination, int numSamples);

    // Training state
    void startTraining();
    void stopTraining();

    // ADDED: Record arm state management for connecting UI to GPU pipeline
    void setJellieRecordArmed(bool armed) { jellieRecordArmed.store(armed); }
    void setPNBTRRecordArmed(bool armed) { pnbtrRecordArmed.store(armed); }
    bool isJellieRecordArmed() const { return jellieRecordArmed.load(); }
    bool isPNBTRRecordArmed() const { return pnbtrRecordArmed.load(); }

private:
    //==============================================================================
    // GPU processing pipeline
    std::unique_ptr<TrainingMetrics> metrics;
    std::unique_ptr<PacketLossSimulator> packetLossSimulator;

    //==============================================================================
    // Processing parameters
    double currentSampleRate = 48000.0;
    int currentBlockSize = 512;

    // Game Engine-style parameters (atomic for sub-ms latency)
    std::atomic<float> packetLossPercentage{2.0f};
    std::atomic<float> jitterAmount{1.0f};
    std::atomic<float> gainDb{0.0f};
    std::atomic<bool> trainingActive{false};
    
    // Command buffer pattern for UI→Audio thread communication
    enum class AudioCommand { SetPacketLoss, SetJitter, SetGain, StartTraining, StopTraining };
    struct CommandData { AudioCommand cmd; float value; uint64_t timestamp_us; };
    
    // Lock-free command queue (like FMOD's command buffer)
    static constexpr size_t MAX_COMMANDS = 256;
    std::array<CommandData, MAX_COMMANDS> commandBuffer;
    std::atomic<size_t> commandWriteIndex{0};
    std::atomic<size_t> commandReadIndex{0};

    //==============================================================================
    // GPU processing stages
    void runJellieEncode();
    void runNetworkSimulation();
    void runPNBTRReconstruction();
    void updateMetrics();

    //==============================================================================
    // Buffer management
    void initializeGPUBuffers();
    void copyInputToGPU(const juce::AudioBuffer<float>& buffer);
    void copyOutputFromGPU(juce::AudioBuffer<float>& buffer);
    
    // 7-stage pipeline output buffer (CORRECTED: stores processed audio from GPU)
    std::vector<float> processedOutputBuffer;

    // ADDED: Record arm state management for connecting UI to GPU pipeline
    std::atomic<bool> jellieRecordArmed{false};
    std::atomic<bool> pnbtrRecordArmed{false};


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PNBTRTrainer)
};