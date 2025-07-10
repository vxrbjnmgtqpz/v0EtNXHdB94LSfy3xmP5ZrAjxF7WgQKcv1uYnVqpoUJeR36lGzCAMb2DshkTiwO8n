/*
  ==============================================================================

    AudioScheduler.h
    Created: Game Engine-style fixed-timestep audio scheduler

    Implements FMOD/Wwise-style high-priority audio thread with:
    - Fixed timestep processing (like Unity's FixedUpdate)
    - Real-time OS priority scheduling
    - Lock-free command buffer for UI→Audio communication
    - Sub-millisecond parameter updates via atomics

  ==============================================================================
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <atomic>
#include <memory>
#include <queue>

// Forward declarations
class PNBTRTrainer;

//==============================================================================
/**
 * Game Engine-Style Audio Scheduler
 * 
 * Inspired by FMOD's 20ms scheduler and Unity's FixedUpdate:
 * - Runs on dedicated high-priority thread
 * - Fixed timestep processing (decoupled from UI frame rate)
 * - Lock-free command buffer for cross-thread communication
 * - Real-time safe parameter updates
 */
class AudioScheduler : public juce::Thread
{
public:
    AudioScheduler(PNBTRTrainer* trainer);
    ~AudioScheduler() override;

    //==============================================================================
    // Game Engine-style lifecycle
    void startAudioEngine();
    void stopAudioEngine();
    bool isAudioEngineRunning() const { return isRunning.load(); }

    //==============================================================================
    // Command Buffer Pattern (like FMOD's command queue)
    enum class CommandType {
        SetPacketLoss,
        SetJitter,
        SetGain,
        StartTraining,
        StopTraining,
        StartRecording,
        StopRecording
    };

    struct AudioCommand {
        CommandType type;
        float value;
        uint64_t timestamp_us;
    };

    // UI Thread: Queue commands (lock-free)
    void queueCommand(CommandType type, float value = 0.0f);

    //==============================================================================
    // Real-time parameters (atomic for sub-ms updates)
    void setPacketLossPercentage(float percentage);
    void setJitterAmount(float jitterMs);
    void setGain(float gainDb);

    float getPacketLossPercentage() const { return packetLossPercentage.load(); }
    float getJitterAmount() const { return jitterAmount.load(); }
    float getGain() const { return gainDb.load(); }

    //==============================================================================
    // Statistics (thread-safe)
    struct AudioStats {
        uint64_t totalBlocks = 0;
        uint64_t totalSamples = 0;
        float averageBlockTime_ms = 0.0f;
        float peakBlockTime_ms = 0.0f;
        uint32_t underruns = 0;
        uint32_t commandsProcessed = 0;
    };

    AudioStats getStats() const { return stats; }

private:
    //==============================================================================
    // Thread implementation (Unity FixedUpdate pattern)
    void run() override;

    //==============================================================================
    // Audio processing core
    void processAudioBlock();
    void processCommands(); // Drain command buffer
    void updateMetrics();

    //==============================================================================
    // Command buffer (lock-free ring buffer)
    static constexpr size_t MAX_COMMANDS = 1024;
    std::array<AudioCommand, MAX_COMMANDS> commandBuffer;
    std::atomic<size_t> commandWriteIndex{0};
    std::atomic<size_t> commandReadIndex{0};

    bool pushCommand(const AudioCommand& cmd);
    bool popCommand(AudioCommand& cmd);

    //==============================================================================
    // Real-time parameters (atomic for zero-latency updates)
    std::atomic<float> packetLossPercentage{2.0f};
    std::atomic<float> jitterAmount{1.0f};
    std::atomic<float> gainDb{0.0f};

    //==============================================================================
    // Training state
    std::atomic<bool> trainingActive{false};
    std::atomic<bool> recordingActive{false};
    std::atomic<bool> isRunning{false};

    //==============================================================================
    // Timing and performance
    double sampleRate = 48000.0;
    int blockSize = 512;
    double blockInterval_ms = 10.67; // Default: 512 samples @ 48kHz
    
    uint64_t nextBlockTime_us = 0;
    // Note: HighResolutionTimer is abstract, using chrono for timing instead

    //==============================================================================
    // Performance monitoring
    mutable AudioStats stats;
    juce::Time lastStatsUpdate;

    //==============================================================================
    // DSP Engine reference
    PNBTRTrainer* trainer;

    //==============================================================================
    // Platform-specific real-time thread setup
    void setRealtimePriority();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioScheduler)
}; 