/*
  ==============================================================================

    AudioScheduler.cpp
    Created: Game Engine-style fixed-timestep audio scheduler implementation

    Implements high-priority audio thread with lock-free command processing
    and atomic parameter updates for sub-millisecond latency.

  ==============================================================================
*/

#include "AudioScheduler.h"
#include "PNBTRTrainer.h"
#include <chrono>
#include <thread>
#include <iostream>

#ifdef __APPLE__
#include <pthread.h>
#include <sched.h>
#endif

//==============================================================================
AudioScheduler::AudioScheduler(PNBTRTrainer* trainer)
    : Thread("AudioScheduler"), trainer(trainer)
{
    // Calculate precise timing based on sample rate and block size
    if (trainer) {
        sampleRate = 48000.0; // Will be updated in prepare()
        blockSize = 512;
        blockInterval_ms = (blockSize / sampleRate) * 1000.0;
    }
    
    std::cout << "[AudioScheduler] Created with " << blockSize << " samples @ " 
              << sampleRate << "Hz (" << blockInterval_ms << "ms per block)" << std::endl;
}

AudioScheduler::~AudioScheduler()
{
    stopAudioEngine();
}

//==============================================================================
void AudioScheduler::startAudioEngine()
{
    if (isRunning.load()) {
        std::cout << "[AudioScheduler] Already running" << std::endl;
        return;
    }

    std::cout << "[AudioScheduler] Starting game engine-style audio thread..." << std::endl;
    
    isRunning.store(true);
    nextBlockTime_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Start high-priority thread (like FMOD's audio mixer thread)
    startThread(Thread::Priority::highest);
    
    std::cout << "[AudioScheduler] Audio engine started with real-time priority" << std::endl;
}

void AudioScheduler::stopAudioEngine()
{
    if (!isRunning.load()) {
        return;
    }

    std::cout << "[AudioScheduler] Stopping audio engine..." << std::endl;
    
    isRunning.store(false);
    signalThreadShouldExit();
    
    // Wait for thread to finish gracefully
    if (isThreadRunning()) {
        waitForThreadToExit(1000); // 1 second timeout
    }
    
    std::cout << "[AudioScheduler] Audio engine stopped. Stats: " 
              << stats.totalBlocks << " blocks, " 
              << stats.averageBlockTime_ms << "ms avg" << std::endl;
}

//==============================================================================
// Unity FixedUpdate-style main loop
void AudioScheduler::run()
{
    setRealtimePriority();
    
    std::cout << "[AudioScheduler] High-priority audio thread started" << std::endl;
    
    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!threadShouldExit() && isRunning.load()) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Process one audio block (like Unity's FixedUpdate)
        processCommands();    // Drain command buffer first
        processAudioBlock();  // Core DSP processing
        updateMetrics();      // Performance monitoring
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto blockTime_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        
        // Update performance stats
        stats.totalBlocks++;
        stats.totalSamples += blockSize;
        float blockTime_ms = blockTime_us / 1000.0f;
        stats.averageBlockTime_ms = (stats.averageBlockTime_ms * 0.95f) + (blockTime_ms * 0.05f);
        stats.peakBlockTime_ms = std::max(stats.peakBlockTime_ms, blockTime_ms);
        
        // Check for underruns (block took longer than available time)
        if (blockTime_ms > blockInterval_ms) {
            stats.underruns++;
            if (stats.underruns % 10 == 1) { // Log every 10th underrun
                std::cout << "[AudioScheduler] WARNING: Audio underrun! Block took " 
                          << blockTime_ms << "ms (limit: " << blockInterval_ms << "ms)" << std::endl;
            }
        }
        
        // Fixed timestep sleep (like FMOD's 20ms scheduler)
        auto sleepTime_us = static_cast<int64_t>(blockInterval_ms * 1000.0) - blockTime_us;
        if (sleepTime_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime_us));
        }
    }
    
    std::cout << "[AudioScheduler] Audio thread exited cleanly" << std::endl;
}

//==============================================================================
void AudioScheduler::processCommands()
{
    // Drain command buffer (lock-free)
    AudioCommand cmd;
    int commandsThisBlock = 0;
    
    while (popCommand(cmd) && commandsThisBlock < 32) { // Limit commands per block
        switch (cmd.type) {
            case CommandType::SetPacketLoss:
                packetLossPercentage.store(cmd.value);
                break;
            case CommandType::SetJitter:
                jitterAmount.store(cmd.value);
                break;
            case CommandType::SetGain:
                gainDb.store(cmd.value);
                break;
            case CommandType::StartTraining:
                trainingActive.store(true);
                break;
            case CommandType::StopTraining:
                trainingActive.store(false);
                break;
            case CommandType::StartRecording:
                recordingActive.store(true);
                break;
            case CommandType::StopRecording:
                recordingActive.store(false);
                break;
        }
        commandsThisBlock++;
    }
    
    stats.commandsProcessed += commandsThisBlock;
}

void AudioScheduler::processAudioBlock()
{
    if (!trainer || !trainingActive.load()) {
        return; // Skip processing if trainer not available or inactive
    }
    
    // 🎮 VIDEO GAME ENGINE: Generate real audio block and process it
    juce::AudioBuffer<float> audioBuffer(2, blockSize); // Stereo buffer
    juce::MidiBuffer midiBuffer;
    
    // Generate test audio or get from audio input (for now, use silence)
    audioBuffer.clear();
    
    // Call the actual DSP processing pipeline
    try {
        trainer->processBlock(audioBuffer, midiBuffer);
        std::cout << "[AudioScheduler] Processed audio block: " << blockSize << " samples" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[AudioScheduler] Exception in processBlock: " << e.what() << std::endl;
    }
}

void AudioScheduler::updateMetrics()
{
    // Update metrics every 100 blocks (~1 second at 10ms blocks)
    if (stats.totalBlocks % 100 == 0) {
        // Could update additional metrics here
        // This runs on audio thread, so keep it minimal
    }
}

//==============================================================================
// Lock-free command buffer implementation
void AudioScheduler::queueCommand(CommandType type, float value)
{
    AudioCommand cmd;
    cmd.type = type;
    cmd.value = value;
    cmd.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    if (!pushCommand(cmd)) {
        std::cout << "[AudioScheduler] WARNING: Command buffer full, dropping command" << std::endl;
    }
}

bool AudioScheduler::pushCommand(const AudioCommand& cmd)
{
    size_t writeIdx = commandWriteIndex.load();
    size_t nextIdx = (writeIdx + 1) % MAX_COMMANDS;
    
    if (nextIdx == commandReadIndex.load()) {
        return false; // Buffer full
    }
    
    commandBuffer[writeIdx] = cmd;
    commandWriteIndex.store(nextIdx);
    return true;
}

bool AudioScheduler::popCommand(AudioCommand& cmd)
{
    size_t readIdx = commandReadIndex.load();
    
    if (readIdx == commandWriteIndex.load()) {
        return false; // Buffer empty
    }
    
    cmd = commandBuffer[readIdx];
    commandReadIndex.store((readIdx + 1) % MAX_COMMANDS);
    return true;
}

//==============================================================================
// Atomic parameter updates (sub-millisecond latency)
void AudioScheduler::setPacketLossPercentage(float percentage)
{
    queueCommand(CommandType::SetPacketLoss, percentage);
}

void AudioScheduler::setJitterAmount(float jitterMs)
{
    queueCommand(CommandType::SetJitter, jitterMs);
}

void AudioScheduler::setGain(float gainDb)
{
    queueCommand(CommandType::SetGain, gainDb);
}

//==============================================================================
// Platform-specific real-time thread setup
void AudioScheduler::setRealtimePriority()
{
#ifdef __APPLE__
    // Set real-time priority on macOS (like Core Audio)
    struct sched_param param;
    param.sched_priority = 63; // High priority for audio
    
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::cout << "[AudioScheduler] WARNING: Could not set real-time priority" << std::endl;
    } else {
        std::cout << "[AudioScheduler] Set real-time thread priority (SCHED_FIFO, 63)" << std::endl;
    }
#else
    // Add Windows/Linux real-time priority setup here
    std::cout << "[AudioScheduler] Real-time priority not implemented for this platform" << std::endl;
#endif
} 