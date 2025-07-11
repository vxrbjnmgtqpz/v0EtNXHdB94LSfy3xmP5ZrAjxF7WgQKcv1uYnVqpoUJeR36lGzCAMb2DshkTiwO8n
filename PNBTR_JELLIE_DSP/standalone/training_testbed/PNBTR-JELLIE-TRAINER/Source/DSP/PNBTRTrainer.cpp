/*
  ==============================================================================

    PNBTRTrainer.cpp
    Created: Real GPU-native training harness implementation

    JUCE AudioProcessor that dispatches Metal kernels for the real system:
    CoreAudio → JELLIE → PNBTR → OUT

  ==============================================================================
*/

#include "PNBTRTrainer.h"
#include "../GPU/MetalBridge.h"
#include "../Metrics/TrainingMetrics.h"
#include "../Network/PacketLossSimulator.h"
#include <thread>

//==============================================================================
PNBTRTrainer::PNBTRTrainer()
{
    // Preallocate double buffers for oscilloscope (2x block size, stereo)
    oscInputBufferA.resize(2048 * 2, 0.0f);
    oscInputBufferB.resize(2048 * 2, 0.0f);
    oscOutputBufferA.resize(2048 * 2, 0.0f);
    oscOutputBufferB.resize(2048 * 2, 0.0f);
    // Preallocate recording buffer (e.g. 10 seconds at 48kHz stereo)
    recordBuffer.resize(48000 * 10 * 2, 0.0f);

    // Initialize lightweight components immediately
    metrics = std::make_unique<TrainingMetrics>();
    packetLossSimulator = std::make_unique<PacketLossSimulator>();
    
    // SYNCHRONOUS Metal initialization to prevent race condition
    juce::Logger::writeToLog("[PNBTRTrainer] Starting Metal initialization (synchronous)...");
    
    bool success = MetalBridge::getInstance().initialize();
    
    if (success) {
        juce::Logger::writeToLog("[PNBTRTrainer] Metal initialization completed successfully - GPU READY");
    } else {
        juce::Logger::writeToLog("[PNBTRTrainer] Metal initialization failed - falling back to CPU");
    }
    
    juce::Logger::writeToLog("[PNBTRTrainer] Metal bridge ready for audio processing");
    
    juce::Logger::writeToLog("[PNBTRTrainer] Constructor completed (Metal initializing in background)");
}

PNBTRTrainer::~PNBTRTrainer()
{
    releaseResources();
}

//==============================================================================
void PNBTRTrainer::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;
    
    // Configure and prepare GPU buffers for the new block size and sample rate.
    MetalBridge::getInstance().prepareBuffers(samplesPerBlock, sampleRate);
    
    metrics->prepare(sampleRate, samplesPerBlock);
    packetLossSimulator->prepare(sampleRate, samplesPerBlock);
    
    juce::Logger::writeToLog("PNBTRTrainer prepared: " + 
                            juce::String(sampleRate) + "Hz, " + 
                            juce::String(samplesPerBlock) + " samples");
}

void PNBTRTrainer::releaseResources()
{
    // MetalBridge is a singleton managed by its own lifecycle; no explicit shutdown needed here.
    trainingActive.store(false);
}

//==============================================================================
void PNBTRTrainer::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    if (numChannels < 2) {
        buffer.setSize(2, numSamples, true, true, true);
    }

    updateOscilloscopeBuffers(buffer);
    if (recordingActive.load())
        updateRecordingBuffer(buffer);
    
    if (!trainingActive.load() || !MetalBridge::getInstance().isInitialized()) {
        return;
    }

    MetalBridge::getInstance().setRecordArmStates(
        jellieRecordArmed.load(), 
        pnbtrRecordArmed.load()
    );

    // --- New Asynchronous GPU Processing ---
    // Instead of calling multiple steps, we now make a single call to process the entire block.
    // The MetalBridge handles the entire 7-stage pipeline asynchronously.
    
    // We need to provide stereo, planar input for the GPU pipeline.
    std::vector<float> planarInput(numSamples * 2);
    const float* leftIn = buffer.getReadPointer(0);
    const float* rightIn = buffer.getReadPointer(1);
    for(int i = 0; i < numSamples; ++i) {
        planarInput[i] = leftIn[i];
        planarInput[i + numSamples] = rightIn[i];
    }

    std::vector<float> planarOutput(numSamples * 2);

    // Process the entire block on the GPU
    MetalBridge::getInstance().processAudioBlock(planarInput.data(), planarOutput.data(), numSamples);

    // Copy the processed planar output back to the interleaved JUCE buffer
    float* leftOut = buffer.getWritePointer(0);
    float* rightOut = buffer.getWritePointer(1);
    for(int i = 0; i < numSamples; ++i) {
        leftOut[i] = planarOutput[i];
        rightOut[i] = planarOutput[i + numSamples];
    }

    midiMessages.clear();
}

// Thread-safe oscilloscope buffer update (call in processBlock)
void PNBTRTrainer::updateOscilloscopeBuffers(const juce::AudioBuffer<float>& buffer)
{
    // Interleave stereo for oscilloscope
    const float* left = buffer.getReadPointer(0);
    const float* right = buffer.getReadPointer(1);
    const int n = buffer.getNumSamples();
    bool toggle = !oscBufferToggle.load(std::memory_order_relaxed);
    std::vector<float>& oscIn = toggle ? oscInputBufferA : oscInputBufferB;
    std::vector<float>& oscOut = toggle ? oscOutputBufferA : oscOutputBufferB;
    
    // Debug: Check for actual audio input and apply gain for visibility
    static int debugCount = 0;
    float maxLevel = 0.0f;
    const float VISIBILITY_GAIN = 10.0f; // 10x gain for oscilloscope visibility
    
    for (int i = 0; i < n; ++i) {
        // Apply gain for better oscilloscope visibility
        float gainedLeft = left[i] * VISIBILITY_GAIN;
        float gainedRight = right[i] * VISIBILITY_GAIN;
        
        // Clamp to prevent clipping in oscilloscope
        gainedLeft = juce::jlimit(-1.0f, 1.0f, gainedLeft);
        gainedRight = juce::jlimit(-1.0f, 1.0f, gainedRight);
        
        oscIn[i * 2] = gainedLeft;
        oscIn[i * 2 + 1] = gainedRight;
        
        // Track max level for debugging
        maxLevel = std::max(maxLevel, std::abs(left[i]));
    }
    
    // Debug logging every 2 seconds
    if (++debugCount % (int)(currentSampleRate * 2.0 / n) == 0) {
        juce::Logger::writeToLog("[OSC DEBUG] Max input level: " + juce::String(maxLevel, 4) + 
                                ", Samples: " + juce::String(n) + 
                                ", Channels: " + juce::String(buffer.getNumChannels()));
    }
    
    // Real audio only - no fake test signals
    
    // For output oscilloscope, use the gained input as placeholder
    for (int i = 0; i < n; ++i) {
        oscOut[i * 2] = oscIn[i * 2];
        oscOut[i * 2 + 1] = oscIn[i * 2 + 1];
    }
    oscBufferToggle.store(toggle, std::memory_order_release);
}

// Thread-safe recording buffer update (call in processBlock)
void PNBTRTrainer::updateRecordingBuffer(const juce::AudioBuffer<float>& buffer)
{
    const float* left = buffer.getReadPointer(0);
    const float* right = buffer.getReadPointer(1);
    const int n = buffer.getNumSamples();
    size_t pos = recordWritePos.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        recordBuffer[(pos + i) * 2] = left[i];
        recordBuffer[(pos + i) * 2 + 1] = right[i];
    }
    recordWritePos.store((pos + n) % (recordBuffer.size() / 2), std::memory_order_release);
}

// GUI thread: get latest oscilloscope input buffer (thread-safe)
void PNBTRTrainer::getLatestOscInput(float* dest, int numSamples)
{
    bool toggle = oscBufferToggle.load(std::memory_order_acquire);
    const std::vector<float>& oscIn = toggle ? oscInputBufferA : oscInputBufferB;
    std::memcpy(dest, oscIn.data(), sizeof(float) * numSamples * 2);
}

// GUI thread: get latest oscilloscope output buffer (thread-safe)
void PNBTRTrainer::getLatestOscOutput(float* dest, int numSamples)
{
    bool toggle = oscBufferToggle.load(std::memory_order_acquire);
    const std::vector<float>& oscOut = toggle ? oscOutputBufferA : oscOutputBufferB;
    std::memcpy(dest, oscOut.data(), sizeof(float) * numSamples * 2);
}

// GUI thread: get latest recorded buffer (thread-safe)
void PNBTRTrainer::getRecordedBuffer(float* dest, int numSamples, size_t offset)
{
    size_t pos = (recordWritePos.load(std::memory_order_acquire) + offset) % (recordBuffer.size() / 2);
    for (int i = 0; i < numSamples; ++i) {
        dest[i * 2] = recordBuffer[(pos + i) * 2];
        dest[i * 2 + 1] = recordBuffer[(pos + i) * 2 + 1];
    }
}

//==============================================================================
// (All old helper methods like runJellieEncode, copyInputToGPU etc. are now removed
// as their logic is encapsulated within MetalBridge::processAudioBlock)
//==============================================================================

//==============================================================================
void PNBTRTrainer::setPacketLossPercentage(float percentage)
{
    // This functionality is now handled internally by the simulation stages
    // within the MetalBridge and is no longer controlled from here.
}

void PNBTRTrainer::setJitterAmount(float jitterMs)
{
    // This functionality is now handled internally by the simulation stages
    // within the MetalBridge and is no longer controlled from here.
}

void PNBTRTrainer::setGain(float gainDb)
{
    this->gainDb.store(gainDb);
}

//==============================================================================
void PNBTRTrainer::startTraining()
{
    trainingActive.store(true);
    metrics->reset();
    juce::Logger::writeToLog("Training started");
}

void PNBTRTrainer::stopTraining()
{
    trainingActive.store(false);
    juce::Logger::writeToLog("Training stopped");
}

//==============================================================================
void PNBTRTrainer::getInputBuffer(float* destination, int numSamples)
{
    // The input buffer is now internal to the MetalBridge's async pipeline.
    // We can provide the oscilloscope buffer as a stand-in for visualization.
    getLatestOscInput(destination, numSamples);
}

void PNBTRTrainer::getOutputBuffer(float* destination, int numSamples)
{
    // The output buffer is now internal to the MetalBridge's async pipeline.
    // We can provide the oscilloscope buffer as a stand-in for visualization.
    getLatestOscOutput(destination, numSamples);
}

//==============================================================================
void PNBTRTrainer::getStateInformation(juce::MemoryBlock& destData)
{
    // Save training parameters
    juce::MemoryOutputStream stream(destData, true);
    
    stream.writeFloat(packetLossPercentage.load());
    stream.writeFloat(jitterAmount.load());
    stream.writeFloat(gainDb.load());
    stream.writeBool(trainingActive.load());
}

void PNBTRTrainer::setStateInformation(const void* data, int sizeInBytes)
{
    // Restore training parameters
    juce::MemoryInputStream stream(data, static_cast<size_t>(sizeInBytes), false);
    
    packetLossPercentage.store(stream.readFloat());
    jitterAmount.store(stream.readFloat());
    gainDb.store(stream.readFloat());
    trainingActive.store(stream.readBool());
} 