/*
  ==============================================================================

    PNBTRTrainer.h
    Created: Real GPU-native training harness

    JUCE AudioProcessor that dispatches Metal kernels for:
    Input → JELLIE Encode → Network Simulation → PNBTR Reconstruction → Output

  ==============================================================================
*/


#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>
#include <atomic>

// Forward declarations
class MetalBridge;
class TrainingMetrics;
class PacketLossSimulator;

//==============================================================================
class PNBTRTrainer : public juce::AudioProcessor
{
public:
    PNBTRTrainer();
    ~PNBTRTrainer() override;

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
    void getInputBuffer(float* destination, int numSamples);
    void getOutputBuffer(float* destination, int numSamples);

    // Training state
    bool isTrainingActive() const { return trainingActive.load(); }
    void startTraining();
    void stopTraining();

private:
    //==============================================================================
    // GPU processing pipeline
    std::unique_ptr<TrainingMetrics> metrics;
    std::unique_ptr<PacketLossSimulator> packetLossSimulator;

    //==============================================================================
    // Processing parameters
    double currentSampleRate = 48000.0;
    int currentBlockSize = 512;

    // Training parameters (atomic for thread safety)
    std::atomic<float> packetLossPercentage{2.0f};
    std::atomic<float> jitterAmount{1.0f};
    std::atomic<float> gainDb{0.0f};
    std::atomic<bool> trainingActive{false};

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PNBTRTrainer)
};