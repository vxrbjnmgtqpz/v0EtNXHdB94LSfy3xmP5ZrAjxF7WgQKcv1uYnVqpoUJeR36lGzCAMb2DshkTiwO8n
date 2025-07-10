#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <memory>
#include <atomic>

// Forward declaration
class MetalBridge;
struct AudioMetrics;

// JUCE AudioProcessor that bridges to MetalBridge for GPU processing
class MetalAudioProcessor : public juce::AudioProcessor {
public:
    MetalAudioProcessor();
    ~MetalAudioProcessor() override;

    //==============================================================================
    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    //==============================================================================
    // AudioProcessor metadata
    const juce::String getName() const override { return "MetalAudioProcessor"; }
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
    // Audio parameters
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }

    //==============================================================================
    // Custom controls for PNBTR+JELLIE system
    void startProcessing();
    void stopProcessing();
    bool isActivelyProcessing() const;
    
    // Parameter controls
    void setNetworkLossPercentage(float percentage);
    void setJitterAmount(float jitterMs);
    AudioMetrics getLatestMetrics() const;
    
    // Metal support info
    bool hasMetalSupport() const;
    double getCurrentSampleRate() const;
    int getCurrentBufferSize() const;
    int getSamplesProcessed() const;

private:
    //==============================================================================
    // Metal integration
    MetalBridge* metalBridge;
    
    // Processing state
    bool isProcessingActive;
    int samplesProcessed;
    
    // Audio format parameters
    double currentSampleRate;
    int currentBufferSize;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MetalAudioProcessor)
}; 