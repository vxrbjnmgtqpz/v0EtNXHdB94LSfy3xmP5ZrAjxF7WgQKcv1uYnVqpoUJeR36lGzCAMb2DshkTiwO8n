#include "MetalAudioProcessor.h"
#include <iostream>

//==============================================================================
MetalAudioProcessor::MetalAudioProcessor()
{
    // Initialize MetalBridge singleton
    metalBridge = &MetalBridge::getInstance();
    
    // Configure audio format
    setPlayConfigDetails(1, 1, 48000.0, 256); // Mono in/out, 48kHz, 256 samples
    
    // Initialize processing state
    isProcessingActive = false;
    samplesProcessed = 0;
}

MetalAudioProcessor::~MetalAudioProcessor()
{
    // MetalBridge is a singleton, so we don't delete it
}

//==============================================================================
void MetalAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // Update audio format settings
    currentSampleRate = sampleRate;
    currentBufferSize = samplesPerBlock;
    
    // Ensure MetalBridge is initialized for this audio format
    if (metalBridge) {
        // MetalBridge will handle buffer allocation based on these parameters
        DBG("MetalAudioProcessor prepared: " + juce::String(sampleRate) + "Hz, " + juce::String(samplesPerBlock) + " samples");
    }
}

void MetalAudioProcessor::releaseResources()
{
    isProcessingActive = false;
    DBG("MetalAudioProcessor resources released");
}

void MetalAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    
    if (!metalBridge || !isProcessingActive) {
        // If processing is not active, just pass through
        return;
    }
    
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    
    if (numChannels < 1 || numSamples == 0) {
        return;
    }
    
    // Get input and output pointers
    const float* inputData = buffer.getReadPointer(0); // Use first channel as input
    float* outputData = buffer.getWritePointer(0);     // Write to first channel
    
    // Process through MetalBridge GPU pipeline
    metalBridge->processAudioBlock(inputData, outputData, numSamples);
    
    // Update sample counter
    samplesProcessed += numSamples;
    
    // Copy output to all channels if multi-channel
    for (int channel = 1; channel < numChannels; ++channel) {
        buffer.copyFrom(channel, 0, buffer, 0, 0, numSamples);
    }
}

//==============================================================================
// AudioProcessor interface implementation (required overrides)

const juce::String MetalAudioProcessor::getName() const
{
    return "PNBTR+JELLIE Training Processor";
}

bool MetalAudioProcessor::acceptsMidi() const { return false; }
bool MetalAudioProcessor::producesMidi() const { return false; }
bool MetalAudioProcessor::isMidiEffect() const { return false; }
double MetalAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int MetalAudioProcessor::getNumPrograms() { return 1; }
int MetalAudioProcessor::getCurrentProgram() { return 0; }
void MetalAudioProcessor::setCurrentProgram(int index) { juce::ignoreUnused(index); }
const juce::String MetalAudioProcessor::getProgramName(int index) { juce::ignoreUnused(index); return "Default"; }
void MetalAudioProcessor::changeProgramName(int index, const juce::String& newName) { juce::ignoreUnused(index, newName); }

//==============================================================================
// Parameter management (simplified for now)

void MetalAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    // For now, we'll use SessionManager for state
    juce::ignoreUnused(destData);
}

void MetalAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    // For now, we'll use SessionManager for state
    juce::ignoreUnused(data, sizeInBytes);
}

//==============================================================================
// Custom controls for PNBTR+JELLIE system

void MetalAudioProcessor::startProcessing()
{
    isProcessingActive = true;
    samplesProcessed = 0;
    DBG("MetalAudioProcessor started");
}

void MetalAudioProcessor::stopProcessing()
{
    isProcessingActive = false;
    DBG("MetalAudioProcessor stopped, total samples: " + juce::String(samplesProcessed));
}

bool MetalAudioProcessor::isActivelyProcessing() const
{
    return isProcessingActive;
}

AudioMetrics MetalAudioProcessor::getLatestMetrics() const
{
    if (metalBridge) {
        return metalBridge->getLatestMetrics();
    }
    return AudioMetrics{}; // Return default empty metrics
}

void MetalAudioProcessor::setNetworkLossPercentage(float percentage)
{
    if (metalBridge) {
        // TODO: Add parameter setting to MetalBridge
        DBG("Setting packet loss to " + juce::String(percentage) + "%");
    }
}

void MetalAudioProcessor::setJitterAmount(float jitterMs)
{
    if (metalBridge) {
        // TODO: Add parameter setting to MetalBridge  
        DBG("Setting jitter to " + juce::String(jitterMs) + "ms");
    }
}

//==============================================================================
// Audio device integration helpers

bool MetalAudioProcessor::hasMetalSupport() const
{
    return metalBridge != nullptr;
}

double MetalAudioProcessor::getCurrentSampleRate() const
{
    return currentSampleRate;
}

int MetalAudioProcessor::getCurrentBufferSize() const
{
    return currentBufferSize;
}

int MetalAudioProcessor::getSamplesProcessed() const
{
    return samplesProcessed;
} 