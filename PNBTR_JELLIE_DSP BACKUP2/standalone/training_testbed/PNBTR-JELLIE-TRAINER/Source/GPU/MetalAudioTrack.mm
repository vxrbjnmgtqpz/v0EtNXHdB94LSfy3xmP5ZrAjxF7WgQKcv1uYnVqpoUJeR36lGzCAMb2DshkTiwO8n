/*
  ==============================================================================

    MetalAudioTrack.mm
    Created: GPU-backed audio track for real-time waveform visualization

    Minimal stub implementation for initial build

  ==============================================================================
*/

#include "MetalAudioTrack.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

//==============================================================================
MetalAudioTrack::MetalAudioTrack(id sourceBuffer, int sampleRate, int bufferLength)
    : metalBuffer(sourceBuffer), sampleRate(sampleRate), bufferLength(bufferLength)
{
    // Initialize Metal device
    metalDevice = MTLCreateSystemDefaultDevice();
    
    // Initialize CPU ring buffer for fallback
    cpuRingBuffer.resize(bufferLength);
    
    // Initialize metrics cache
    cachedRMS.store(0.0f);
    cachedTHD.store(0.0f);
    metricsValid.store(false);
    
    initializeBuffers();
}

MetalAudioTrack::~MetalAudioTrack()
{
    // Note: Using ARC, so no manual memory management needed
    metalDevice = nil;
}

//==============================================================================
void MetalAudioTrack::updateFromGPU()
{
    if (!metalBuffer) return;
    
    // Sync from GPU buffer to CPU ring buffer
    syncFromGPU();
    
    // Mark metrics as invalid for recalculation
    metricsValid.store(false);
}

void MetalAudioTrack::syncFromGPU()
{
    if (!metalBuffer) return;
    
    // Copy data from Metal buffer to CPU ring buffer
    float* gpuData = (float*)[(id<MTLBuffer>)metalBuffer contents];
    
    // Simple copy for now - in full implementation this would be more sophisticated
    for (int i = 0; i < bufferLength && i < cpuRingBuffer.size(); ++i) {
        cpuRingBuffer[i] = gpuData[i];
    }
    
    availableSamples.store(bufferLength);
}

//==============================================================================
float MetalAudioTrack::getSampleAt(int frame) const
{
    if (frame >= 0 && frame < cpuRingBuffer.size()) {
        return cpuRingBuffer[frame];
    }
    return 0.0f;
}

void MetalAudioTrack::getSamples(float* destination, int startFrame, int numFrames) const
{
    if (!destination) return;
    
    for (int i = 0; i < numFrames; ++i) {
        int frameIndex = startFrame + i;
        if (frameIndex >= 0 && frameIndex < cpuRingBuffer.size()) {
            destination[i] = cpuRingBuffer[frameIndex];
        } else {
            destination[i] = 0.0f;
        }
    }
}

//==============================================================================
void MetalAudioTrack::drawWaveform(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    // Simple CPU waveform drawing
    g.setColour(juce::Colours::lime);
    
    int width = (int)bounds.getWidth();
    int height = (int)bounds.getHeight();
    float centerY = bounds.getCentreY();
    
    int samplesPerPixel = std::max(1, (int)cpuRingBuffer.size() / width);
    
    for (int x = 0; x < width; ++x) {
        float maxSample = -1.0f;
        float minSample = 1.0f;
        
        int startSample = x * samplesPerPixel;
        int endSample = std::min(startSample + samplesPerPixel, (int)cpuRingBuffer.size());
        
        for (int i = startSample; i < endSample; ++i) {
            float sample = cpuRingBuffer[i];
            maxSample = std::max(maxSample, sample);
            minSample = std::min(minSample, sample);
        }
        
        float amplitude = (maxSample - minSample) * 0.5f;
        float yPos = centerY - amplitude * height * 0.4f;
        float yPos2 = centerY + amplitude * height * 0.4f;
        
        g.drawVerticalLine(bounds.getX() + x, yPos, yPos2);
    }
}

void MetalAudioTrack::drawWaveformGPU(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    // GPU-accelerated drawing - stub for now
    drawWaveform(g, bounds);
}

//==============================================================================
void MetalAudioTrack::analyzeForMetrics()
{
    updateMetricsCache();
}

float MetalAudioTrack::calculateRMS() const
{
    if (!metricsValid.load()) {
        updateMetricsCache();
    }
    return cachedRMS.load();
}

float MetalAudioTrack::calculateTHD() const
{
    if (!metricsValid.load()) {
        updateMetricsCache();
    }
    return cachedTHD.load();
}

float MetalAudioTrack::calculateSNR(const MetalAudioTrack& reference) const
{
    // Simple SNR calculation - stub implementation
    float signalRMS = calculateRMS();
    float referenceRMS = reference.calculateRMS();
    
    if (referenceRMS > 0.0f) {
        return 20.0f * log10f(signalRMS / referenceRMS);
    }
    return 0.0f;
}

//==============================================================================
bool MetalAudioTrack::exportToWav(const juce::File& outputFile) const
{
    return exportToWav(outputFile, 0, cpuRingBuffer.size());
}

bool MetalAudioTrack::exportToWav(const juce::File& outputFile, int startFrame, int numFrames) const
{
    // WAV export - stub implementation
    // In full implementation this would use JUCE's AudioFormatWriter
    return false;
}

//==============================================================================
void MetalAudioTrack::setBufferLength(int newLength)
{
    bufferLength = newLength;
    cpuRingBuffer.resize(newLength);
    availableSamples.store(0);
}

//==============================================================================
void MetalAudioTrack::initializeBuffers()
{
    // Initialize CPU ring buffer
    cpuRingBuffer.resize(bufferLength);
    std::fill(cpuRingBuffer.begin(), cpuRingBuffer.end(), 0.0f);
    
    writePosition.store(0);
    availableSamples.store(0);
}

void MetalAudioTrack::updateMetricsCache() const
{
    // Calculate RMS
    float rms = 0.0f;
    for (float sample : cpuRingBuffer) {
        rms += sample * sample;
    }
    rms = sqrt(rms / cpuRingBuffer.size());
    cachedRMS.store(rms);
    
    // Calculate THD (simplified)
    cachedTHD.store(0.0f); // Stub
    
    metricsValid.store(true);
}

float MetalAudioTrack::calculatePeakInRange(int startFrame, int numFrames) const
{
    float peak = 0.0f;
    int endFrame = std::min(startFrame + numFrames, (int)cpuRingBuffer.size());
    
    for (int i = startFrame; i < endFrame; ++i) {
        peak = std::max(peak, std::abs(cpuRingBuffer[i]));
    }
    
    return peak;
} 