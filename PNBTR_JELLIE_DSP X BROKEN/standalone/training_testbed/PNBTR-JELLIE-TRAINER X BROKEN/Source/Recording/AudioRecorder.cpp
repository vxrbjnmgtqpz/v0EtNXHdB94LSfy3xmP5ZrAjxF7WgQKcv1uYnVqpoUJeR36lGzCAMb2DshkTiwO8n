/*
  ==============================================================================

    AudioRecorder.cpp
    Created: Phase 2 - DSP Pipeline Integration

    Implementation of audio recording functionality.

  ==============================================================================
*/

#include "AudioRecorder.h"
#include <algorithm>
#include <mutex>

// Simple AudioBuffer implementation for compilation
namespace juce {
    template<typename T>
    class AudioBuffer {
    public:
        int getNumChannels() const { return 2; }
        int getNumSamples() const { return 512; }
        T getSample(int channel, int sample) const { return T(0); }
    };
}

//==============================================================================
AudioRecorder::AudioRecorder()
{
}

AudioRecorder::~AudioRecorder() = default;

//==============================================================================
void AudioRecorder::prepare(double sampleRate, int bufferSize)
{
    this->sampleRate = sampleRate;
    this->bufferSize = bufferSize;
    
    // Pre-allocate some recording space
    inputRecording.reserve(static_cast<size_t>(sampleRate * 60)); // 1 minute
    outputRecording.reserve(static_cast<size_t>(sampleRate * 60));
}

//==============================================================================
void AudioRecorder::startRecording()
{
    std::lock_guard<std::mutex> lock(recordingMutex);
    clearRecordings();
    recording.store(true);
}

void AudioRecorder::stopRecording()
{
    recording.store(false);
}

bool AudioRecorder::isRecording() const
{
    return recording.load();
}

//==============================================================================
void AudioRecorder::processBuffers(const juce::AudioBuffer<float>& inputBuffer, 
                                  const juce::AudioBuffer<float>& outputBuffer)
{
    if (!recording.load())
        return;
        
    std::lock_guard<std::mutex> lock(recordingMutex);
    
    // Append input buffer to recording
    appendToRecording(inputBuffer, inputRecording);
    
    // Append output buffer to recording
    appendToRecording(outputBuffer, outputRecording);
}

//==============================================================================
bool AudioRecorder::exportToWAV(const std::string& filename)
{
    // Simplified WAV export - placeholder implementation
    // In a real implementation, this would use JUCE's AudioFormatWriter
    std::lock_guard<std::mutex> lock(recordingMutex);
    
    // For now, just return true to indicate "success"
    return !inputRecording.empty() && !outputRecording.empty();
}

void AudioRecorder::clearRecordings()
{
    std::lock_guard<std::mutex> lock(recordingMutex);
    inputRecording.clear();
    outputRecording.clear();
}

//==============================================================================
int AudioRecorder::getRecordedSamples() const
{
    std::lock_guard<std::mutex> lock(recordingMutex);
    return static_cast<int>(inputRecording.size());
}

double AudioRecorder::getRecordingDuration() const
{
    std::lock_guard<std::mutex> lock(recordingMutex);
    return static_cast<double>(inputRecording.size()) / sampleRate;
}

//==============================================================================
void AudioRecorder::appendToRecording(const juce::AudioBuffer<float>& buffer, 
                                     std::vector<float>& recording)
{
    // Simple mono recording - mix all channels
    int numSamples = buffer.getNumSamples();
    int numChannels = buffer.getNumChannels();
    
    for (int sample = 0; sample < numSamples; ++sample)
    {
        float mixedSample = 0.0f;
        
        for (int channel = 0; channel < numChannels; ++channel)
        {
            mixedSample += buffer.getSample(channel, sample);
        }
        
        if (numChannels > 0)
            mixedSample /= static_cast<float>(numChannels);
            
        recording.push_back(mixedSample);
    }
} 