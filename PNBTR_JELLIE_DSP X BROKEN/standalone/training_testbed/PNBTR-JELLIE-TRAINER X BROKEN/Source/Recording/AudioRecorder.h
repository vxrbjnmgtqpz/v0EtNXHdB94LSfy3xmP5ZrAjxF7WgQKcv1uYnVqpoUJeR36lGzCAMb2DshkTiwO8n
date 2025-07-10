/*
  ==============================================================================

    AudioRecorder.h
    Created: Phase 2 - DSP Pipeline Integration

    Audio Recorder: Handles recording of input and output audio for analysis.

  ==============================================================================
*/

#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <string>

// Forward declaration
namespace juce { template<typename T> class AudioBuffer; }

//==============================================================================
class AudioRecorder
{
public:
    AudioRecorder();
    ~AudioRecorder();

    // Initialization
    void prepare(double sampleRate, int bufferSize);

    // Recording control
    void startRecording();
    void stopRecording();
    bool isRecording() const;

    // Process audio buffers
    void processBuffers(const juce::AudioBuffer<float>& inputBuffer, 
                       const juce::AudioBuffer<float>& outputBuffer);

    // Export functions
    bool exportToWAV(const std::string& filename);
    void clearRecordings();

    // Get recording info
    int getRecordedSamples() const;
    double getRecordingDuration() const;

private:
    // Recording parameters
    double sampleRate = 48000.0;
    int bufferSize = 512;
    std::atomic<bool> recording{false};
    
    // Recording buffers
    std::vector<float> inputRecording;
    std::vector<float> outputRecording;
    
    // Thread safety
    mutable std::mutex recordingMutex;
    
    // Utility functions
    void appendToRecording(const juce::AudioBuffer<float>& buffer, 
                          std::vector<float>& recording);

    // No copy constructor or assignment
    AudioRecorder(const AudioRecorder&) = delete;
    AudioRecorder& operator=(const AudioRecorder&) = delete;
}; 