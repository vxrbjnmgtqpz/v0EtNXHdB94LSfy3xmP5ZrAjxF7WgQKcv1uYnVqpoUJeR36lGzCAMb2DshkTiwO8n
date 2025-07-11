#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>

/**
 * Spectral Audio Track Component
 * 
 * Combines waveform visualization with real-time spectral analysis.
 * Designed for JELLIE (recorded input) and PNBTR (reconstructed output) tracks.
 * 
 * Features:
 * - Real-time waveform display
 * - FFT-based spectral analysis
 * - Frequency spectrum visualization
 * - Record-arm functionality with visual feedback
 * - Integration with PNBTR trainer for live audio data
 */
class SpectralAudioTrack : public juce::Component, public juce::Timer
{
public:
    enum class TrackType {
        JELLIE_INPUT,      // Records input audio (before JELLIE encoding)
        PNBTR_OUTPUT       // Records reconstructed audio (after PNBTR decoding)
    };

    SpectralAudioTrack(TrackType type, const std::string& trackName);
    ~SpectralAudioTrack() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    // Audio data input
    void addAudioData(const float* audioData, int numSamples, double sampleRate);
    void clearAudioData();
    
    // Recording functionality
    void setRecordArmed(bool armed);
    bool isRecordArmed() const { return recordArmed; }
    void startRecording();
    void stopRecording();
    bool isRecording() const { return recording; }
    
    // Spectral analysis
    void performFFT();
    void updateSpectrum();
    
    // Visual configuration
    void setWaveformColour(juce::Colour colour) { waveformColour = colour; }
    void setSpectrumColour(juce::Colour colour) { spectrumColour = colour; }
    
    // Export functionality
    bool exportToWAV(const juce::File& file);
    
    // Integration with PNBTRTrainer
    void setTrainer(class PNBTRTrainer* trainer) { 
        pnbtrTrainer = trainer; 
        
        // Start timer only after trainer is set
        if (pnbtrTrainer) {
            startTimer(33); // 30 FPS updates
        }
    }

private:
    // Track identification
    TrackType trackType;
    std::string trackName;
    
    // Audio data storage
    std::vector<float> audioBuffer;
    std::vector<float> recordingBuffer;
    double currentSampleRate = 48000.0;
    static constexpr int MAX_BUFFER_SIZE = 48000 * 60; // 60 seconds at 48kHz
    
    // Recording state
    std::atomic<bool> recordArmed{false};
    std::atomic<bool> recording{false};
    
    // FFT and spectral analysis
    static constexpr int FFT_SIZE = 2048;
    juce::dsp::FFT fft{11}; // 2^11 = 2048
    juce::dsp::WindowingFunction<float> window{FFT_SIZE, juce::dsp::WindowingFunction<float>::hann};
    std::vector<float> fftBuffer;
    std::vector<float> spectrumData;
    std::mutex spectrumMutex;
    
    // Visual layout
    juce::Rectangle<int> headerArea;
    juce::Rectangle<int> waveformArea;
    juce::Rectangle<int> spectrumArea;
    juce::Rectangle<int> controlsArea;
    
    // Visual styling
    juce::Colour waveformColour{juce::Colours::cyan};
    juce::Colour spectrumColour{juce::Colours::yellow};
    juce::Colour recordArmColour{juce::Colours::red};
    
    // ADDED: Record arm button for user interaction
    std::unique_ptr<juce::TextButton> recordArmButton;
    
    // Integration
    class PNBTRTrainer* pnbtrTrainer = nullptr;
    
    // Drawing methods
    void drawHeader(juce::Graphics& g, juce::Rectangle<int> area);
    void drawWaveform(juce::Graphics& g, juce::Rectangle<int> area);
    void drawSpectrum(juce::Graphics& g, juce::Rectangle<int> area);
    void drawControls(juce::Graphics& g, juce::Rectangle<int> area);
    void drawRecordArmIndicator(juce::Graphics& g, juce::Rectangle<int> area);
    
    // Audio processing helpers
    void processAudioForDisplay(const float* data, int numSamples);
    float mapFrequencyToX(float frequency, float width) const;
    float mapAmplitudeToY(float amplitude, float height) const;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectralAudioTrack)
}; 