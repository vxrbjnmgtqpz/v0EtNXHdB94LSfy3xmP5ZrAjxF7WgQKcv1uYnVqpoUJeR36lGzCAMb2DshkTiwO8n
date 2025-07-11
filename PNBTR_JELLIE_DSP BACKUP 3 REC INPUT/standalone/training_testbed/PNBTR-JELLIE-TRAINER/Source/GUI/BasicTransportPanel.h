#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>

class BasicTransportPanel : public juce::Component, private juce::Timer
{
public:
    BasicTransportPanel();
    ~BasicTransportPanel() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

private:
    void startTransport();
    void stopTransport();
    void resetTransport();
    void updateDisplay();

    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::TextButton resetButton;
    juce::Label statusLabel;
    juce::Label tempoLabel;
    juce::Label positionLabel;
    juce::Slider tempoSlider;
    
    std::atomic<bool> isPlaying{false};
    std::atomic<double> currentTempo{120.0};
    std::atomic<int64_t> currentSample{0};
    std::chrono::steady_clock::time_point startTime;
    
    static constexpr double sampleRate = 44100.0;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BasicTransportPanel)
};
