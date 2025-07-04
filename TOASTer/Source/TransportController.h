#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <chrono>

//==============================================================================
class TransportButton : public juce::Button
{
public:
    enum ButtonType { Play, Stop, Record };
    
    TransportButton(const juce::String& name, ButtonType type) 
        : juce::Button(name), buttonType(type) {}
    
    void paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
    
private:
    ButtonType buttonType;
};

//==============================================================================
class TransportController : public juce::Component, public juce::Timer
{
public:
    TransportController();
    ~TransportController() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void playButtonClicked();
    void stopButtonClicked();
    void recordButtonClicked();
    
    void updateDisplay();
    void startTransport();
    void stopTransport();
    
    TransportButton playButton;
    TransportButton stopButton;
    TransportButton recordButton;
    juce::Label sessionTimeLabel;
    juce::Label barsBeatsLabel;
    
    bool isPlaying = false;
    bool isRecording = false;
    
    // Transport timing
    std::chrono::high_resolution_clock::time_point transportStartTime;
    std::chrono::microseconds currentPosition{0};
    
    // Musical timing (4/4 time, 120 BPM default)
    double bpm = 120.0;
    int beatsPerBar = 4;
    int subdivisions = 4; // quarter notes
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TransportController)
};