#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

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
class TransportController : public juce::Component
{
public:
    TransportController();
    ~TransportController() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void playButtonClicked();
    void stopButtonClicked();
    void recordButtonClicked();
    
    void updateDisplay();
    
    TransportButton playButton;
    TransportButton stopButton;
    TransportButton recordButton;
    juce::Label sessionTimeLabel;
    juce::Label barsBeatsLabel;
    
    bool isPlaying = false;
    bool isRecording = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TransportController)
};