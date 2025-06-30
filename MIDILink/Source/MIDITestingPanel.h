#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
class MIDITestingPanel : public juce::Component
{
public:
    MIDITestingPanel();
    ~MIDITestingPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void sendTestNoteClicked();
    void clearLogClicked();
    
    juce::Label titleLabel;
    juce::TextButton sendTestNoteButton;
    juce::TextButton clearLogButton;
    juce::TextEditor logDisplay;
    juce::ComboBox midiChannelSelector;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MIDITestingPanel)
};