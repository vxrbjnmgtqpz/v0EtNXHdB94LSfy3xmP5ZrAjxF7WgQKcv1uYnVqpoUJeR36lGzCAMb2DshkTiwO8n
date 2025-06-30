#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "JSONMIDIMessage.h"
#include "JSONMIDIParser.h"

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
    void logMessage(const juce::String& message);
    
    juce::Label titleLabel;
    juce::TextButton sendTestNoteButton;
    juce::TextButton clearLogButton;
    juce::TextEditor logDisplay;
    juce::ComboBox midiChannelSelector;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;
    
    // JSONMIDI Framework integration
    std::unique_ptr<JSONMIDI::BassoonParser> parser;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MIDITestingPanel)
};