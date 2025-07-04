#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "JMIDMessage.h"
#include "JMIDParser.h"

// Forward declaration
class MIDIManager;

//==============================================================================
class MIDITestingPanel : public juce::Component,
                        public juce::ChangeListener,
                        public juce::Timer
{
public:
    MIDITestingPanel();
    ~MIDITestingPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    
    // Set reference to MIDI manager
    void setMIDIManager(MIDIManager* manager);
    
    // ChangeListener interface
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    
    // Timer interface
    void timerCallback() override;

private:
    void sendTestNoteClicked();
    void clearLogClicked();
    void logMessage(const juce::String& message);
    void refreshDeviceLists();
    void inputDeviceChanged();
    void outputDeviceChanged();
    void onMIDIMessageReceived(std::shared_ptr<JMID::MIDIMessage> message);
    
    juce::Label titleLabel;
    
    // Device selection
    juce::Label inputDeviceLabel;
    juce::ComboBox inputDeviceSelector;
    juce::Label outputDeviceLabel;
    juce::ComboBox outputDeviceSelector;
    juce::TextButton refreshDevicesButton;
    
    // Test message controls
    juce::TextButton sendTestNoteButton;
    juce::TextButton clearLogButton;
    juce::Label logLabel;
    juce::TextEditor logDisplay;
    juce::ComboBox midiChannelSelector;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;
    juce::Label channelLabel;
    juce::Label noteLabel;
    juce::Label velocityLabel;
    
    // JMID Framework integration
    std::unique_ptr<JMID::BassoonParser> parser;
    
    // MIDI I/O reference
    MIDIManager* midiManager = nullptr;
    
    // Helper function for emoji-compatible font setup
    static juce::Font getEmojiCompatibleFont(float size = 12.0f);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MIDITestingPanel)
};