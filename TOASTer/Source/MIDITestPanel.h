#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>

class MIDITestPanel : public juce::Component, private juce::Timer, private juce::MidiInputCallback
{
public:
    MIDITestPanel()
    {
        // Device selection
        addAndMakeVisible(deviceComboBox);
        addAndMakeVisible(deviceLabel);
        addAndMakeVisible(refreshButton);
        
        // Test controls
        addAndMakeVisible(testNoteButton);
        addAndMakeVisible(monitorButton);
        
        // Status display
        addAndMakeVisible(statusTextEditor);
        
        // Setup components
        deviceLabel.setText("MIDI Device:", juce::dontSendNotification);
        deviceLabel.setColour(juce::Label::textColourId, juce::Colours::white);
        
        deviceComboBox.setTextWhenNoChoicesAvailable("No MIDI devices");
        deviceComboBox.setTextWhenNothingSelected("Select MIDI device");
        
        refreshButton.setButtonText("Refresh");
        refreshButton.onClick = [this] { refreshMIDIDevices(); };
        
        testNoteButton.setButtonText("Send Test Note");
        testNoteButton.onClick = [this] { sendTestNote(); };
        
        monitorButton.setButtonText("Start Monitor");
        monitorButton.onClick = [this] { toggleMonitoring(); };
        
        statusTextEditor.setMultiLine(true);
        statusTextEditor.setReadOnly(true);
        statusTextEditor.setCaretVisible(false);
        statusTextEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff1a1a1a));
        statusTextEditor.setColour(juce::TextEditor::textColourId, juce::Colours::lightgreen);
        statusTextEditor.setFont(juce::Font("Courier New", 12.0f, juce::Font::plain));
        
        // JDAT placeholders
        addAndMakeVisible(jdatTestButton);
        addAndMakeVisible(jdatStatusLabel);
        
        jdatTestButton.setButtonText("Test JDAT Audio");
        jdatTestButton.onClick = [this] { 
            jdatStatusLabel.setText("JDAT: Test audio streaming via JAM Framework v2", juce::dontSendNotification);
        };
        jdatStatusLabel.setText("JDAT: Ready for audio streaming tests", juce::dontSendNotification);
        jdatStatusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        
        // JVID placeholders
        addAndMakeVisible(jvidTestButton);
        addAndMakeVisible(jvidStatusLabel);
        
        jvidTestButton.setButtonText("Test JVID Video");
        jvidTestButton.onClick = [this] { 
            jvidStatusLabel.setText("JVID: Test video streaming via JAM Framework v2", juce::dontSendNotification);
        };
        jvidStatusLabel.setText("JVID: Ready for video streaming tests", juce::dontSendNotification);
        jvidStatusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);

        refreshMIDIDevices();
        startTimer(100); // Update status every 100ms
    }

    ~MIDITestPanel() override
    {
        stopTimer();
        if (midiInput)
            midiInput->stop();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff1a1a1a));
        g.setColour(juce::Colour(0xff3a3a3a));
        g.drawRect(getLocalBounds(), 1);
        
        g.setColour(juce::Colours::white);
        g.setFont(juce::Font(14.0f, juce::Font::bold));
        g.drawText("Media Testing Panel", getLocalBounds().removeFromTop(25), 
                   juce::Justification::centred, true);
        
        // Draw sections as COLUMNS (each section gets its own vertical column)
        auto bounds = getLocalBounds();
        bounds.removeFromTop(25);
        bounds.reduce(5, 2);
        
        auto sectionWidth = bounds.getWidth() / 3;
        
        // MIDI section - left COLUMN
        auto midiSection = bounds.removeFromLeft(sectionWidth);
        g.setColour(juce::Colour(0xff2a4a2a));
        g.fillRect(midiSection.reduced(2));
        g.setColour(juce::Colours::lightgreen);
        g.setFont(juce::Font(12.0f, juce::Font::bold));
        g.drawText("🎵 MIDI Testing", midiSection.removeFromTop(20), juce::Justification::centred, true);
        
        // JDAT section - middle COLUMN
        auto jdatSection = bounds.removeFromLeft(sectionWidth);
        g.setColour(juce::Colour(0xff4a2a2a));
        g.fillRect(jdatSection.reduced(2));
        g.setColour(juce::Colours::lightcoral);
        g.setFont(juce::Font(12.0f, juce::Font::bold));
        g.drawText("🎧 JDAT Audio Testing", jdatSection.removeFromTop(20), juce::Justification::centred, true);
        
        // JVID section - right COLUMN
        auto jvidSection = bounds;
        g.setColour(juce::Colour(0xff2a2a4a));
        g.fillRect(jvidSection.reduced(2));
        g.setColour(juce::Colours::lightblue);
        g.setFont(juce::Font(12.0f, juce::Font::bold));
        g.drawText("🎬 JVID Video Testing", jvidSection.removeFromTop(20), juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(25); // Title area
        bounds.reduce(5, 2);
        
        auto sectionWidth = bounds.getWidth() / 3;
        
        // MIDI section (left column)
        auto midiSection = bounds.removeFromLeft(sectionWidth).reduced(4);
        midiSection.removeFromTop(20); // Section title space
        
        // MIDI controls in compact layout
        auto midiControls = midiSection.removeFromTop(25);
        deviceLabel.setBounds(midiControls.removeFromLeft(50));
        deviceComboBox.setBounds(midiControls.removeFromLeft(120));
        midiControls.removeFromLeft(5);
        refreshButton.setBounds(midiControls.removeFromLeft(60));
        
        midiSection.removeFromTop(3);
        
        // MIDI buttons
        auto midiButtons = midiSection.removeFromTop(25);
        testNoteButton.setBounds(midiButtons.removeFromLeft(100));
        midiButtons.removeFromLeft(5);
        monitorButton.setBounds(midiButtons.removeFromLeft(100));
        
        midiSection.removeFromTop(3);
        
        // MIDI status (remaining space in MIDI section)
        statusTextEditor.setBounds(midiSection);
        
        // JDAT section (middle column) - audio streaming test
        auto jdatSection = bounds.removeFromLeft(sectionWidth).reduced(4);
        jdatSection.removeFromTop(20); // Section title space
        
        auto jdatControls = jdatSection.removeFromTop(25);
        jdatTestButton.setBounds(jdatControls.removeFromLeft(120));
        jdatSection.removeFromTop(3);
        jdatStatusLabel.setBounds(jdatSection);
        
        // JVID section (right column) - video streaming test
        auto jvidSection = bounds.reduced(4);
        jvidSection.removeFromTop(20); // Section title space
        
        auto jvidControls = jvidSection.removeFromTop(25);
        jvidTestButton.setBounds(jvidControls.removeFromLeft(120));
        jvidSection.removeFromTop(3);
        jvidStatusLabel.setBounds(jvidSection);
    }

private:
    void refreshMIDIDevices()
    {
        deviceComboBox.clear();
        
        auto midiInputs = juce::MidiInput::getAvailableDevices();
        auto midiOutputs = juce::MidiOutput::getAvailableDevices();
        
        for (int i = 0; i < midiInputs.size(); ++i)
        {
            deviceComboBox.addItem("IN: " + midiInputs[i].name, i + 1);
        }
        
        for (int i = 0; i < midiOutputs.size(); ++i)
        {
            deviceComboBox.addItem("OUT: " + midiOutputs[i].name, midiInputs.size() + i + 1);
        }
        
        addStatusMessage("Found " + juce::String(midiInputs.size()) + " input(s), " + 
                        juce::String(midiOutputs.size()) + " output(s)");
    }

    void sendTestNote()
    {
        auto midiOutputs = juce::MidiOutput::getAvailableDevices();
        if (!midiOutputs.isEmpty())
        {
            auto midiOut = juce::MidiOutput::openDevice(midiOutputs[0].identifier);
            if (midiOut)
            {
                // Send Note On C4
                auto noteOnMsg = juce::MidiMessage::noteOn(1, 60, (juce::uint8)100);
                midiOut->sendMessageNow(noteOnMsg);
                
                addStatusMessage("Sent: Note On C4 (60) velocity 100");
                
                // Store the device identifier for the delayed note off
                auto deviceId = midiOutputs[0].identifier;
                
                // Send Note Off after 500ms
                juce::Timer::callAfterDelay(500, [this, deviceId]() {
                    auto midiOutDelayed = juce::MidiOutput::openDevice(deviceId);
                    if (midiOutDelayed)
                    {
                        auto noteOffMsg = juce::MidiMessage::noteOff(1, 60);
                        midiOutDelayed->sendMessageNow(noteOffMsg);
                        addStatusMessage("Sent: Note Off C4 (60)");
                    }
                });
            }
        }
        else
        {
            addStatusMessage("ERROR: No MIDI output devices available");
        }
    }

    void toggleMonitoring()
    {
        if (isMonitoring)
        {
            if (midiInput)
            {
                midiInput->stop();
                midiInput.reset();
            }
            monitorButton.setButtonText("Start Monitor");
            isMonitoring = false;
            addStatusMessage("MIDI monitoring stopped");
        }
        else
        {
            auto midiInputs = juce::MidiInput::getAvailableDevices();
            if (!midiInputs.isEmpty())
            {
                midiInput = juce::MidiInput::openDevice(midiInputs[0].identifier, this);
                if (midiInput)
                {
                    midiInput->start();
                    monitorButton.setButtonText("Stop Monitor");
                    isMonitoring = true;
                    addStatusMessage("MIDI monitoring started on: " + midiInputs[0].name);
                }
            }
            else
            {
                addStatusMessage("ERROR: No MIDI input devices available");
            }
        }
    }

    void handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message) override
    {
        juce::String msgText = "RX: ";
        
        if (message.isNoteOn())
            msgText += "Note On " + juce::String(message.getNoteNumber()) + " vel " + juce::String(message.getVelocity());
        else if (message.isNoteOff())
            msgText += "Note Off " + juce::String(message.getNoteNumber());
        else if (message.isController())
            msgText += "CC " + juce::String(message.getControllerNumber()) + " val " + juce::String(message.getControllerValue());
        else
            msgText += "Other MIDI msg";
            
        addStatusMessage(msgText);
    }

    void addStatusMessage(const juce::String& message)
    {
        auto currentTime = juce::Time::getCurrentTime().toString(true, true, true, true);
        auto newText = statusTextEditor.getText() + "[" + currentTime + "] " + message + "\n";
        
        // Keep only last 50 lines
        auto lines = juce::StringArray::fromLines(newText);
        if (lines.size() > 50)
            lines.removeRange(0, lines.size() - 50);
            
        statusTextEditor.setText(lines.joinIntoString("\n"));
        statusTextEditor.moveCaretToEnd();
    }

    void timerCallback() override
    {
        // Update connection status periodically
        if (messageCounter++ % 10 == 0) // Every second
        {
            auto midiInputs = juce::MidiInput::getAvailableDevices();
            auto midiOutputs = juce::MidiOutput::getAvailableDevices();
            // Could add periodic device check here
        }
    }

    juce::ComboBox deviceComboBox;
    juce::Label deviceLabel;
    juce::TextButton refreshButton;
    juce::TextButton testNoteButton;
    juce::TextButton monitorButton;
    juce::TextEditor statusTextEditor;
    
    std::unique_ptr<juce::MidiInput> midiInput;
    bool isMonitoring = false;
    int messageCounter = 0;

    // JDAT testing components
    juce::TextButton jdatTestButton;
    juce::Label jdatStatusLabel;
    
    // JVID testing components
    juce::TextButton jvidTestButton;
    juce::Label jvidStatusLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MIDITestPanel)
};
