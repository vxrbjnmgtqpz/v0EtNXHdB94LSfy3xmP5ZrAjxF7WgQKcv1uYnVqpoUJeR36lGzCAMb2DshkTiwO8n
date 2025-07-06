#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>

class BasicMIDIPanel : public juce::Component, private juce::Timer
{
public:
    BasicMIDIPanel()
    {
        addAndMakeVisible(deviceList);
        addAndMakeVisible(testButton);
        addAndMakeVisible(statusLabel);
        
        deviceList.setTextWhenNoChoicesAvailable("No MIDI devices found");
        deviceList.setTextWhenNothingSelected("Select a MIDI device");
        
        testButton.setButtonText("Send Test Note");
        testButton.onClick = [this] { sendTestNote(); };
        
        statusLabel.setText("MIDI Status: Ready", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        
        refreshDevices();
        startTimer(2000); // Refresh devices every 2 seconds
    }

    ~BasicMIDIPanel() override
    {
        stopTimer();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff1a1a1a));
        g.setColour(juce::Colour(0xff3a3a3a));
        g.drawRect(getLocalBounds(), 1);
        
        g.setColour(juce::Colours::white);
        g.setFont(16.0f);
        g.drawText("MIDI Testing Panel", getLocalBounds().removeFromTop(30), 
                   juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(30); // Title area
        bounds.reduce(10, 5);
        
        deviceList.setBounds(bounds.removeFromTop(30));
        bounds.removeFromTop(10);
        testButton.setBounds(bounds.removeFromTop(30));
        bounds.removeFromTop(10);
        statusLabel.setBounds(bounds.removeFromTop(30));
    }

private:
    void timerCallback() override
    {
        refreshDevices();
    }

    void refreshDevices()
    {
        deviceList.clear();
        
        auto midiInputs = juce::MidiInput::getAvailableDevices();
        for (auto& input : midiInputs)
        {
            deviceList.addItem("IN: " + input.name, deviceList.getNumItems() + 1);
        }
        
        auto midiOutputs = juce::MidiOutput::getAvailableDevices();
        for (auto& output : midiOutputs)
        {
            deviceList.addItem("OUT: " + output.name, deviceList.getNumItems() + 1);
        }
        
        if (deviceList.getNumItems() == 0)
        {
            statusLabel.setText("MIDI Status: No devices found", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        }
        else
        {
            statusLabel.setText("MIDI Status: " + juce::String(deviceList.getNumItems()) + " devices found", 
                              juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        }
    }

    void sendTestNote()
    {
        // Send a simple test MIDI note
        auto midiOutputs = juce::MidiOutput::getAvailableDevices();
        if (!midiOutputs.isEmpty())
        {
            auto midiOut = juce::MidiOutput::openDevice(midiOutputs[0].identifier);
            if (midiOut)
            {
                // Send Note On C4 (60) with velocity 100
                juce::MidiMessage noteOn = juce::MidiMessage::noteOn(1, 60, (juce::uint8)100);
                midiOut->sendMessageNow(noteOn);
                
                // Send Note Off after 500ms - use shared_ptr to avoid copy issues
                std::shared_ptr<juce::MidiOutput> sharedMidiOut = std::move(midiOut);
                juce::Timer::callAfterDelay(500, [sharedMidiOut]() {
                    juce::MidiMessage noteOff = juce::MidiMessage::noteOff(1, 60);
                    sharedMidiOut->sendMessageNow(noteOff);
                });
                
                statusLabel.setText("MIDI Status: Test note sent", juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
            }
        }
        else
        {
            statusLabel.setText("MIDI Status: No output devices available", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        }
    }

    juce::ComboBox deviceList;
    juce::TextButton testButton;
    juce::Label statusLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BasicMIDIPanel)
};
