#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

class SimpleClockSyncPanel : public juce::Component, private juce::Timer
{
public:
    SimpleClockSyncPanel()
    {
        addAndMakeVisible(syncStatusLabel);
        addAndMakeVisible(accuracyLabel);
        addAndMakeVisible(peersLabel);
        addAndMakeVisible(driftLabel);
        addAndMakeVisible(calibrateButton);
        
        syncStatusLabel.setText("Sync: Master", juce::dontSendNotification);
        accuracyLabel.setText("Accuracy: ±1 μs", juce::dontSendNotification);
        peersLabel.setText("Peers: 0", juce::dontSendNotification);
        driftLabel.setText("Drift: 0 ns/s", juce::dontSendNotification);
        
        syncStatusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        accuracyLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
        peersLabel.setColour(juce::Label::textColourId, juce::Colours::white);
        driftLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        
        calibrateButton.setButtonText("Calibrate");
        calibrateButton.onClick = [this] { calibrate(); };
        
        startTimer(2000); // Update every 2 seconds
    }

    ~SimpleClockSyncPanel() override
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
        g.drawText("Clock Synchronization", getLocalBounds().removeFromTop(30), 
                   juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(30); // Title area
        bounds.reduce(10, 5);
        
        auto labelHeight = (bounds.getHeight() - 35) / 4; // Reserve space for button
        
        syncStatusLabel.setBounds(bounds.removeFromTop(labelHeight));
        accuracyLabel.setBounds(bounds.removeFromTop(labelHeight));
        peersLabel.setBounds(bounds.removeFromTop(labelHeight));
        driftLabel.setBounds(bounds.removeFromTop(labelHeight));
        
        bounds.removeFromTop(5);
        calibrateButton.setBounds(bounds.removeFromTop(30));
    }

private:
    void timerCallback() override
    {
        // Simulate clock sync metrics
        static int counter = 0;
        counter++;
        
        auto accuracy = 1 + (counter % 5);
        auto peers = counter % 4;
        auto drift = (counter % 100) - 50;
        
        accuracyLabel.setText("Accuracy: ±" + juce::String(accuracy) + " μs", juce::dontSendNotification);
        peersLabel.setText("Peers: " + juce::String(peers), juce::dontSendNotification);
        driftLabel.setText("Drift: " + juce::String(drift) + " ns/s", juce::dontSendNotification);
        
        if (peers > 0)
        {
            syncStatusLabel.setText("Sync: Networked", juce::dontSendNotification);
            syncStatusLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
        }
        else
        {
            syncStatusLabel.setText("Sync: Master", juce::dontSendNotification);
            syncStatusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        }
    }
    
    void calibrate()
    {
        accuracyLabel.setText("Accuracy: Calibrating...", juce::dontSendNotification);
        accuracyLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
        
        // Reset after delay
        juce::Timer::callAfterDelay(2000, [this]() {
            accuracyLabel.setText("Accuracy: ±1 μs", juce::dontSendNotification);
            accuracyLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
        });
    }

    juce::Label syncStatusLabel;
    juce::Label accuracyLabel;
    juce::Label peersLabel;
    juce::Label driftLabel;
    juce::TextButton calibrateButton;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimpleClockSyncPanel)
};
