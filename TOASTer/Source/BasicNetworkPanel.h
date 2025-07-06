#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "ConnectionDiscovery.h"

class BasicNetworkPanel : public juce::Component, private juce::Timer
{
public:
    BasicNetworkPanel()
    {
        addAndMakeVisible(startButton);
        addAndMakeVisible(stopButton);
        addAndMakeVisible(statusLabel);
        addAndMakeVisible(peersLabel);
        
        startButton.setButtonText("Start Discovery");
        startButton.onClick = [this] { startDiscovery(); };
        
        stopButton.setButtonText("Stop Discovery");
        stopButton.onClick = [this] { stopDiscovery(); };
        stopButton.setEnabled(false);
        
        statusLabel.setText("Network Status: Stopped", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        
        peersLabel.setText("Peers: 0", juce::dontSendNotification);
        peersLabel.setColour(juce::Label::textColourId, juce::Colours::white);
        
        startTimer(1000); // Update every second
    }

    ~BasicNetworkPanel() override
    {
        stopTimer();
        discovery.reset();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff1a1a1a));
        g.setColour(juce::Colour(0xff3a3a3a));
        g.drawRect(getLocalBounds(), 1);
        
        g.setColour(juce::Colours::white);
        g.setFont(16.0f);
        g.drawText("Network Discovery Panel", getLocalBounds().removeFromTop(30), 
                   juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(30); // Title area
        bounds.reduce(10, 5);
        
        auto buttonRow = bounds.removeFromTop(30);
        startButton.setBounds(buttonRow.removeFromLeft(buttonRow.getWidth() / 2 - 5));
        buttonRow.removeFromLeft(10);
        stopButton.setBounds(buttonRow);
        
        bounds.removeFromTop(10);
        statusLabel.setBounds(bounds.removeFromTop(30));
        bounds.removeFromTop(10);
        peersLabel.setBounds(bounds.removeFromTop(30));
    }

private:
    void timerCallback() override
    {
        if (discovery && discovery->isRunning())
        {
            auto peers = discovery->getDiscoveredPeers();
            peersLabel.setText("Peers: " + juce::String(peers.size()), juce::dontSendNotification);
            
            if (!peers.empty())
            {
                peersLabel.setColour(juce::Label::textColourId, juce::Colours::green);
            }
        }
    }

    void startDiscovery()
    {
        discovery = std::make_unique<ConnectionDiscovery>();
        discovery->startDiscovery();
        
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        
        statusLabel.setText("Network Status: Discovering...", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    }

    void stopDiscovery()
    {
        if (discovery)
        {
            discovery->stopDiscovery();
            discovery.reset();
        }
        
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        
        statusLabel.setText("Network Status: Stopped", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        
        peersLabel.setText("Peers: 0", juce::dontSendNotification);
        peersLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    }

    juce::TextButton startButton;
    juce::TextButton stopButton;
    juce::Label statusLabel;
    juce::Label peersLabel;
    
    std::unique_ptr<ConnectionDiscovery> discovery;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BasicNetworkPanel)
};
