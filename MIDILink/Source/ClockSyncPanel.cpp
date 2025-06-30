#include "ClockSyncPanel.h"

ClockSyncPanel::ClockSyncPanel()
    : syncGroup("Clock Sync"), settingsGroup("Sync Settings"),
      masterModeButton("Master Mode"), syncRateLabel("Sync Rate:"),
      syncRateSlider(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight)
{
    // Set up sync group
    syncGroup.setText("Clock Sync Status");
    addAndMakeVisible(syncGroup);
    
    // Set up master mode button
    masterModeButton.setToggleState(false, juce::dontSendNotification);
    masterModeButton.onClick = [this] { enableSyncClicked(); };
    addAndMakeVisible(masterModeButton);
    
    // Set up status labels
    syncStatusLabel.setText("Status: Disabled", juce::dontSendNotification);
    addAndMakeVisible(syncStatusLabel);
    
    networkOffsetLabel.setText("Network Offset: 0ms", juce::dontSendNotification);
    addAndMakeVisible(networkOffsetLabel);
    
    qualityLabel.setText("Sync Quality: Unknown", juce::dontSendNotification);
    addAndMakeVisible(qualityLabel);
    
    // Set up settings group
    settingsGroup.setText("Settings");
    addAndMakeVisible(settingsGroup);
    
    // Set up sync rate controls
    addAndMakeVisible(syncRateLabel);
    
    syncRateSlider.setRange(1.0, 100.0, 1.0);
    syncRateSlider.setValue(24.0); // Default to 24 Hz
    syncRateSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    syncRateSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(syncRateSlider);
}

ClockSyncPanel::~ClockSyncPanel()
{
}

void ClockSyncPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void ClockSyncPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    // Sync group takes top half
    auto syncBounds = bounds.removeFromTop(bounds.getHeight() / 2);
    syncGroup.setBounds(syncBounds);
    
    auto syncContentBounds = syncBounds.reduced(15, 25);
    masterModeButton.setBounds(syncContentBounds.removeFromTop(25));
    syncContentBounds.removeFromTop(5);
    syncStatusLabel.setBounds(syncContentBounds.removeFromTop(20));
    networkOffsetLabel.setBounds(syncContentBounds.removeFromTop(20));
    qualityLabel.setBounds(syncContentBounds.removeFromTop(20));
    
    bounds.removeFromTop(10); // spacing
    
    // Settings group takes bottom half
    settingsGroup.setBounds(bounds);
    auto settingsContentBounds = bounds.reduced(15, 25);
    
    auto row = settingsContentBounds.removeFromTop(25);
    syncRateLabel.setBounds(row.removeFromLeft(80));
    syncRateSlider.setBounds(row);
}

void ClockSyncPanel::enableSyncClicked()
{
    syncEnabled = masterModeButton.getToggleState();
    
    if (syncEnabled)
    {
        syncStatusLabel.setText("Status: Master Mode", juce::dontSendNotification);
        syncStatusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        networkOffsetLabel.setText("Network Offset: N/A (Master)", juce::dontSendNotification);
        qualityLabel.setText("Sync Quality: Excellent", juce::dontSendNotification);
    }
    else
    {
        syncStatusLabel.setText("Status: Disabled", juce::dontSendNotification);
        syncStatusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        networkOffsetLabel.setText("Network Offset: 0ms", juce::dontSendNotification);
        qualityLabel.setText("Sync Quality: Unknown", juce::dontSendNotification);
    }
}

void ClockSyncPanel::calibrateClicked()
{
    // Placeholder for calibration functionality
}