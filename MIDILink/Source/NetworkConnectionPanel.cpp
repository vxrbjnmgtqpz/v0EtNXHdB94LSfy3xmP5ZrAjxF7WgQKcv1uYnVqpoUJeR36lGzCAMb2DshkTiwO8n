#include "NetworkConnectionPanel.h"

NetworkConnectionPanel::NetworkConnectionPanel()
    : connectButton("Connect"), disconnectButton("Disconnect")
{
    // Set up title
    titleLabel.setText("Network Connection", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);
    
    // Set up IP address editor
    ipAddressEditor.setText("127.0.0.1");
    ipAddressEditor.setTextToShowWhenEmpty("IP Address", juce::Colours::grey);
    addAndMakeVisible(ipAddressEditor);
    
    // Set up port editor
    portEditor.setText("8080");
    portEditor.setTextToShowWhenEmpty("Port", juce::Colours::grey);
    addAndMakeVisible(portEditor);
    
    // Set up buttons
    connectButton.onClick = [this] { connectButtonClicked(); };
    addAndMakeVisible(connectButton);
    
    disconnectButton.onClick = [this] { disconnectButtonClicked(); };
    disconnectButton.setEnabled(false);
    addAndMakeVisible(disconnectButton);
    
    // Set up status label
    statusLabel.setText("Disconnected", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    addAndMakeVisible(statusLabel);
}

NetworkConnectionPanel::~NetworkConnectionPanel()
{
}

void NetworkConnectionPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void NetworkConnectionPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    auto row = bounds.removeFromTop(25);
    ipAddressEditor.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(5);
    portEditor.setBounds(row.removeFromLeft(60));
    
    bounds.removeFromTop(5);
    
    row = bounds.removeFromTop(25);
    connectButton.setBounds(row.removeFromLeft(80));
    row.removeFromLeft(5);
    disconnectButton.setBounds(row.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    statusLabel.setBounds(bounds.removeFromTop(25));
}

void NetworkConnectionPanel::connectButtonClicked()
{
    isConnected = true;
    connectButton.setEnabled(false);
    disconnectButton.setEnabled(true);
    statusLabel.setText("Connected", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
}

void NetworkConnectionPanel::disconnectButtonClicked()
{
    isConnected = false;
    connectButton.setEnabled(true);
    disconnectButton.setEnabled(false);
    statusLabel.setText("Disconnected", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
}