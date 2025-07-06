#include "MainComponent.h"
#include "BasicMIDIPanel.h"
#include "BasicNetworkPanel.h"
#include "BasicTransportPanel.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Create simplified panels without heavy framework dependencies
    transportPanel = new BasicTransportPanel();
    midiPanel = new BasicMIDIPanel();
    networkPanel = new BasicNetworkPanel();
    
    addAndMakeVisible(transportPanel);
    addAndMakeVisible(midiPanel);
    addAndMakeVisible(networkPanel);
    
    setSize(800, 600);
    
    // Start UI update timer
    startTimer(50); // 20 FPS for UI updates
}

MainComponent::~MainComponent()
{
    stopTimer();
    delete transportPanel;
    delete midiPanel;
    delete networkPanel;
}

void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff2a2a2a));
    
    g.setColour(juce::Colour(0xff4a4a4a));
    g.drawRect(getLocalBounds(), 2);
    
    g.setColour(juce::Colours::white);
    g.setFont(24.0f);
    g.drawText("TOASTer - MIDI/Audio Testing Tool", 
               getLocalBounds().removeFromTop(50), 
               juce::Justification::centred, true);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromTop(50); // Title area
    
    auto panelHeight = bounds.getHeight() / 3;
    
    transportPanel->setBounds(bounds.removeFromTop(panelHeight).reduced(5));
    midiPanel->setBounds(bounds.removeFromTop(panelHeight).reduced(5));
    networkPanel->setBounds(bounds.reduced(5));
}

void MainComponent::timerCallback()
{
    // Simple periodic updates
    repaint();
}
