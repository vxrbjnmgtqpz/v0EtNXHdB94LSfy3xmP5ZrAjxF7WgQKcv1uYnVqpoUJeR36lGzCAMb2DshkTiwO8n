#include "MainComponent.h"
#include "ProfessionalTransportController.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Create ONLY the professional transport controller for now
    transportPanel = new ProfessionalTransportController();
    
    addAndMakeVisible(transportPanel);
    
    setSize(800, 400); // Smaller size to focus on transport
    
    // Start UI update timer
    startTimer(50); // 20 FPS for UI updates
}

MainComponent::~MainComponent()
{
    stopTimer();
    delete transportPanel;
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
    
    // Give all remaining space to the transport controller
    transportPanel->setBounds(bounds.reduced(10));
}

void MainComponent::timerCallback()
{
    // Simple periodic updates
    repaint();
}
