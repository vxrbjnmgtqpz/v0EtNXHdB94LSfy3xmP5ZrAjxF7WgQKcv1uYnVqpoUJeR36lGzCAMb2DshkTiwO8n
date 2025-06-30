#include "MainComponent.h"
#include "TransportController.h"
#include "NetworkConnectionPanel.h"
#include "MIDITestingPanel.h"
#include "PerformanceMonitorPanel.h"
#include "ClockSyncPanel.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Create and add child components
    transportController = std::make_unique<TransportController>();
    addAndMakeVisible(*transportController);
    
    networkPanel = std::make_unique<NetworkConnectionPanel>();
    addAndMakeVisible(*networkPanel);
    
    midiPanel = std::make_unique<MIDITestingPanel>();
    addAndMakeVisible(*midiPanel);
    
    performancePanel = std::make_unique<PerformanceMonitorPanel>();
    addAndMakeVisible(*performancePanel);
    
    clockSyncPanel = std::make_unique<ClockSyncPanel>();
    addAndMakeVisible(*clockSyncPanel);
    
    // Set the main window size
    setSize(800, 600);
}

MainComponent::~MainComponent()
{
}

void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::darkgrey);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Transport bar at the top (fixed height)
    transportController->setBounds(bounds.removeFromTop(50));
    
    // Divide remaining space into quarters for the panels
    auto remainingBounds = bounds.reduced(10);
    auto panelHeight = remainingBounds.getHeight() / 2;
    auto panelWidth = remainingBounds.getWidth() / 2;
    
    // Top row
    auto topRow = remainingBounds.removeFromTop(panelHeight);
    networkPanel->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));
    midiPanel->setBounds(topRow.reduced(5));
    
    // Bottom row  
    auto bottomRow = remainingBounds;
    performancePanel->setBounds(bottomRow.removeFromLeft(panelWidth).reduced(5));
    clockSyncPanel->setBounds(bottomRow.reduced(5));
}