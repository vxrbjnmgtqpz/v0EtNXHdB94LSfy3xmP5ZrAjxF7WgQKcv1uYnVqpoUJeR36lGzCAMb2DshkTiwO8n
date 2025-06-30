#include "MainComponent.h"
#include "TransportController.h"
#include "NetworkConnectionPanel.h"
#include "MIDITestingPanel.h"
#include "PerformanceMonitorPanel.h"
#include "ClockSyncPanel.h"
#include "JSONMIDIIntegrationPanel.h"
#include "MIDIManager.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Initialize MIDI I/O system first
    midiManager = std::make_unique<MIDIManager>();
    
    // Create and add child components
    transportController = std::make_unique<TransportController>();
    addAndMakeVisible(*transportController);
    
    networkPanel = std::make_unique<NetworkConnectionPanel>();
    addAndMakeVisible(*networkPanel);
    
    midiPanel = std::make_unique<MIDITestingPanel>();
    midiPanel->setMIDIManager(midiManager.get()); // Connect MIDI manager
    addAndMakeVisible(*midiPanel);
    
    performancePanel = std::make_unique<PerformanceMonitorPanel>();
    addAndMakeVisible(*performancePanel);
    
    clockSyncPanel = std::make_unique<ClockSyncPanel>();
    addAndMakeVisible(*clockSyncPanel);
    
    jsonmidiPanel = std::make_unique<JSONMIDIIntegrationPanel>();
    addAndMakeVisible(*jsonmidiPanel);
    
    // Set the main window size (increased for new panel)
    setSize(1200, 800);
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
    
    // Divide remaining space into panels
    auto remainingBounds = bounds.reduced(10);
    auto panelHeight = remainingBounds.getHeight() / 2;
    auto panelWidth = remainingBounds.getWidth() / 3;
    
    // Top row - three panels
    auto topRow = remainingBounds.removeFromTop(panelHeight);
    networkPanel->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));
    midiPanel->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));
    jsonmidiPanel->setBounds(topRow.reduced(5)); // JSONMIDI panel gets remaining space
    
    // Bottom row - two panels  
    auto bottomRow = remainingBounds;
    performancePanel->setBounds(bottomRow.removeFromLeft(panelWidth * 1.5).reduced(5));
    clockSyncPanel->setBounds(bottomRow.reduced(5));
}