#include "MainComponent.h"
#include "TransportController.h"
#include "NetworkConnectionPanel.h"
#include "MIDITestingPanel.h"
#include "PerformanceMonitorPanel.h"
#include "ClockSyncPanel.h"
#include "JMIDIntegrationPanel.h"
#include "MIDIManager.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Initialize MIDI I/O system first
    midiManager = std::make_unique<MIDIManager>();
    
    // Create and add child components
    transportController = std::make_unique<TransportController>();
    addAndMakeVisible(transportController.get());
    
    networkPanel = std::make_unique<NetworkConnectionPanel>();
    addAndMakeVisible(networkPanel.get());
    
    midiPanel = std::make_unique<MIDITestingPanel>();
    midiPanel->setMIDIManager(midiManager.get()); // Connect MIDI manager
    addAndMakeVisible(midiPanel.get());
    
    performancePanel = std::make_unique<PerformanceMonitorPanel>();
    addAndMakeVisible(performancePanel.get());
    
    clockSyncPanel = std::make_unique<ClockSyncPanel>();
    addAndMakeVisible(clockSyncPanel.get());
    
    jmidPanel = std::make_unique<JMIDIntegrationPanel>();
    addAndMakeVisible(jmidPanel.get());
    
    // Connect TransportController to NetworkPanel for automatic sync
    transportController->setNetworkPanel(networkPanel.get());
    
    // Start timer to coordinate state updates between panels
    startTimer(250); // Update 4 times per second
    
    // Set the main window size (increased for new panel)
    setSize(1200, 800);
}

MainComponent::~MainComponent()
{
    stopTimer();
}

void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::darkgrey);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Transport bar at the top (fixed height)
    transportController.get()->setBounds(bounds.removeFromTop(50));
    
    // Divide remaining space into panels
    auto remainingBounds = bounds.reduced(10);
    auto panelHeight = remainingBounds.getHeight() / 2;
    auto panelWidth = remainingBounds.getWidth() / 3;
    
    // Top row - three panels
    auto topRow = remainingBounds.removeFromTop(panelHeight);
    networkPanel.get()->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));
    midiPanel.get()->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));
    jmidPanel.get()->setBounds(topRow.reduced(5)); // JMID panel gets remaining space
    
    // Bottom row - two panels  
    auto bottomRow = remainingBounds;
    performancePanel.get()->setBounds(bottomRow.removeFromLeft(panelWidth * 1.5).reduced(5));
    clockSyncPanel.get()->setBounds(bottomRow.reduced(5));
}

void MainComponent::timerCallback()
{
    // Update shared state timestamp
    appState.lastUpdate = std::chrono::high_resolution_clock::now();
    
    // Push current state to all panels
    performancePanel.get()->setConnectionState(appState.isNetworkConnected, appState.activeConnections);
    performancePanel.get()->setNetworkLatency(appState.networkLatency);
    performancePanel.get()->setClockAccuracy(appState.clockAccuracy);
    performancePanel->setMessageProcessingRate(appState.messageProcessingRate);
    performancePanel->setMIDIThroughput(appState.midiThroughput);
}

void MainComponent::updateNetworkState(bool connected, int connections, const std::string& ip, int port)
{
    appState.isNetworkConnected = connected;
    appState.activeConnections = connections;
    appState.connectedIP = ip;
    appState.connectedPort = port;
    appState.lastUpdate = std::chrono::high_resolution_clock::now();
}

void MainComponent::updateNetworkLatency(double latencyMs)
{
    appState.networkLatency = latencyMs;
    appState.lastUpdate = std::chrono::high_resolution_clock::now();
}

void MainComponent::updateClockSync(bool enabled, double accuracy, double offset, uint64_t rtt)
{
    appState.isClockSyncEnabled = enabled;
    appState.clockAccuracy = accuracy;
    appState.clockOffset = offset;
    appState.roundTripTime = rtt;
    appState.lastUpdate = std::chrono::high_resolution_clock::now();
}

void MainComponent::updatePerformanceMetrics(int msgRate, int midiRate)
{
    appState.messageProcessingRate = msgRate;
    appState.midiThroughput = midiRate;
    appState.lastUpdate = std::chrono::high_resolution_clock::now();
}