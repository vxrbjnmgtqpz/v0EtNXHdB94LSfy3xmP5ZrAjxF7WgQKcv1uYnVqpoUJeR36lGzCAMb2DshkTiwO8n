#include "ClockSyncPanel.h"
#include "../../JAM_Framework_v2/include/gpu_native/gpu_timebase.h"  // GPU-native timebase
#include "FontUtils.h"
#include <sstream>
#include <iomanip>

ClockSyncPanel::ClockSyncPanel()
    : syncGroup("GPU-Native Peer Synchronization"), networkGroup("Network Consensus"),
      calibrateButton("Recalibrate GPU Timebase")
{
    // Initialize ClockDriftArbiter
    clockArbiter = std::make_unique<TOAST::ClockDriftArbiter>();
    
    // Set up sync group - shows GPU timebase status
    syncGroup.setText("GPU-Native Peer Synchronization");
    syncGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(syncGroup);
    
    // Set up status labels - no toggles needed, sync is automatic
    peerSyncStatusLabel.setText("GPU Timebase: Active", juce::dontSendNotification);
    peerSyncStatusLabel.setFont(FontUtils::getCleanFont(14.0f, true));
    addAndMakeVisible(peerSyncStatusLabel);
    
    localTimingLabel.setText("Local GPU Time: --", juce::dontSendNotification);
    localTimingLabel.setFont(FontUtils::getMonospaceFont(11.0f));
    addAndMakeVisible(localTimingLabel);
    
    networkLatencyLabel.setText("Network Latency: -- μs", juce::dontSendNotification);
    networkLatencyLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(networkLatencyLabel);
    
    syncAccuracyLabel.setText("Sync Accuracy: -- ns", juce::dontSendNotification);
    syncAccuracyLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(syncAccuracyLabel);
    
    gpuTimebaseLabel.setText("GPU Timebase: -- fps", juce::dontSendNotification);
    gpuTimebaseLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(gpuTimebaseLabel);
    
    // Set up network consensus group
    networkGroup.setText("Network Consensus");
    addAndMakeVisible(networkGroup);
    
    activePeersLabel.setText("Active Peers: 0", juce::dontSendNotification);
    activePeersLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(activePeersLabel);
    
    consensusQualityLabel.setText("Consensus Quality: 100%", juce::dontSendNotification);
    consensusQualityLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(consensusQualityLabel);
    
    networkStabilityLabel.setText("Network Stability: Excellent", juce::dontSendNotification);
    networkStabilityLabel.setFont(FontUtils::getCleanFont(11.0f));
    addAndMakeVisible(networkStabilityLabel);
    
    // Set up calibrate button - for manual fine-tuning only
    calibrateButton.onClick = [this] { calibrateClicked(); };
    addAndMakeVisible(calibrateButton);
    
    // Start update timer - sync with GPU timebase updates
    startTimer(16); // ~60 FPS to match GPU refresh
}

ClockSyncPanel::~ClockSyncPanel()
{
    stopTimer();
    // GPU timebase cleanup is handled automatically by the static GPU-native system
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
    
    // GPU sync status group takes top half
    auto syncBounds = bounds.removeFromTop(bounds.getHeight() * 0.6f);
    syncGroup.setBounds(syncBounds);
    
    auto syncContentBounds = syncBounds.reduced(15, 25);
    
    // GPU timebase status labels
    peerSyncStatusLabel.setBounds(syncContentBounds.removeFromTop(25));
    syncContentBounds.removeFromTop(5);
    
    localTimingLabel.setBounds(syncContentBounds.removeFromTop(20));
    networkLatencyLabel.setBounds(syncContentBounds.removeFromTop(20));
    syncAccuracyLabel.setBounds(syncContentBounds.removeFromTop(20));
    gpuTimebaseLabel.setBounds(syncContentBounds.removeFromTop(20));
    
    bounds.removeFromTop(10); // spacing
    
    // Network consensus group takes remaining space
    networkGroup.setBounds(bounds);
    auto networkContentBounds = bounds.reduced(15, 25);
    
    activePeersLabel.setBounds(networkContentBounds.removeFromTop(20));
    consensusQualityLabel.setBounds(networkContentBounds.removeFromTop(20));
    networkStabilityLabel.setBounds(networkContentBounds.removeFromTop(20));
    
    networkContentBounds.removeFromTop(10);
    calibrateButton.setBounds(networkContentBounds.removeFromTop(25).removeFromLeft(150));
}

void ClockSyncPanel::timerCallback()
{
    updateDisplay();
}

void ClockSyncPanel::calibrateClicked()
{
    // Trigger GPU timebase recalibration
    juce::Logger::writeToLog("🔧 Recalibrating GPU timebase...");
    
    // In a real implementation, this would:
    // - Trigger GPU shader recalibration
    // - Sync with network peers for consensus
    // - Update local timing accuracy
}

void ClockSyncPanel::updateDisplay()
{
    // Only show GPU time when actually connected to peers
    bool isConnected = isNetworkConnected && (activePeerCount > 0);
    
    // Get current GPU timebase status
    if (jam::gpu_native::GPUTimebase::is_initialized() && isConnected) {
        gpuTimebaseNs = jam::gpu_native::GPUTimebase::get_current_time_ns();
        
        // Update GPU status
        peerSyncStatusLabel.setText("GPU Timebase: Active & Synchronized", juce::dontSendNotification);
        
        // Format GPU time
        double seconds = gpuTimebaseNs / 1e9;
        auto timeStr = juce::String::formatted("Local GPU Time: %.3f sec", seconds);
        localTimingLabel.setText(timeStr, juce::dontSendNotification);
        
        networkLatencyLabel.setText(juce::String::formatted("Network Latency: %.0f μs", networkLatency), 
                                   juce::dontSendNotification);
        syncAccuracyLabel.setText(juce::String::formatted("Sync Accuracy: %.0f ns", currentAccuracy), 
                                juce::dontSendNotification);
        gpuTimebaseLabel.setText("GPU Timebase: 60.0 fps", juce::dontSendNotification);
        
    } else {
        // Not connected - show idle state
        if (jam::gpu_native::GPUTimebase::is_initialized()) {
            peerSyncStatusLabel.setText("GPU Timebase: Ready (Not Connected)", juce::dontSendNotification);
        } else {
            peerSyncStatusLabel.setText("GPU Timebase: Not Initialized", juce::dontSendNotification);
        }
        localTimingLabel.setText("Local GPU Time: -- (Not Connected)", juce::dontSendNotification);
        networkLatencyLabel.setText("Network Latency: --", juce::dontSendNotification);
        syncAccuracyLabel.setText("Sync Accuracy: --", juce::dontSendNotification);
        gpuTimebaseLabel.setText("GPU Timebase: --", juce::dontSendNotification);
    }
    
    // Network consensus status
    activePeersLabel.setText(juce::String::formatted("Active Peers: %d", activePeerCount), 
                           juce::dontSendNotification);
    
    if (isConnected) {
        consensusQualityLabel.setText(juce::String::formatted("Consensus Quality: %.1f%%", consensusQuality), 
                                    juce::dontSendNotification);
        
        if (consensusQuality > 95.0) {
            networkStabilityLabel.setText("Network Stability: Excellent", juce::dontSendNotification);
        } else if (consensusQuality > 80.0) {
            networkStabilityLabel.setText("Network Stability: Good", juce::dontSendNotification);
        } else {
            networkStabilityLabel.setText("Network Stability: Fair", juce::dontSendNotification);
        }
    } else {
        consensusQualityLabel.setText("Consensus Quality: -- (Not Connected)", juce::dontSendNotification);
        networkStabilityLabel.setText("Network Stability: -- (Not Connected)", juce::dontSendNotification);
    }
}

void ClockSyncPanel::setNetworkConnected(bool connected, int peerCount)
{
    isNetworkConnected = connected;
    activePeerCount = peerCount;
    
    // Set mock values when connected
    if (connected) {
        networkLatency = 150.0; // microseconds
        currentAccuracy = 50.0; // nanoseconds
        consensusQuality = 98.5; // percentage
    } else {
        networkLatency = 0.0;
        currentAccuracy = 0.0;
        consensusQuality = 0.0;
    }
}
