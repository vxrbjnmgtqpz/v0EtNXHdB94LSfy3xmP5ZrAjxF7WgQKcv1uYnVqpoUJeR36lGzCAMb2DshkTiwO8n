#include "ClockSyncPanel.h"
#include <sstream>
#include <iomanip>

ClockSyncPanel::ClockSyncPanel()
    : syncGroup("Clock Synchronization Status"), settingsGroup("Sync Settings"),
      enableSyncToggle("Enable Sync"), forceMasterToggle("Force Master"), 
      syncRateLabel("Sync Rate:"), calibrateButton("Calibrate"),
      syncRateSlider(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight)
{
    // Initialize ClockDriftArbiter
    clockArbiter = std::make_unique<TOAST::ClockDriftArbiter>();
    
    // Set up sync group
    syncGroup.setText("Clock Synchronization Status");
    syncGroup.setTextLabelPosition(juce::Justification::centredTop);
    addAndMakeVisible(syncGroup);
    
    // Set up controls
    enableSyncToggle.setButtonText("Enable Sync");
    enableSyncToggle.onClick = [this] { toggleSync(); };
    addAndMakeVisible(enableSyncToggle);
    
    forceMasterToggle.setButtonText("Force Master");
    forceMasterToggle.onClick = [this] { toggleForceMaster(); };
    addAndMakeVisible(forceMasterToggle);
    
    // Set up status labels
    roleLabel.setText("Role: Uninitialized", juce::dontSendNotification);
    addAndMakeVisible(roleLabel);
    
    syncStatusLabel.setText("Status: Disabled", juce::dontSendNotification);
    addAndMakeVisible(syncStatusLabel);
    
    networkOffsetLabel.setText("Network Offset: -- us", juce::dontSendNotification);
    addAndMakeVisible(networkOffsetLabel);
    
    syncQualityLabel.setText("Sync Quality: 0.0 %", juce::dontSendNotification);
    addAndMakeVisible(syncQualityLabel);
    
    rttLabel.setText("Round Trip: -- us", juce::dontSendNotification);
    addAndMakeVisible(rttLabel);
    
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
    
    // Set up calibrate button
    calibrateButton.onClick = [this] { calibrateClicked(); };
    addAndMakeVisible(calibrateButton);
    
    // Start update timer
    startTimer(100); // Update display 10 times per second
}

ClockSyncPanel::~ClockSyncPanel()
{
    stopTimer();
    if (clockArbiter && syncEnabled) {
        clockArbiter->shutdown();
    }
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
    
    // Sync group takes most of the space
    auto syncBounds = bounds.removeFromTop(bounds.getHeight() * 0.7f);
    syncGroup.setBounds(syncBounds);
    
    auto syncContentBounds = syncBounds.reduced(15, 25);
    
    // Control buttons row
    auto controlRow = syncContentBounds.removeFromTop(25);
    enableSyncToggle.setBounds(controlRow.removeFromLeft(100));
    controlRow.removeFromLeft(10);
    forceMasterToggle.setBounds(controlRow.removeFromLeft(100));
    
    syncContentBounds.removeFromTop(10);
    
    // Status labels
    roleLabel.setBounds(syncContentBounds.removeFromTop(20));
    syncStatusLabel.setBounds(syncContentBounds.removeFromTop(20));
    networkOffsetLabel.setBounds(syncContentBounds.removeFromTop(20));
    syncQualityLabel.setBounds(syncContentBounds.removeFromTop(20));
    rttLabel.setBounds(syncContentBounds.removeFromTop(20));
    
    bounds.removeFromTop(10); // spacing
    
    // Settings group takes remaining space
    settingsGroup.setBounds(bounds);
    auto settingsContentBounds = bounds.reduced(15, 25);
    
    auto row = settingsContentBounds.removeFromTop(25);
    syncRateLabel.setBounds(row.removeFromLeft(80));
    syncRateSlider.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(10);
    calibrateButton.setBounds(row.removeFromLeft(80));
}

void ClockSyncPanel::toggleSync()
{
    syncEnabled = enableSyncToggle.getToggleState();
    
    if (syncEnabled && clockArbiter) {
        // Start clock synchronization
        double syncRate = syncRateSlider.getValue();
        bool forceMaster = forceMasterToggle.getToggleState();
        
        try {
            // clockArbiter->start(this); // Will implement callbacks later
            if (forceMaster) {
                clockArbiter->forceMasterRole();
                currentRole = TOAST::ClockRole::MASTER;
            } else {
                clockArbiter->startMasterElection();
                currentRole = TOAST::ClockRole::CANDIDATE;
            }
            
            syncStatusLabel.setText("Status: Synchronizing", juce::dontSendNotification);
            forceMasterToggle.setEnabled(false); // Can't change while running
        } catch (const std::exception& e) {
            syncStatusLabel.setText("Status: Start failed", juce::dontSendNotification);
            enableSyncToggle.setToggleState(false, juce::dontSendNotification);
            syncEnabled = false;
        }
    } else if (clockArbiter) {
        // Stop clock synchronization
        clockArbiter->shutdown();
        syncStatusLabel.setText("Status: Stopped", juce::dontSendNotification);
        forceMasterToggle.setEnabled(true);
        
        // Reset display values
        currentRole = TOAST::ClockRole::UNINITIALIZED;
        currentQuality = 0.0;
        currentOffset = 0.0;
        currentRTT = 0;
        updateDisplay();
    }
}

void ClockSyncPanel::toggleForceMaster()
{
    // Implementation needed
}

void ClockSyncPanel::calibrateClicked()
{
    if (clockArbiter && syncEnabled) {
        clockArbiter->startMasterElection();
        syncStatusLabel.setText("Status: Calibrating...", juce::dontSendNotification);
    }
}

void ClockSyncPanel::timerCallback()
{
    updateDisplay();
    
    // Get real timing data from ClockDriftArbiter if available and sync is enabled
    if (syncEnabled && clockArbiter) {
        try {
            // Get real data from ClockDriftArbiter
            auto connectedPeers = clockArbiter->getConnectedPeers();
            
            if (!connectedPeers.empty()) {
                // Get data from the first connected peer
                const std::string& peerId = connectedPeers[0];
                
                // Get real network metrics
                double realLatency = clockArbiter->getNetworkLatency(peerId);
                double realDrift = clockArbiter->getClockDrift(peerId);
                
                // Update with real data
                currentOffset = realDrift;
                currentQuality = std::max(0.0, 1.0 - (realLatency / 100.0)); // Quality based on latency
                currentRTT = static_cast<uint64_t>(realLatency * 2.0 * 1000.0); // Convert to μs
                
                if (currentRole == TOAST::ClockRole::MASTER) {
                    currentOffset = 0.0; // Master has no offset
                    currentQuality = 1.0; // Master has perfect quality
                }
            } else {
                // No peers connected, reset values
                currentOffset = 0.0;
                currentQuality = 0.0;
                currentRTT = 0;
            }
        } catch (const std::exception& e) {
            // Fall back to showing no data if there's an error
            currentOffset = 0.0;
            currentQuality = 0.0;
            currentRTT = 0;
        }
    } else {
        // Sync not enabled, show no data
        currentOffset = 0.0;
        currentQuality = 0.0;
        currentRTT = 0;
    }
}

void ClockSyncPanel::updateDisplay()
{
    // Update role display
    std::string roleText = "Role: ";
    if (currentRole == TOAST::ClockRole::MASTER) {
        roleText += "Master";
    } else if (currentRole == TOAST::ClockRole::SLAVE) {
        roleText += "Slave";
    } else if (currentRole == TOAST::ClockRole::CANDIDATE) {
        roleText += "Candidate";
    } else {
        roleText += "Uninitialized";
    }
    roleLabel.setText(roleText, juce::dontSendNotification);
    
    // Update offset display
    std::ostringstream offsetStream;
    offsetStream << "Network Offset: " << std::fixed << std::setprecision(1) << currentOffset << " us";
    networkOffsetLabel.setText(offsetStream.str(), juce::dontSendNotification);
    
    // Update quality display  
    std::ostringstream qualityStream;
    qualityStream << "Sync Quality: " << std::fixed << std::setprecision(1) << (currentQuality * 100.0) << " %";
    syncQualityLabel.setText(qualityStream.str(), juce::dontSendNotification);
    
    // Update RTT display
    std::ostringstream rttStream;
    rttStream << "Round Trip: " << currentRTT << " us";
    rttLabel.setText(rttStream.str(), juce::dontSendNotification);
}