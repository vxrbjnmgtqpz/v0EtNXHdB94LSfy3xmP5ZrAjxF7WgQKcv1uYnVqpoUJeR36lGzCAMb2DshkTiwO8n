#include "ClockSyncPanel.h"
#include <sstream>
#include <iomanip>

ClockSyncPanel::ClockSyncPanel()
    : syncGroup("Clock Sync"), settingsGroup("Sync Settings"),
      enableSyncButton("Enable Sync"), masterModeButton("Force Master"), 
      syncRateLabel("Sync Rate:"), calibrateButton("Calibrate"),
      syncRateSlider(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight)
{
    // Initialize ClockDriftArbiter
    clockArbiter = std::make_unique<TOAST::ClockDriftArbiter>();
    
    // Set up sync group
    syncGroup.setText("🕐 Clock Synchronization Status");
    addAndMakeVisible(syncGroup);
    
    // Set up enable sync button
    enableSyncButton.setToggleState(false, juce::dontSendNotification);
    enableSyncButton.onClick = [this] { enableSyncClicked(); };
    addAndMakeVisible(enableSyncButton);
    
    // Set up master mode button
    masterModeButton.setToggleState(false, juce::dontSendNotification);
    masterModeButton.setEnabled(false);
    addAndMakeVisible(masterModeButton);
    
    // Set up status labels
    roleLabel.setText("Role: Uninitialized", juce::dontSendNotification);
    addAndMakeVisible(roleLabel);
    
    syncStatusLabel.setText("Status: ❌ Disabled", juce::dontSendNotification);
    addAndMakeVisible(syncStatusLabel);
    
    networkOffsetLabel.setText("Network Offset: -- μs", juce::dontSendNotification);
    addAndMakeVisible(networkOffsetLabel);
    
    qualityLabel.setText("Sync Quality: -- %", juce::dontSendNotification);
    addAndMakeVisible(qualityLabel);
    
    rttLabel.setText("Round Trip: -- μs", juce::dontSendNotification);
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
    enableSyncButton.setBounds(controlRow.removeFromLeft(100));
    controlRow.removeFromLeft(10);
    masterModeButton.setBounds(controlRow.removeFromLeft(100));
    
    syncContentBounds.removeFromTop(10);
    
    // Status labels
    roleLabel.setBounds(syncContentBounds.removeFromTop(20));
    syncStatusLabel.setBounds(syncContentBounds.removeFromTop(20));
    networkOffsetLabel.setBounds(syncContentBounds.removeFromTop(20));
    qualityLabel.setBounds(syncContentBounds.removeFromTop(20));
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

void ClockSyncPanel::enableSyncClicked()
{
    syncEnabled = enableSyncButton.getToggleState();
    
    if (syncEnabled && clockArbiter) {
        // Start clock synchronization
        double syncRate = syncRateSlider.getValue();
        bool forceMaster = masterModeButton.getToggleState();
        
        try {
            // clockArbiter->start(this); // Will implement callbacks later
            if (forceMaster) {
                clockArbiter->forceMasterRole();
                currentRole = TOAST::ClockRole::MASTER;
            } else {
                clockArbiter->startMasterElection();
                currentRole = TOAST::ClockRole::CANDIDATE;
            }
            
            syncStatusLabel.setText("Status: ✅ Synchronizing", juce::dontSendNotification);
            masterModeButton.setEnabled(false); // Can't change while running
        } catch (const std::exception& e) {
            syncStatusLabel.setText("Status: ❌ Start failed", juce::dontSendNotification);
            enableSyncButton.setToggleState(false, juce::dontSendNotification);
            syncEnabled = false;
        }
    } else if (clockArbiter) {
        // Stop clock synchronization
        clockArbiter->shutdown();
        syncStatusLabel.setText("Status: ⏹️ Stopped", juce::dontSendNotification);
        masterModeButton.setEnabled(true);
        
        // Reset display values
        currentRole = TOAST::ClockRole::UNINITIALIZED;
        currentQuality = 0.0;
        currentOffset = 0.0;
        currentRTT = 0;
        updateDisplay();
    }
}

void ClockSyncPanel::calibrateClicked()
{
    if (clockArbiter && syncEnabled) {
        clockArbiter->startMasterElection();
        syncStatusLabel.setText("Status: 🔄 Calibrating...", juce::dontSendNotification);
    }
}

void ClockSyncPanel::timerCallback()
{
    updateDisplay();
    
    // Simulate some timing data for demo purposes
    if (syncEnabled) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        
        // Simulate realistic sync metrics
        currentOffset = 0.5 + (std::sin(elapsed * 0.1) * 0.3);
        currentQuality = 0.85 + (std::sin(elapsed * 0.2) * 0.1);
        currentRTT = 850 + (std::sin(elapsed * 0.15) * 150);
        
        if (currentRole == TOAST::ClockRole::MASTER) {
            currentOffset = 0.0; // Master has no offset
        }
    }
}

void ClockSyncPanel::updateDisplay()
{
    // Update role display
    std::string roleText = "Role: ";
    switch (currentRole) {
        case TOAST::ClockRole::MASTER:
            roleText += "👑 Master";
            break;
        case TOAST::ClockRole::SLAVE:
            roleText += "🎯 Slave";
            break;
        case TOAST::ClockRole::CANDIDATE:
            roleText += "⏳ Candidate";
            break;
        case TOAST::ClockRole::UNINITIALIZED:
        default:
            roleText += "❓ Uninitialized";
            break;
    }
    roleLabel.setText(roleText, juce::dontSendNotification);
    
    // Update offset display
    std::ostringstream offsetStream;
    offsetStream << "Network Offset: " << std::fixed << std::setprecision(1) << currentOffset << " μs";
    networkOffsetLabel.setText(offsetStream.str(), juce::dontSendNotification);
    
    // Update quality display  
    std::ostringstream qualityStream;
    qualityStream << "Sync Quality: " << std::fixed << std::setprecision(1) << (currentQuality * 100.0) << " %";
    qualityLabel.setText(qualityStream.str(), juce::dontSendNotification);
    
    // Update RTT display
    std::ostringstream rttStream;
    rttStream << "Round Trip: " << currentRTT << " μs";
    rttLabel.setText(rttStream.str(), juce::dontSendNotification);
}