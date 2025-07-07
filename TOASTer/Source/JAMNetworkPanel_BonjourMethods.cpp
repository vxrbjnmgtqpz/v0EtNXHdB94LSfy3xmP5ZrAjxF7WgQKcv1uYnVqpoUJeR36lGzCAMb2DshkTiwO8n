#include "JAMNetworkPanel.h"
#include <string>

void JAMNetworkPanel::deviceDiscovered(const BonjourDiscovery::DiscoveredDevice& device) {
    // Bonjour discovered a device - could auto-suggest connection
    juce::Logger::writeToLog("Bonjour discovered device: " + juce::String(device.name) + " at " + juce::String(device.hostname));
}

void JAMNetworkPanel::deviceLost(const std::string& deviceName) {
    juce::Logger::writeToLog("Bonjour lost device: " + juce::String(deviceName));
}

void JAMNetworkPanel::deviceConnected(const BonjourDiscovery::DiscoveredDevice& device) {
    juce::Logger::writeToLog("Bonjour connected to device: " + juce::String(device.name));
}

void JAMNetworkPanel::timerCallback() {
    updateUI();
}

void JAMNetworkPanel::updateUI() {
    // Update any dynamic UI elements that need periodic refresh
    if (jamFramework && networkConnected) {
        // Get latest performance metrics
        auto metrics = jamFramework->getPerformanceMetrics();
        
        currentLatency = metrics.latency_us;
        currentThroughput = metrics.throughput_mbps;
        activePeers = metrics.active_peers;
        predictionAccuracy = metrics.prediction_accuracy;
        
        updatePerformanceDisplay();
    }
}

void JAMNetworkPanel::updatePerformanceDisplay() {
    latencyLabel.setText("Latency: " + juce::String(currentLatency, 1) + " μs", juce::dontSendNotification);
    throughputLabel.setText("Throughput: " + juce::String(currentThroughput, 2) + " Mbps", juce::dontSendNotification);
    peersLabel.setText("Peers: " + juce::String(activePeers), juce::dontSendNotification);
    
    if (predictionAccuracy > 0.0) {
        predictionLabel.setText("Prediction: " + juce::String(predictionAccuracy * 100.0, 1) + "%", juce::dontSendNotification);
    } else {
        predictionLabel.setText("Prediction: --", juce::dontSendNotification);
    }
    
    // Color code latency
    if (currentLatency < 100.0) {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    } else if (currentLatency < 1000.0) {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    } else {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
}
