#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "ClockDriftArbiter.h"
#include <memory>

class ClockSyncPanel : public juce::Component, 
                      public juce::Timer
{
public:
    ClockSyncPanel();
    ~ClockSyncPanel() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void calibrateClicked();
    void updateDisplay();
    
    // UI Components
    juce::GroupComponent syncGroup;
    juce::Label peerSyncStatusLabel;
    juce::Label localTimingLabel;
    juce::Label networkLatencyLabel;
    juce::Label syncAccuracyLabel;
    juce::Label gpuTimebaseLabel;
    
    juce::GroupComponent networkGroup;
    juce::Label activePeersLabel;
    juce::Label consensusQualityLabel;
    juce::Label networkStabilityLabel;
    juce::TextButton calibrateButton;
    
    // GPU-Native Clock Sync Integration
    std::unique_ptr<TOAST::ClockDriftArbiter> clockArbiter;
    
    // Network consensus state (no master/slave)
    double currentAccuracy = 0.0;
    double networkLatency = 0.0;
    uint64_t gpuTimebaseNs = 0;
    int activePeerCount = 0;
    double consensusQuality = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClockSyncPanel)
};
