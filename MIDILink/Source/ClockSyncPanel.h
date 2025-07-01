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
    void toggleSync();
    void toggleForceMaster();
    void calibrateClicked();
    void updateDisplay();
    
    // UI Components
    juce::GroupComponent syncGroup;
    juce::ToggleButton enableSyncToggle;
    juce::ToggleButton forceMasterToggle;
    juce::Label roleLabel;
    juce::Label syncStatusLabel;
    juce::Label networkOffsetLabel;
    juce::Label syncQualityLabel;
    juce::Label rttLabel;
    
    juce::GroupComponent settingsGroup;
    juce::Label syncRateLabel;
    juce::Slider syncRateSlider;
    juce::TextButton calibrateButton;
    
    // Clock Sync Integration
    std::unique_ptr<TOAST::ClockDriftArbiter> clockArbiter;
    
    bool syncEnabled = false;
    TOAST::ClockRole currentRole = TOAST::ClockRole::UNINITIALIZED;
    double currentQuality = 0.0;
    double currentOffset = 0.0;
    uint64_t currentRTT = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClockSyncPanel)
};
