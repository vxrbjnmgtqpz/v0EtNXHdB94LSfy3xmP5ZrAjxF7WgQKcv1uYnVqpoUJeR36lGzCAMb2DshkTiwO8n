#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class ClockSyncPanel : public juce::Component
{
public:
    ClockSyncPanel();
    ~ClockSyncPanel() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void enableSyncClicked();
    void calibrateClicked();
    
    juce::GroupComponent syncGroup;
    juce::ToggleButton masterModeButton;
    juce::Label syncStatusLabel;
    juce::Label networkOffsetLabel;
    juce::Label qualityLabel;
    
    juce::GroupComponent settingsGroup;
    juce::Label syncRateLabel;
    juce::Slider syncRateSlider;
    
    bool syncEnabled = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClockSyncPanel)
};
