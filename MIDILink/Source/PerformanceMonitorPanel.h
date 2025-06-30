#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
class PerformanceMonitorPanel : public juce::Component, private juce::Timer
{
public:
    PerformanceMonitorPanel();
    ~PerformanceMonitorPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void updateMetrics();
    void timerCallback() override { updateMetrics(); }
    
    juce::Label titleLabel;
    juce::Label latencyLabel;
    juce::Label cpuUsageLabel;
    juce::Label memoryUsageLabel;
    juce::Label networkStatsLabel;
    juce::Label midiThroughputLabel;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PerformanceMonitorPanel)
};