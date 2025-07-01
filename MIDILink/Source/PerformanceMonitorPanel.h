#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "JSONMIDIParser.h"
#include <memory>

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
    juce::Label frameworkLatencyLabel;
    juce::Label networkLatencyLabel;
    juce::Label messageProcessingLabel;
    juce::Label clockAccuracyLabel;
    juce::Label midiThroughputLabel;
    juce::Label connectionStatsLabel;
    juce::Label memoryUsageLabel;
    
    // Framework integration
    // std::unique_ptr<JSONMIDI::PerformanceProfiler> profiler; // Not implemented yet
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PerformanceMonitorPanel)
};