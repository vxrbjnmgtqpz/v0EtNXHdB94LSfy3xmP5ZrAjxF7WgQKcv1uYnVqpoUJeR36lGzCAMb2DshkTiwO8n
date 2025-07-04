#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "JMIDParser.h"
#include <memory>

// Forward declarations for framework components
namespace TOAST {
    class ClockDriftArbiter;
    class ConnectionManager;
}

//==============================================================================
class PerformanceMonitorPanel : public juce::Component, private juce::Timer
{
public:
    PerformanceMonitorPanel();
    ~PerformanceMonitorPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    
    // Methods to receive real data from other components
    void setConnectionState(bool connected, int activeConnections = 0);
    void setNetworkLatency(double latencyMs);
    void setClockAccuracy(double accuracyUs);
    void setMessageProcessingRate(int messagesPerSecond);
    void setMIDIThroughput(int midiMessagesPerSecond);
    void updateMemoryUsage();

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
    
    // Real data storage
    bool isConnected = false;
    int activeConnections = 0;
    double networkLatency = 0.0;
    double clockAccuracy = 0.0;
    int messageProcessingRate = 0;
    int midiThroughput = 0;
    double memoryUsage = 0.0;
    
    // Framework latency tracking
    std::chrono::high_resolution_clock::time_point lastFrameworkTest;
    double frameworkLatency = 0.0;
    
    // Framework integration
    // std::unique_ptr<JMID::PerformanceProfiler> profiler; // Not implemented yet
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PerformanceMonitorPanel)
};