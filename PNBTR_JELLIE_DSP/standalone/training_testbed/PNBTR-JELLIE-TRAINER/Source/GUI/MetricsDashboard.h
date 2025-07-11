#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

#include <memory>
#include "../DSP/PNBTRTrainer.h"

// Forward declarations
class MetalBridge;
class PNBTRTrainer;
class TOASTNetworkOscilloscope;
struct AudioMetrics;

/**
 * Real-time metrics dashboard component
 * Displays SNR, THD, Latency, Reconstruction Rate, Gap Fill Quality, Overall Quality
 * Connects to MetalBridge.getLatestMetrics() for GPU-computed values
 * Part of the schematic-based GUI implementation
 */
class MetricsDashboard : public juce::Component, public juce::Timer
{
public:
    void setTrainer(PNBTRTrainer* trainerPtr);
    void setTOASTNetworkOscilloscope(TOASTNetworkOscilloscope* toastOsc);
    MetricsDashboard();
    ~MetricsDashboard() override;
    
    //==============================================================================
    // Component interface
    void paint(juce::Graphics& g) override;
    void resized() override;
private:
    PNBTRTrainer* trainer = nullptr;
    TOASTNetworkOscilloscope* toastNetworkOsc = nullptr;
    // Timer callback for real-time updates
    void timerCallback() override;
    
    //==============================================================================
    // Configuration
    void setRefreshRate(int hz) { refreshRateHz = hz; startTimer(1000 / hz); }
    void setShowProgressBars(bool show) { showProgressBars = show; repaint(); }
    void setShowNumericalValues(bool show) { showValues = show; repaint(); }
    
    //==============================================================================
    // Metrics access
    void updateMetrics(); // Called by timer to read latest metrics
    //==============================================================================
    // Metric display structure
    struct MetricDisplay {
        juce::String name;
        juce::String unit;
        float currentValue = 0.0f;
        float targetValue = 1.0f;  // For progress bar scaling
        float minValue = 0.0f;
        float maxValue = 1.0f;
        juce::Colour colour;
        juce::Rectangle<int> bounds;
        juce::Rectangle<int> progressBounds;
        juce::Rectangle<int> valueBounds;
        
        MetricDisplay(const juce::String& n, const juce::String& u, 
                     float min, float max, float target, juce::Colour c)
            : name(n), unit(u), minValue(min), maxValue(max), targetValue(target), colour(c) {}
    };
    
    //==============================================================================
    // Metrics configuration
    std::vector<std::unique_ptr<MetricDisplay>> metrics;
    MetalBridge* metalBridge;
    
    //==============================================================================
    // Display parameters
    int refreshRateHz = 30; // Lower rate for metrics
    bool showProgressBars = true;
    bool showValues = true;
    bool isActive = false;
    
    //==============================================================================
    // Visual styling
    juce::Colour backgroundColour{0xff2a2a2a};
    juce::Colour borderColour{0xff444444};
    juce::Colour textColour{0xffeeeeee};
    juce::Colour progressBackgroundColour{0xff1a1a1a};
    
    //==============================================================================
    // Layout parameters
    int titleHeight = 30;
    int metricHeight = 50;
    int progressBarHeight = 12;
    int margin = 8;
    
    //==============================================================================
    // Internal methods
    void initializeMetrics();
    void updateMetricValues();
    void drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawMetric(juce::Graphics& g, const MetricDisplay& metric);
    void drawProgressBar(juce::Graphics& g, const MetricDisplay& metric);
    void drawNumericalValue(juce::Graphics& g, const MetricDisplay& metric);
    
    // Utility methods
    float normalizeValue(float value, float min, float max) const;
    juce::String formatValue(float value, const juce::String& unit) const;
    juce::Colour getQualityColour(float normalizedValue) const;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MetricsDashboard)
}; 