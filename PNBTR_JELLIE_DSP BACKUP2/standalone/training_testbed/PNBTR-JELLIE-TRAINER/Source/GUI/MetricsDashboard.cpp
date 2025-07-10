
#include "MetricsDashboard.h"
#include "TOASTNetworkOscilloscope.h"
#include "../DSP/PNBTRTrainer.h"
#include "../Metrics/TrainingMetrics.h"
#include "../GPU/MetalBridgeInterface.h"
#include <algorithm>
#include <iomanip>
#include <sstream>

void MetricsDashboard::setTrainer(PNBTRTrainer* trainerPtr) { trainer = trainerPtr; }

void MetricsDashboard::setTOASTNetworkOscilloscope(TOASTNetworkOscilloscope* toastOsc) 
{ 
    toastNetworkOsc = toastOsc; 
}

//==============================================================================
MetricsDashboard::MetricsDashboard()
{
    // Get MetalBridge singleton
    metalBridge = &MetalBridgeInterface::getInstance();
    
    // Initialize metrics displays
    initializeMetrics();
    
    // Start timer for updates
    startTimer(1000 / refreshRateHz);
    
    // Set initial size
    setSize(800, 120);
}

MetricsDashboard::~MetricsDashboard()
{
    stopTimer();
}

//==============================================================================
void MetricsDashboard::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Draw background
    g.fillAll(backgroundColour);
    g.setColour(borderColour);
    g.drawRect(bounds, 1);
    
    // Draw title
    auto titleArea = bounds.removeFromTop(titleHeight);
    drawTitle(g, titleArea);
    
    // Draw metrics
    for (const auto& metric : metrics) {
        drawMetric(g, *metric);
    }
}

void MetricsDashboard::resized()
{
    auto bounds = getLocalBounds().reduced(margin);
    bounds.removeFromTop(titleHeight); // Skip title area
    
    // Calculate layout for metrics
    int numMetrics = static_cast<int>(metrics.size());
    if (numMetrics == 0) return;
    
    int metricWidth = bounds.getWidth() / numMetrics;
    
    for (int i = 0; i < numMetrics; ++i) {
        auto metricBounds = bounds.removeFromLeft(metricWidth).reduced(margin / 2);
        
        metrics[i]->bounds = metricBounds;
        
        // Subdivide metric bounds
        auto nameBounds = metricBounds.removeFromTop(20);
        auto progressBounds = metricBounds.removeFromTop(progressBarHeight);
        auto valueBounds = metricBounds;
        
        metrics[i]->progressBounds = progressBounds;
        metrics[i]->valueBounds = valueBounds;
    }
}

void MetricsDashboard::timerCallback()
{
    // Store previous metrics to detect changes
    std::vector<float> oldValues;
    for (const auto& metric : metrics) {
        oldValues.push_back(metric->currentValue);
    }
    
    updateMetrics();
    
    // Only repaint if metrics actually changed (massive performance improvement)
    bool valuesChanged = false;
    for (size_t i = 0; i < metrics.size() && i < oldValues.size(); ++i) {
        if (std::abs(metrics[i]->currentValue - oldValues[i]) > 0.01f) {
            valuesChanged = true;
            break;
        }
    }
    
    if (valuesChanged) {
        repaint();
    }
}

//==============================================================================
void MetricsDashboard::updateMetrics()
{
    updateMetricValues();
}

void MetricsDashboard::initializeMetrics()
{
    // Clear existing metrics
    metrics.clear();
    
    // SNR (Signal-to-Noise Ratio) - higher is better, dB scale
    metrics.push_back(std::make_unique<MetricDisplay>(
        "SNR", "dB", 0.0f, 60.0f, 40.0f, juce::Colours::cyan));
    
    // THD (Total Harmonic Distortion) - lower is better, percentage
    metrics.push_back(std::make_unique<MetricDisplay>(
        "THD", "%", 0.0f, 10.0f, 1.0f, juce::Colours::orange));
    
    // Latency - lower is better, milliseconds
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Latency", "ms", 0.0f, 100.0f, 10.0f, juce::Colours::yellow));
    
    // Reconstruction Rate - higher is better, percentage
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Recon Rate", "%", 0.0f, 100.0f, 95.0f, juce::Colours::green));
    
    // Gap Fill Quality - higher is better, normalized 0-1
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Gap Fill", "", 0.0f, 1.0f, 0.9f, juce::Colours::lightblue));
    
    // Overall Quality - higher is better, normalized 0-1
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Quality", "", 0.0f, 1.0f, 0.85f, juce::Colours::lightgreen));
    
    // Network Packet Loss from TOAST protocol - lower is better, percentage
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Packet Loss", "%", 0.0f, 10.0f, 1.0f, juce::Colours::red));
    
    // Network Jitter from TOAST protocol - lower is better, milliseconds
    metrics.push_back(std::make_unique<MetricDisplay>(
        "Jitter", "ms", 0.0f, 50.0f, 5.0f, juce::Colours::purple));
}

void MetricsDashboard::updateMetricValues()
{
    // Generate realistic test values for now
    // TODO: Replace with actual MetalBridge.getLatestMetrics() calls
    
    if (trainer && metrics.size() >= 6) {
        auto* tm = trainer->getMetrics();
        if (tm) {
            metrics[0]->currentValue = tm->getSNR();
            metrics[1]->currentValue = tm->getTHD();
            metrics[2]->currentValue = tm->getLatency();
            // For now, use packet loss as recon rate, and gap fill/quality as averages
            metrics[3]->currentValue = 100.0f - tm->getPacketLossPercentage();
            metrics[4]->currentValue = tm->getAverageSNR() / 60.0f; // Normalized
            metrics[5]->currentValue = tm->getAverageLatency() < 20.0f ? 1.0f : 0.7f; // Example
        }
        
        // Update TOAST network metrics if available
        if (toastNetworkOsc && metrics.size() >= 8) {
            auto networkMetrics = toastNetworkOsc->getCurrentMetrics();
            metrics[6]->currentValue = networkMetrics.packet_loss_rate * 100.0f; // Convert to percentage
            metrics[7]->currentValue = networkMetrics.jitter_ms;
        }
        
        isActive = true;
    } else {
        // Fallback: show zeroes
        for (auto& m : metrics) m->currentValue = 0.0f;
        isActive = false;
    }
}

//==============================================================================
// Drawing methods

void MetricsDashboard::drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(textColour);
    g.setFont(juce::Font(juce::FontOptions(16.0f, juce::Font::bold)));
    g.drawText("Metrics Dashboard", bounds, juce::Justification::centred);
    
    // Add activity indicator
    if (isActive) {
        g.setColour(juce::Colours::green);
        g.fillEllipse(bounds.getRight() - 20, bounds.getCentreY() - 3, 6, 6);
    }
}

void MetricsDashboard::drawMetric(juce::Graphics& g, const MetricDisplay& metric)
{
    auto bounds = metric.bounds;
    
    // Draw metric name
    g.setColour(textColour);
    g.setFont(juce::Font(juce::FontOptions(12.0f, juce::Font::bold)));
    auto nameBounds = bounds.removeFromTop(20);
    g.drawText(metric.name, nameBounds, juce::Justification::centred);
    
    // Draw progress bar if enabled
    if (showProgressBars) {
        drawProgressBar(g, metric);
    }
    
    // Draw numerical value if enabled
    if (showValues) {
        drawNumericalValue(g, metric);
    }
}

void MetricsDashboard::drawProgressBar(juce::Graphics& g, const MetricDisplay& metric)
{
    auto bounds = metric.progressBounds;
    
    // Draw background
    g.setColour(progressBackgroundColour);
    g.fillRoundedRectangle(bounds.toFloat(), 2.0f);
    
    // Calculate fill percentage
    float normalizedValue = normalizeValue(metric.currentValue, metric.minValue, metric.maxValue);
    float fillWidth = bounds.getWidth() * std::clamp(normalizedValue, 0.0f, 1.0f);
    
    // Draw fill with quality-based color
    g.setColour(getQualityColour(normalizedValue));
    auto fillBounds = bounds.withWidth(static_cast<int>(fillWidth));
    g.fillRoundedRectangle(fillBounds.toFloat(), 2.0f);
    
    // Draw border
    g.setColour(borderColour);
    g.drawRoundedRectangle(bounds.toFloat(), 2.0f, 1.0f);
}

void MetricsDashboard::drawNumericalValue(juce::Graphics& g, const MetricDisplay& metric)
{
    auto bounds = metric.valueBounds;
    
    g.setColour(textColour);
    g.setFont(juce::Font(juce::FontOptions(14.0f)));
    
    juce::String valueText = formatValue(metric.currentValue, metric.unit);
    g.drawText(valueText, bounds, juce::Justification::centred);
}

//==============================================================================
// Utility methods

float MetricsDashboard::normalizeValue(float value, float min, float max) const
{
    if (max <= min) return 0.0f;
    return (value - min) / (max - min);
}

juce::String MetricsDashboard::formatValue(float value, const juce::String& unit) const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value;
    return juce::String(oss.str()) + (unit.isEmpty() ? "" : " " + unit);
}

juce::Colour MetricsDashboard::getQualityColour(float normalizedValue) const
{
    // Green for good values, yellow for medium, red for poor
    if (normalizedValue >= 0.8f) {
        return juce::Colours::green;
    } else if (normalizedValue >= 0.5f) {
        return juce::Colours::yellow;
    } else if (normalizedValue >= 0.3f) {
        return juce::Colours::orange;
    } else {
        return juce::Colours::red;
    }
} 