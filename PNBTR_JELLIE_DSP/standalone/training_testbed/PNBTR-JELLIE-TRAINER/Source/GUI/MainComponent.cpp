/*
  ==============================================================================

    MainComponent.cpp
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the exact schematic layout:
    Row 1: 4 Oscilloscopes (Input, Network Sim, Log/Status, Output)
    Row 2: 2 Waveform Analysis (Original vs Reconstructed)
    Row 3: 2 Audio Track Placeholders (JELLIE & PNBTR)
    Row 4: Metrics Dashboard (6 metrics horizontal)
    Row 5: Controls (Start/Stop/Export + sliders)

  ==============================================================================
*/

#include "MainComponent.h"
#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
MainComponent::MainComponent()
{
    // Set size to match schematic layout
    setSize(1400, 800);
    
    // Initialize SessionManager for configuration and control
    sessionManager = std::make_unique<SessionManager>();
    
    // Create all components according to schematic
    createOscilloscopes();
    createWaveformAnalysis();
    createAudioTracks();
    createMetricsDashboard();
    createControls();
    createLogStatusWindow();
    
    // Start timer for real-time updates
    startTimer(33); // ~30 FPS
}

MainComponent::~MainComponent()
{
    stopTimer();
}

//==============================================================================
void MainComponent::paint(juce::Graphics& g)
{
    // Fill background
    g.fillAll(juce::Colour(0xff1e1e1e));
    
    // Draw title
    auto bounds = getLocalBounds();
    auto titleBounds = bounds.removeFromTop(layout.titleHeight);
    drawTitle(g, titleBounds);
    
    // Draw row separators
    int currentY = layout.titleHeight + layout.oscilloscopeRowHeight;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.waveformRowHeight;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.audioTrackRowHeight;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.metricsRowHeight;
    drawRowSeparator(g, currentY, getWidth());
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Title area
    bounds.removeFromTop(layout.titleHeight);
    
    // Layout each row according to schematic
    layoutOscilloscopeRow(bounds);
    layoutWaveformRow(bounds);
    layoutAudioTrackRow(bounds);
    layoutMetricsRow(bounds);
    layoutControlsRow(bounds);
}

void MainComponent::timerCallback()
{
    // Update for real-time display
    repaint();
}

//==============================================================================
// Layout methods

void MainComponent::layoutOscilloscopeRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.oscilloscopeRowHeight).reduced(layout.margin);
    
    // Divide into 4 equal sections: Input, Network, Log, Output
    int sectionWidth = (rowBounds.getWidth() - 3 * layout.componentSpacing) / 4;
    
    // Input oscilloscope
    if (inputOscilloscope) {
        inputOscilloscope->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    // Network oscilloscope  
    if (networkOscilloscope) {
        networkOscilloscope->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    // Log/Status window
    if (logStatusWindow) {
        logStatusWindow->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    // Output oscilloscope
    if (outputOscilloscope) {
        outputOscilloscope->setBounds(rowBounds);
    }
}

void MainComponent::layoutWaveformRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.waveformRowHeight).reduced(layout.margin);
    
    // Divide into 2 equal sections: Original vs Reconstructed
    int sectionWidth = (rowBounds.getWidth() - layout.componentSpacing) / 2;
    
    if (originalWaveform) {
        originalWaveform->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    if (reconstructedWaveform) {
        reconstructedWaveform->setBounds(rowBounds);
    }
}

void MainComponent::layoutAudioTrackRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.audioTrackRowHeight).reduced(layout.margin);
    
    // Divide into 2 equal sections: JELLIE track & PNBTR track
    int sectionWidth = (rowBounds.getWidth() - layout.componentSpacing) / 2;
    
    if (jellieTrack) {
        jellieTrack->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    if (pnbtrTrack) {
        pnbtrTrack->setBounds(rowBounds);
    }
}

void MainComponent::layoutMetricsRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.metricsRowHeight).reduced(layout.margin);
    
    // Full width for metrics dashboard
    if (metricsDashboard) {
        metricsDashboard->setBounds(rowBounds);
    }
}

void MainComponent::layoutControlsRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.controlsRowHeight).reduced(layout.margin);
    
    // Divide into buttons and sliders
    auto buttonArea = rowBounds.removeFromLeft(300);
    
    // Layout buttons
    auto buttonWidth = (buttonArea.getWidth() - 2 * layout.componentSpacing) / 3;
    if (startButton) {
        startButton->setBounds(buttonArea.removeFromLeft(buttonWidth));
        buttonArea.removeFromLeft(layout.componentSpacing);
    }
    if (stopButton) {
        stopButton->setBounds(buttonArea.removeFromLeft(buttonWidth));
        buttonArea.removeFromLeft(layout.componentSpacing);
    }
    if (exportButton) {
        exportButton->setBounds(buttonArea);
    }
    
    // Layout sliders
    auto sliderWidth = (rowBounds.getWidth() - 2 * layout.componentSpacing) / 3;
    auto sliderHeight = rowBounds.getHeight() / 2 - layout.componentSpacing;
    
    // Top row: labels
    auto labelRow = rowBounds.removeFromTop(sliderHeight);
    if (packetLossLabel) {
        packetLossLabel->setBounds(labelRow.removeFromLeft(sliderWidth));
        labelRow.removeFromLeft(layout.componentSpacing);
    }
    if (jitterLabel) {
        jitterLabel->setBounds(labelRow.removeFromLeft(sliderWidth));
        labelRow.removeFromLeft(layout.componentSpacing);
    }
    if (gainLabel) {
        gainLabel->setBounds(labelRow);
    }
    
    rowBounds.removeFromTop(layout.componentSpacing);
    
    // Bottom row: sliders
    auto sliderRow = rowBounds;
    if (packetLossSlider) {
        packetLossSlider->setBounds(sliderRow.removeFromLeft(sliderWidth));
        sliderRow.removeFromLeft(layout.componentSpacing);
    }
    if (jitterSlider) {
        jitterSlider->setBounds(sliderRow.removeFromLeft(sliderWidth));
        sliderRow.removeFromLeft(layout.componentSpacing);
    }
    if (gainSlider) {
        gainSlider->setBounds(sliderRow);
    }
}

//==============================================================================
// Drawing methods

void MainComponent::drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    g.drawText("PNBTR+JELLIE Training Testbed", bounds, juce::Justification::centred);
}

void MainComponent::drawRowSeparator(juce::Graphics& g, int y, int width)
{
    g.setColour(juce::Colour(0xff444444));
    g.drawHorizontalLine(y, 0, width);
}

//==============================================================================
// Component creation methods

void MainComponent::createOscilloscopes()
{
    // Row 1: Four oscilloscopes
    inputOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::AudioInput, "Input (Mic)");
    addAndMakeVisible(*inputOscilloscope);
    
    networkOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::NetworkProcessed, "Network Sim");
    addAndMakeVisible(*networkOscilloscope);
    
    outputOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::Reconstructed, "Output (Reconstructed)");
    addAndMakeVisible(*outputOscilloscope);
}

void MainComponent::createWaveformAnalysis()
{
    // Row 2: Waveform analysis oscilloscopes
    originalWaveform = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::AudioInput, "Original Waveform");
    originalWaveform->setTimeWindow(0.5f); // Longer time window for analysis
    addAndMakeVisible(*originalWaveform);
    
    reconstructedWaveform = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::Reconstructed, "Reconstructed Waveform");
    reconstructedWaveform->setTimeWindow(0.5f);
    addAndMakeVisible(*reconstructedWaveform);
}

void MainComponent::createAudioTracks()
{
    // Row 3: Simple placeholder components for audio tracks
    jellieTrack = std::make_unique<juce::Component>();
    addAndMakeVisible(*jellieTrack);
    
    pnbtrTrack = std::make_unique<juce::Component>();
    addAndMakeVisible(*pnbtrTrack);
}

void MainComponent::createMetricsDashboard()
{
    // Row 4: Metrics dashboard
    metricsDashboard = std::make_unique<MetricsDashboard>();
    addAndMakeVisible(*metricsDashboard);
}

void MainComponent::createControls()
{
    // Row 5: Control buttons and sliders
    
    // Buttons
    startButton = std::make_unique<juce::TextButton>("Start");
    startButton->setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    startButton->onClick = [this]() { startProcessing(); };
    addAndMakeVisible(*startButton);
    
    stopButton = std::make_unique<juce::TextButton>("Stop");
    stopButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopButton->setEnabled(false);
    stopButton->onClick = [this]() { stopProcessing(); };
    addAndMakeVisible(*stopButton);
    
    exportButton = std::make_unique<juce::TextButton>("Export");
    exportButton->setColour(juce::TextButton::buttonColourId, juce::Colours::blue);
    exportButton->setEnabled(false);
    exportButton->onClick = [this]() { exportSession(); };
    addAndMakeVisible(*exportButton);
    
    // Sliders and labels
    packetLossLabel = std::make_unique<juce::Label>("", "Packet Loss");
    packetLossLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*packetLossLabel);
    
    packetLossSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    packetLossSlider->setRange(0.0, 20.0, 0.1);
    packetLossSlider->setValue(2.0);
    packetLossSlider->onValueChange = [this]() { updateNetworkParameters(); };
    addAndMakeVisible(*packetLossSlider);
    
    jitterLabel = std::make_unique<juce::Label>("", "Jitter");
    jitterLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*jitterLabel);
    
    jitterSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    jitterSlider->setRange(0.0, 10.0, 0.1);
    jitterSlider->setValue(1.0);
    jitterSlider->onValueChange = [this]() { updateNetworkParameters(); };
    addAndMakeVisible(*jitterSlider);
    
    gainLabel = std::make_unique<juce::Label>("", "Gain");
    gainLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*gainLabel);
    
    gainSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    gainSlider->setRange(0.0, 2.0, 0.01);
    gainSlider->setValue(1.0);
    gainSlider->onValueChange = [this]() { updateNetworkParameters(); };
    addAndMakeVisible(*gainSlider);
}

void MainComponent::createLogStatusWindow()
{
    // Log/Status window placeholder
    logStatusWindow = std::make_unique<juce::Component>();
    addAndMakeVisible(*logStatusWindow);
}

//==============================================================================
// Control methods

void MainComponent::startProcessing()
{
    if (!isProcessing && sessionManager) {
        sessionManager->startSession();
        isProcessing = true;
        
        startButton->setEnabled(false);
        stopButton->setEnabled(true);
        exportButton->setEnabled(false);
    }
}

void MainComponent::stopProcessing()
{
    if (isProcessing && sessionManager) {
        sessionManager->stopSession();
        isProcessing = false;
        
        startButton->setEnabled(true);
        stopButton->setEnabled(false);
        exportButton->setEnabled(true);
    }
}

void MainComponent::exportSession()
{
    if (!isProcessing && sessionManager) {
        SessionManager::ExportOptions options;
        options.sessionName = "PNBTR_JELLIE_Session";
        options.includeWaveforms = true;
        options.includeMetrics = true;
        options.includeConfig = true;
        
        sessionManager->exportSession(options);
    }
}

void MainComponent::updateNetworkParameters()
{
    if (sessionManager) {
        auto config = sessionManager->getConfig();
        config.packetLossPercent = static_cast<float>(packetLossSlider->getValue());
        config.jitterMs = static_cast<float>(jitterSlider->getValue());
        sessionManager->updateConfig(config);
    }
}
