#include "SchematicMainWindow.h"
#include "../Core/SessionManager.h"

//==============================================================================
SchematicMainWindow::SchematicMainWindow()
    : juce::DocumentWindow("PNBTR+JELLIE Training Testbed - Schematic View", 
                          juce::Colours::darkgrey, 
                          juce::DocumentWindow::allButtons)
{
    // Create session manager
    sessionManager = std::make_unique<SessionManager>();
    
    // Create content component
    content = std::make_unique<SchematicContent>();
    setContentOwned(content.release(), true);
    
    // Set window properties
    setResizable(true, true);
    setUsingNativeTitleBar(true);
    
    // Set initial size to match schematic layout
    setSize(1400, 700);
    
    // Center on screen
    centreWithSize(getWidth(), getHeight());
}

SchematicMainWindow::~SchematicMainWindow()
{
    // Content is owned by DocumentWindow, so it will be cleaned up automatically
}

//==============================================================================
void SchematicMainWindow::closeButtonPressed()
{
    // Hide instead of closing to allow reopening
    setVisible(false);
}

void SchematicMainWindow::resized()
{
    juce::DocumentWindow::resized();
}

void SchematicMainWindow::updateLayout()
{
    if (content) {
        content->resized();
    }
}

void SchematicMainWindow::showWindow()
{
    setVisible(true);
    toFront(true);
}

void SchematicMainWindow::hideWindow()
{
    setVisible(false);
}

//==============================================================================
// SchematicContent implementation

SchematicMainWindow::SchematicContent::SchematicContent()
{
    // Create oscilloscope components for Row 1
    inputOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::AudioInput, "Input (Mic)");
    addAndMakeVisible(*inputOscilloscope);
    
    networkOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::NetworkProcessed, "Network Sim");
    addAndMakeVisible(*networkOscilloscope);
    
    outputOscilloscope = std::make_unique<OscilloscopeComponent>(
        OscilloscopeComponent::BufferType::Reconstructed, "Output (Reconstructed)");
    addAndMakeVisible(*outputOscilloscope);
    
    // Create metrics dashboard for Row 4
    metricsDashboard = std::make_unique<MetricsDashboard>();
    addAndMakeVisible(*metricsDashboard);
    
    // Create placeholder components
    createPlaceholderComponents();
}

SchematicMainWindow::SchematicContent::~SchematicContent()
{
    // Unique pointers will clean up automatically
}

void SchematicMainWindow::SchematicContent::paint(juce::Graphics& g)
{
    // Draw background
    g.fillAll(backgroundColour);
    
    // Draw row separators
    int currentY = layout.oscilloscopeRowHeight + layout.margin;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.waveformRowHeight + layout.margin;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.audioTracksRowHeight + layout.margin;
    drawRowSeparator(g, currentY, getWidth());
    
    currentY += layout.metricsRowHeight + layout.margin;
    drawRowSeparator(g, currentY, getWidth());
}

void SchematicMainWindow::SchematicContent::resized()
{
    auto bounds = getLocalBounds().reduced(layout.margin);
    
    // Layout each row according to schematic
    layoutOscilloscopeRow(bounds);
    layoutWaveformRow(bounds);
    layoutAudioTracksRow(bounds);
    layoutMetricsRow(bounds);
    layoutControlsRow(bounds);
}

//==============================================================================
// Layout methods

void SchematicMainWindow::SchematicContent::layoutOscilloscopeRow(juce::Rectangle<int>& bounds)
{
    auto rowBounds = bounds.removeFromTop(layout.oscilloscopeRowHeight);
    
    // Divide into 4 equal sections: Input, Network, Log, Output
    int sectionWidth = (rowBounds.getWidth() - 3 * layout.componentSpacing) / 4;
    
    // Input oscilloscope
    inputOscilloscope->setBounds(rowBounds.removeFromLeft(sectionWidth));
    rowBounds.removeFromLeft(layout.componentSpacing);
    
    // Network oscilloscope  
    networkOscilloscope->setBounds(rowBounds.removeFromLeft(sectionWidth));
    rowBounds.removeFromLeft(layout.componentSpacing);
    
    // Log window (placeholder)
    if (logWindow) {
        logWindow->setBounds(rowBounds.removeFromLeft(sectionWidth));
        rowBounds.removeFromLeft(layout.componentSpacing);
    }
    
    // Output oscilloscope
    outputOscilloscope->setBounds(rowBounds);
}

void SchematicMainWindow::SchematicContent::layoutWaveformRow(juce::Rectangle<int>& bounds)
{
    bounds.removeFromTop(layout.margin); // Space between rows
    auto rowBounds = bounds.removeFromTop(layout.waveformRowHeight);
    
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

void SchematicMainWindow::SchematicContent::layoutAudioTracksRow(juce::Rectangle<int>& bounds)
{
    bounds.removeFromTop(layout.margin); // Space between rows
    auto rowBounds = bounds.removeFromTop(layout.audioTracksRowHeight);
    
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

void SchematicMainWindow::SchematicContent::layoutMetricsRow(juce::Rectangle<int>& bounds)
{
    bounds.removeFromTop(layout.margin); // Space between rows
    auto rowBounds = bounds.removeFromTop(layout.metricsRowHeight);
    
    // Full width for metrics dashboard
    if (metricsDashboard) {
        metricsDashboard->setBounds(rowBounds);
    }
}

void SchematicMainWindow::SchematicContent::layoutControlsRow(juce::Rectangle<int>& bounds)
{
    bounds.removeFromTop(layout.margin); // Space between rows
    auto rowBounds = bounds.removeFromTop(layout.controlsRowHeight);
    
    // Full width for controls panel
    if (controlsPanel) {
        controlsPanel->setBounds(rowBounds);
    }
}

//==============================================================================
// Helper methods

void SchematicMainWindow::SchematicContent::createPlaceholderComponents()
{
    // Create placeholder components for parts not yet implemented
    
    // Log window placeholder
    logWindow = std::make_unique<juce::Component>();
    logWindow->setName("Log/Status");
    addAndMakeVisible(*logWindow);
    
    // Waveform analysis placeholders
    originalWaveform = std::make_unique<juce::Component>();
    originalWaveform->setName("Original Waveform");
    addAndMakeVisible(*originalWaveform);
    
    reconstructedWaveform = std::make_unique<juce::Component>();
    reconstructedWaveform->setName("Reconstructed Waveform");
    addAndMakeVisible(*reconstructedWaveform);
    
    // Audio track placeholders
    jellieTrack = std::make_unique<juce::Component>();
    jellieTrack->setName("JELLIE Track");
    addAndMakeVisible(*jellieTrack);
    
    pnbtrTrack = std::make_unique<juce::Component>();
    pnbtrTrack->setName("PNBTR Track");
    addAndMakeVisible(*pnbtrTrack);
    
    // Controls panel placeholder
    controlsPanel = std::make_unique<juce::Component>();
    controlsPanel->setName("Controls Panel");
    addAndMakeVisible(*controlsPanel);
    
    // Set placeholder colors to distinguish sections
    auto setPlaceholderAppearance = [](juce::Component& comp, juce::Colour colour) {
        comp.setPaintingIsUnclipped(true);
        // Add a custom paint function via lambda (simplified approach)
    };
    
    // Apply different colors to distinguish the placeholder sections
    // These will be replaced with actual components later
}

void SchematicMainWindow::SchematicContent::drawRowSeparator(juce::Graphics& g, int y, int width)
{
    g.setColour(separatorColour);
    g.drawHorizontalLine(y, 0, width);
} 