#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>
#include "OscilloscopeComponent.h"
#include "MetricsDashboard.h"

// Forward declarations
class SessionManager;

/**
 * Main window implementing the exact layout from Schematic.md
 * Does NOT replace the existing MainComponent - this is a NEW schematic-based GUI
 * 
 * Layout (matching Schematic.md):
 * Row 1: Input/Network/Log/Output Oscilloscopes  
 * Row 2: Waveform Analysis (Original vs Reconstructed)
 * Row 3: JUCE Audio Tracks (JELLIE & PNBTR)
 * Row 4: Metrics Dashboard (SNR/THD/Latency/etc)
 * Row 5: Controls (Start/Stop/Export/Sliders)
 */
class SchematicMainWindow : public juce::DocumentWindow
{
public:
    SchematicMainWindow();
    ~SchematicMainWindow() override;
    
    //==============================================================================
    // DocumentWindow interface
    void closeButtonPressed() override;
    void resized() override;
    
    //==============================================================================
    // Schematic layout management
    void updateLayout();
    void showWindow();
    void hideWindow();
    
private:
    //==============================================================================
    // Main content component
    class SchematicContent : public juce::Component
    {
    public:
        SchematicContent();
        ~SchematicContent() override;
        
        void paint(juce::Graphics& g) override;
        void resized() override;
        
    private:
        //==============================================================================
        // Row 1: Oscilloscopes (4 across)
        std::unique_ptr<OscilloscopeComponent> inputOscilloscope;
        std::unique_ptr<OscilloscopeComponent> networkOscilloscope;
        std::unique_ptr<juce::Component> logWindow; // Placeholder for now
        std::unique_ptr<OscilloscopeComponent> outputOscilloscope;
        
        //==============================================================================
        // Row 2: Waveform Analysis
        std::unique_ptr<juce::Component> originalWaveform; // Placeholder
        std::unique_ptr<juce::Component> reconstructedWaveform; // Placeholder
        
        //==============================================================================
        // Row 3: JUCE Audio Tracks
        std::unique_ptr<juce::Component> jellieTrack; // Placeholder
        std::unique_ptr<juce::Component> pnbtrTrack; // Placeholder
        
        //==============================================================================
        // Row 4: Metrics Dashboard
        std::unique_ptr<MetricsDashboard> metricsDashboard;
        
        //==============================================================================
        // Row 5: Controls
        std::unique_ptr<juce::Component> controlsPanel; // Placeholder
        
        //==============================================================================
        // Layout parameters (matching schematic proportions)
        struct LayoutConfig {
            int oscilloscopeRowHeight = 200;
            int waveformRowHeight = 150;
            int audioTracksRowHeight = 100;
            int metricsRowHeight = 120;
            int controlsRowHeight = 80;
            int margin = 10;
            int componentSpacing = 5;
        } layout;
        
        //==============================================================================
        // Visual styling
        juce::Colour backgroundColour{0xff1e1e1e};
        juce::Colour separatorColour{0xff444444};
        
        //==============================================================================
        // Internal methods
        void createPlaceholderComponents();
        void layoutOscilloscopeRow(juce::Rectangle<int>& bounds);
        void layoutWaveformRow(juce::Rectangle<int>& bounds);
        void layoutAudioTracksRow(juce::Rectangle<int>& bounds);
        void layoutMetricsRow(juce::Rectangle<int>& bounds);
        void layoutControlsRow(juce::Rectangle<int>& bounds);
        void drawRowSeparator(juce::Graphics& g, int y, int width);
        
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SchematicContent)
    };
    
    //==============================================================================
    // Window content
    std::unique_ptr<SchematicContent> content;
    
    //==============================================================================
    // Session integration
    std::unique_ptr<SessionManager> sessionManager;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SchematicMainWindow)
}; 