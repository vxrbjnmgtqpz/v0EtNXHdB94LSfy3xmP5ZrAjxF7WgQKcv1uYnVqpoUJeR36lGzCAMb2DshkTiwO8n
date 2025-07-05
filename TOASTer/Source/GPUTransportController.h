#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../../JAM_Framework_v2/include/gpu_native/gpu_timebase.h"
#include "../../JAM_Framework_v2/include/gpu_native/gpu_shared_timeline.h"
#include "../../JAM_Framework_v2/include/gpu_transport/gpu_transport_manager.h"

// Forward declaration
class JAMNetworkPanel;

//==============================================================================
/**
 * Custom transport button that renders shapes via canvas instead of relying on emoji fonts
 */
class GPUTransportButton : public juce::Button
{
public:
    enum ButtonType { Play, Stop, Pause, Record };
    
    GPUTransportButton(const juce::String& name, ButtonType type) 
        : juce::Button(name), buttonType(type) {}
    
    void paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override
    {
        auto bounds = getLocalBounds().reduced(2).toFloat();
        
        // Background
        juce::Colour bgColour = juce::Colours::darkgrey;
        if (buttonType == Play) bgColour = juce::Colours::green;
        else if (buttonType == Stop) bgColour = juce::Colours::red;
        else if (buttonType == Pause) bgColour = juce::Colours::orange;
        else if (buttonType == Record) bgColour = juce::Colours::red;
        
        if (shouldDrawButtonAsDown)
            bgColour = bgColour.brighter(0.3f);
        else if (shouldDrawButtonAsHighlighted)
            bgColour = bgColour.brighter(0.1f);
            
        g.setColour(bgColour.withAlpha(0.3f));
        g.fillRoundedRectangle(bounds, 4.0f);
        
        g.setColour(bgColour);
        g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
        
        // Draw shape
        auto center = bounds.getCentre();
        auto size = juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.4f;
        
        g.setColour(juce::Colours::white);
        
        if (buttonType == Play)
        {
            // Triangle pointing right
            juce::Path triangle;
            triangle.addTriangle(center.x - size*0.3f, center.y - size*0.5f,
                               center.x - size*0.3f, center.y + size*0.5f,
                               center.x + size*0.5f, center.y);
            g.fillPath(triangle);
        }
        else if (buttonType == Stop)
        {
            // Square
            auto square = juce::Rectangle<float>(center.x - size*0.4f, center.y - size*0.4f, 
                                               size*0.8f, size*0.8f);
            g.fillRect(square);
        }
        else if (buttonType == Pause)
        {
            // Two vertical bars
            auto bar1 = juce::Rectangle<float>(center.x - size*0.3f, center.y - size*0.4f, 
                                             size*0.2f, size*0.8f);
            auto bar2 = juce::Rectangle<float>(center.x + size*0.1f, center.y - size*0.4f, 
                                             size*0.2f, size*0.8f);
            g.fillRect(bar1);
            g.fillRect(bar2);
        }
        else if (buttonType == Record)
        {
            // Circle
            g.fillEllipse(center.x - size*0.4f, center.y - size*0.4f, size*0.8f, size*0.8f);
        }
        
        // Draw text below shape
        auto textBounds = bounds.withTop(center.y + size*0.6f);
        g.setColour(juce::Colours::white);
        g.setFont(juce::Font(juce::FontOptions(12.0f)));
        g.drawText(getButtonText(), textBounds, juce::Justification::centred);
    }
    
private:
    ButtonType buttonType;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUTransportButton)
};

//==============================================================================
/**
 * GPU-Native Transport Controller for TOASTer
 * 
 * Replaces the legacy CPU-based TransportController with GPU-native timing.
 * All transport operations (play/stop/pause/record) are synchronized with
 * the GPU timebase for sub-microsecond precision.
 */
class GPUTransportController : public juce::Component, public juce::Timer
{
public:
    GPUTransportController();
    ~GPUTransportController() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

    // GPU-native transport controls
    void play();
    void stop();
    void pause();
    void record();
    void seek(uint64_t gpuFrame);
    
    // GPU timeline queries
    bool isPlaying() const;
    bool isRecording() const;
    bool isPaused() const;
    uint64_t getCurrentGPUFrame() const;
    double getCurrentTimeInSeconds() const;
    
    // Network sync integration
    void setNetworkPanel(JAMNetworkPanel* panel) { networkPanel = panel; }
    void handleRemoteTransportCommand(const std::string& command, uint64_t gpuTimestamp);
    
    // BPM and tempo (GPU-synchronized)
    void setBPM(double bpm);
    double getBPM() const { return currentBPM; }

private:
    // GPU-native infrastructure (now static, no instances needed)
    
    // Transport state (GPU-synchronized)
    enum class GPUTransportState {
        Stopped,
        Playing,
        Paused,
        Recording
    };
    
    GPUTransportState currentState = GPUTransportState::Stopped;
    uint64_t playStartFrame = 0;
    uint64_t pausedFrame = 0;
    double currentBPM = 120.0;
    
    // Network sync
    JAMNetworkPanel* networkPanel = nullptr;
    
    // Timing parameters for bars/beats
    int beatsPerBar = 4;        // Time signature numerator
    int beatUnit = 4;           // Time signature denominator (4 = quarter note)
    int subdivision = 4;        // Subdivisions per beat (for fine display)
    
    // GUI components
    std::unique_ptr<GPUTransportButton> playButton;
    std::unique_ptr<GPUTransportButton> stopButton;
    std::unique_ptr<GPUTransportButton> pauseButton;
    std::unique_ptr<GPUTransportButton> recordButton;
    std::unique_ptr<juce::Label> positionLabel;
    std::unique_ptr<juce::Label> barsBeatsLabel;  // Added bars/beats display
    std::unique_ptr<juce::Label> bpmLabel;
    std::unique_ptr<juce::Slider> bpmSlider;
    
    // Button callbacks
    void playButtonClicked();
    void stopButtonClicked();
    void pauseButtonClicked();
    void recordButtonClicked();
    void bpmSliderChanged();
    
    // Send transport commands over network
    void sendTransportCommand(const std::string& command);
    
    // Update position display
    void updatePositionDisplay();
    void updateBarsBeatsDisplay();  // Added bars/beats update method
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUTransportController)
};
