#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <chrono>
#include <atomic>
#include <memory>
#include <functional>

// Forward declarations
class PNBTRTrainer;
class MainComponent;

//==============================================================================
/**
 * Custom transport button that renders shapes via canvas instead of relying on emoji fonts
 */
class TransportButton : public juce::Button
{
public:
    enum ButtonType { Play, Stop, Pause, Record };
    
    TransportButton(const juce::String& name, ButtonType type) 
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
            // Draw triangle (play)
            juce::Path triangle;
            triangle.addTriangle(center.x - size * 0.3f, center.y - size * 0.5f,
                               center.x - size * 0.3f, center.y + size * 0.5f,
                               center.x + size * 0.5f, center.y);
            g.fillPath(triangle);
        }
        else if (buttonType == Stop)
        {
            // Draw square (stop)
            g.fillRect(center.x - size * 0.4f, center.y - size * 0.4f, size * 0.8f, size * 0.8f);
        }
        else if (buttonType == Pause)
        {
            // Draw two vertical bars (pause)
            float barWidth = size * 0.25f;
            g.fillRect(center.x - size * 0.3f, center.y - size * 0.4f, barWidth, size * 0.8f);
            g.fillRect(center.x + size * 0.05f, center.y - size * 0.4f, barWidth, size * 0.8f);
        }
        else if (buttonType == Record)
        {
            // Draw circle (record)
            g.fillEllipse(center.x - size * 0.4f, center.y - size * 0.4f, size * 0.8f, size * 0.8f);
        }
    }

private:
    ButtonType buttonType;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TransportButton)
};

//==============================================================================
/**
 * Professional Transport Controller with microsecond precision timing
 * All transport operations are synchronized with high-resolution timing
 * for sub-microsecond precision.
 */
class ProfessionalTransportController : public juce::Component, public juce::Timer
{
    // Add sliders for Packet Loss [%] and Jitter [ms]
    std::unique_ptr<juce::Label> packetLossLabel;
    std::unique_ptr<juce::Slider> packetLossSlider;
    std::unique_ptr<juce::Label> jitterLabel;
    std::unique_ptr<juce::Slider> jitterSlider;
    void packetLossSliderChanged();
    void jitterSliderChanged();
public:
    // Callbacks from MainComponent
    std::function<void()> onPlay;
    std::function<void()> onStop; 
    std::function<void()> onRecord;
    
    // 🔧 NEW: Debug logging callback
    std::function<void(const juce::String&)> onLogMessage;
    ProfessionalTransportController();
    ~ProfessionalTransportController() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

    // Transport controls
    void play();
    void stop();
    void pause();
    void record();
    void seek(double timeInSeconds);
    
    // Timeline queries
    bool isPlaying() const;
    bool isRecording() const;
    bool isPaused() const;
    double getCurrentTimeInSeconds() const;
    uint64_t getCurrentSample() const;
    
    // BPM and tempo
    void setBPM(double bpm);
    double getBPM() const { return currentBPM.load(); }

public:
    // Set the trainer instance for DSP control
    void setTrainer(class PNBTRTrainer* trainer) { pnbtrTrainer = trainer; }

private:
    // Pointer to the DSP trainer (not owned)
    class PNBTRTrainer* pnbtrTrainer = nullptr;
    // Transport state
    enum class TransportState {
        Stopped,
        Playing,
        Paused,
        Recording
    };
    
    std::atomic<TransportState> currentState{TransportState::Stopped};
    std::chrono::steady_clock::time_point playStartTime;
    std::chrono::steady_clock::time_point pausedTime;
    std::atomic<double> currentBPM{120.0};
    std::atomic<uint64_t> currentSample{0};
    
    // Timing parameters
    static constexpr double sampleRate = 44100.0;
    std::atomic<int> beatsPerBar{4};        // Time signature numerator
    std::atomic<int> beatUnit{4};           // Time signature denominator
    std::atomic<int> subdivision{4};        // Subdivisions per beat
    
    // GUI components  
    std::unique_ptr<juce::TextButton> playButton;
    std::unique_ptr<juce::TextButton> stopButton;
    std::unique_ptr<juce::TextButton> pauseButton;
    std::unique_ptr<juce::TextButton> recordButton;
    std::unique_ptr<juce::Label> sessionTimeLabel;    // SESSION TIME: 00:00:00.000.000
    std::unique_ptr<juce::Label> barsBeatsLabel;      // BARS: 1.1.1
    std::unique_ptr<juce::Label> bpmLabel;
    std::unique_ptr<juce::Slider> bpmSlider;
    
    // Button callbacks
    void playButtonClicked();
    void stopButtonClicked();
    void pauseButtonClicked();
    void recordButtonClicked();
    void bpmSliderChanged();
    
    // Update displays
    void updateSessionTimeDisplay();
    void updateBarsBeatsDisplay();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProfessionalTransportController)
};
