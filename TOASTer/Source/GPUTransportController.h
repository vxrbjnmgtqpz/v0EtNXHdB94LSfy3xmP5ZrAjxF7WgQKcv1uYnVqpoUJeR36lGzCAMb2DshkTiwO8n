#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "gpu_native/gpu_timebase.h"
#include "gpu_native/gpu_shared_timeline.h"

// Forward declaration
class JAMNetworkPanel;

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
    
    // GUI components
    std::unique_ptr<juce::TextButton> playButton;
    std::unique_ptr<juce::TextButton> stopButton;
    std::unique_ptr<juce::TextButton> pauseButton;
    std::unique_ptr<juce::TextButton> recordButton;
    std::unique_ptr<juce::Label> positionLabel;
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
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUTransportController)
};
