#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>

// Forward declarations
class PNBTRTrainer;

//==============================================================================
/**
 * ControlsRow Component
 * Row 5 in the schematic: Transport controls + network parameter sliders
 * Leverages GPU-aware transport controls from TOASTer
 */
class ControlsRow : public juce::Component
{
public:
    ControlsRow();
    ~ControlsRow() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    
    // Connect to PNBTRTrainer for control callbacks
    void setTrainer(PNBTRTrainer* trainer);
    
    // GPU-safe control updates
    void updateTransportState(bool isPlaying, bool isRecording);
    void updateNetworkParameters(float packetLoss, float jitter, float gain);

private:
    // Transport controls (GPU-aware, borrowed from TOASTer pattern)
    std::unique_ptr<juce::TextButton> startButton;
    std::unique_ptr<juce::TextButton> stopButton;
    std::unique_ptr<juce::TextButton> exportButton;
    
    // Network parameter sliders (write to GPU config atomically)
    std::unique_ptr<juce::Slider> packetLossSlider;
    std::unique_ptr<juce::Slider> jitterSlider;
    std::unique_ptr<juce::Slider> gainSlider;
    
    // Labels
    std::unique_ptr<juce::Label> packetLossLabel;
    std::unique_ptr<juce::Label> jitterLabel;
    std::unique_ptr<juce::Label> gainLabel;
    
    // Session manager reference
    PNBTRTrainer* trainer = nullptr;
    
    // Control callbacks
    void startProcessing();
    void stopProcessing();
    void exportSession();
    void onPacketLossChanged();
    void onJitterChanged();
    void onGainChanged();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlsRow)
};
