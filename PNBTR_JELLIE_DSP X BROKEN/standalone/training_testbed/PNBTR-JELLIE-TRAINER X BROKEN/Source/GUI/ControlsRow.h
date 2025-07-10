#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>

// Forward declarations
class PNBTRTrainer;

//==============================================================================
/**
 * ControlsRow Component
 * Row 5 in the schematic: Network parameter sliders + export functionality
 * Transport controls are handled by ProfessionalTransportController
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
    
    // GPU-safe parameter updates
    void updateNetworkParameters(float packetLoss, float jitter, float gain);

private:
    // Export functionality (utility function, not transport control)
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
    void exportSession();
    void onPacketLossChanged();
    void onJitterChanged();
    void onGainChanged();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlsRow)
};
