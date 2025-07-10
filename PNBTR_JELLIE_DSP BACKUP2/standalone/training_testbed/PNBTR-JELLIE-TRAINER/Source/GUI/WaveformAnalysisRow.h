#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "OscilloscopeComponent.h"
#include "AudioThumbnailComponent.h"
#include "../DSP/PNBTRTrainer.h"

class WaveformAnalysisRow : public juce::Component, private juce::Timer
{
public:
    void setTrainer(PNBTRTrainer* trainerPtr);
    WaveformAnalysisRow();
    ~WaveformAnalysisRow() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    OscilloscopeComponent& getInputScope()       { return inputOsc; }
    OscilloscopeComponent& getReconstructedScope() { return reconOsc; }
    AudioThumbnailComponent& getInputThumb()       { return jellieTrack; }

private:
    void timerCallback() override;
    void updateThumbnail();

    OscilloscopeComponent inputOsc;
    OscilloscopeComponent reconOsc;
    AudioThumbnailComponent jellieTrack;
    PNBTRTrainer* trainer = nullptr;
    int bufferSize = 2048;
    double sampleRate = 48000.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WaveformAnalysisRow)
};
