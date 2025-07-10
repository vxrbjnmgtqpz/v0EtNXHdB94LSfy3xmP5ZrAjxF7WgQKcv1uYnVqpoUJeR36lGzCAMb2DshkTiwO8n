#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "AudioThumbnailComponent.h"

class AudioTracksRow : public juce::Component, private juce::Timer
{
public:
    void setTrainer(class PNBTRTrainer* trainerPtr);
    AudioTracksRow();
    ~AudioTracksRow() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    AudioThumbnailComponent& getInputThumb()       { return jellieTrack; }
    AudioThumbnailComponent& getReconstructedThumb() { return pnptrTrack; }

private:
    void timerCallback() override;
    void updateThumbnails();

    AudioThumbnailComponent jellieTrack;
    AudioThumbnailComponent pnptrTrack;
    PNBTRTrainer* trainer = nullptr;
    int bufferSize = 2048; // or match your oscilloscope buffer size
    double sampleRate = 48000.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioTracksRow)
};
