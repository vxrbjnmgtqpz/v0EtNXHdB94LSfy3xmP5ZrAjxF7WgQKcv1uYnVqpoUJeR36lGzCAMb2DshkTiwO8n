#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>

class AudioThumbnailComponent : public juce::Component
{

public:
    AudioThumbnailComponent();
    ~AudioThumbnailComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    void loadFile(const juce::File& file);

    // New: Load waveform from a raw buffer (for live DSP data)
    void loadFromBuffer(const float* data, int numSamples, double sampleRate);

private:
    juce::AudioFormatManager formatManager;
    juce::AudioThumbnailCache thumbnailCache;
    juce::AudioThumbnail thumbnail;
    juce::File currentFile;
    std::vector<float> liveBuffer;
    double liveSampleRate = 44100.0;
    bool usingLiveBuffer = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioThumbnailComponent)
};
