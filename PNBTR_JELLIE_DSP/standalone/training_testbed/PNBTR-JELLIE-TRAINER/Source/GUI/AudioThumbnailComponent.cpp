#include "AudioThumbnailComponent.h"

AudioThumbnailComponent::AudioThumbnailComponent()
    : thumbnailCache(5), // cache size: number of thumbnails
      thumbnail(512, formatManager, thumbnailCache)
{
    formatManager.registerBasicFormats();
}

AudioThumbnailComponent::~AudioThumbnailComponent() = default;

void AudioThumbnailComponent::resized() {}

void AudioThumbnailComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds());

    if (usingLiveBuffer && liveBuffer.size() > 0)
    {
        g.setColour(juce::Colours::orange);
        thumbnail.drawChannels(g, getLocalBounds().reduced(4), 0.0, thumbnail.getTotalLength(), 1.0f);
    }
    else if (thumbnail.getTotalLength() > 0.0)
    {
        g.setColour(juce::Colours::limegreen);
        thumbnail.drawChannels(g, getLocalBounds().reduced(4), 0.0, thumbnail.getTotalLength(), 1.0f);
    }
    else
    {
        g.setColour(juce::Colours::grey);
        g.setFont(13.0f);
        g.drawText("No audio loaded", getLocalBounds(), juce::Justification::centred);
    }
}

void AudioThumbnailComponent::loadFile(const juce::File& file)
{
    currentFile = file;
    thumbnail.setSource(new juce::FileInputSource(file));
    usingLiveBuffer = false;
    repaint();
}

void AudioThumbnailComponent::loadFromBuffer(const float* data, int numSamples, double sampleRate)
{
    if (numSamples <= 0 || data == nullptr)
        return;

    // LIVE AUDIO FIX: Always update for streaming data, no comparison check
    // The expensive comparison was preventing live audio updates
    
    liveBuffer.assign(data, data + numSamples);
    liveSampleRate = sampleRate;
    usingLiveBuffer = true;

    // Create a temporary AudioBuffer for thumbnail
    juce::AudioBuffer<float> tempBuffer(1, numSamples);
    tempBuffer.copyFrom(0, 0, data, numSamples);
    
    // Reset and rebuild thumbnail with new data
    thumbnail.reset(sampleRate, numSamples);
    thumbnail.addBlock(0, tempBuffer, 0, numSamples);
    
    // DEBUG: Check if we're getting meaningful audio data
    float maxLevel = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        maxLevel = std::max(maxLevel, std::abs(data[i]));
    }
    
    static int debugCount = 0;
    if (++debugCount % 60 == 0) { // Every ~30 seconds at 2Hz
        printf("[AUDIO THUMBNAIL] Samples: %d, Max Level: %.4f, Total Length: %.2fs\n", 
               numSamples, maxLevel, thumbnail.getTotalLength());
    }
    
    repaint();
}
