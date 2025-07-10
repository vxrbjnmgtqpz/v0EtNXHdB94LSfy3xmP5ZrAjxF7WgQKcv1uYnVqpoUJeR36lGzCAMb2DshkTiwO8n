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

    // Check if data actually changed to avoid expensive thumbnail recalculation
    std::vector<float> newBuffer(data, data + numSamples);
    if (newBuffer == liveBuffer && sampleRate == liveSampleRate) {
        return; // No change, skip expensive thumbnail rebuild
    }

    liveBuffer = std::move(newBuffer);
    liveSampleRate = sampleRate;
    usingLiveBuffer = true;

    // Create a temporary AudioBuffer for thumbnail
    juce::AudioBuffer<float> tempBuffer(1, numSamples);
    tempBuffer.copyFrom(0, 0, data, numSamples);
    thumbnail.reset(sampleRate, numSamples);
    thumbnail.addBlock(0, tempBuffer, 0, numSamples);
    repaint();
}
