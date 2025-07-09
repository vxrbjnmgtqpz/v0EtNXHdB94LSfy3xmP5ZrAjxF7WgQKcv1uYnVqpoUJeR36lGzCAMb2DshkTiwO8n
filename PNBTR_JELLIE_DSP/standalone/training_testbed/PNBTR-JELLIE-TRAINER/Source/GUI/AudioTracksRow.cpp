#include "../DSP/PNBTRTrainer.h"
#include "AudioTracksRow.h"

AudioTracksRow::AudioTracksRow()
{
    addAndMakeVisible(jellieTrack);
    addAndMakeVisible(pnptrTrack);
}

AudioTracksRow::~AudioTracksRow() = default;

void AudioTracksRow::setTrainer(PNBTRTrainer* trainerPtr)
{
    trainer = trainerPtr;
    if (trainer)
        startTimerHz(15); // update thumbnails at 15 Hz
    else
        stopTimer();
}

void AudioTracksRow::timerCallback()
{
    updateThumbnails();
}

void AudioTracksRow::updateThumbnails()
{
    if (!trainer) return;
    std::vector<float> inputBuf(bufferSize);
    std::vector<float> outputBuf(bufferSize);
    trainer->getInputBuffer(inputBuf.data(), bufferSize);
    trainer->getOutputBuffer(outputBuf.data(), bufferSize);
    // Optionally, get actual sample rate from trainer if available
    jellieTrack.loadFromBuffer(inputBuf.data(), bufferSize, sampleRate);
    pnptrTrack.loadFromBuffer(outputBuf.data(), bufferSize, sampleRate);
}

void AudioTracksRow::resized()
{
    auto area = getLocalBounds().reduced(4);
    int trackHeight = area.getHeight() / 2;
    jellieTrack.setBounds(area.removeFromTop(trackHeight).reduced(0, 2));
    pnptrTrack.setBounds(area.reduced(0, 2));
}

void AudioTracksRow::paint(juce::Graphics& g)
{
    g.setColour(juce::Colours::white);
    g.setFont(12.0f);

    g.drawText("JUCE::AudioThumbnail: JELLIE Track\n(recorded input, .wav)",
               jellieTrack.getBounds().reduced(6), juce::Justification::topLeft, true);

    g.drawText("JUCE::AudioThumbnail: PNBTR Track\n(reconstructed output, .wav)",
               pnptrTrack.getBounds().reduced(6), juce::Justification::topLeft, true);

    g.setFont(11.0f);
    g.setColour(juce::Colours::grey);
    g.drawText("[PNBTR_JELLIE_DSP/standalone/juce/]", getLocalBounds().withTop(getBottom() - 15), juce::Justification::centredBottom, false);
}
