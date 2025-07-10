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
        startTimerHz(2); // update thumbnails at 2 Hz - much more reasonable for thumbnails
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
    
    // Use the correct thread-safe methods from PNBTRTrainer
    std::vector<float> inputBuf(bufferSize * 2); // Stereo data
    std::vector<float> outputBuf(bufferSize * 2); // Stereo data
    
    // Get live oscilloscope input (microphone data)
    trainer->getLatestOscInput(inputBuf.data(), bufferSize);
    
    // Get live oscilloscope output (reconstructed data)  
    trainer->getLatestOscOutput(outputBuf.data(), bufferSize);
    
    // Load mono data into thumbnails (take left channel)
    std::vector<float> monoInput(bufferSize);
    std::vector<float> monoOutput(bufferSize);
    
    for (int i = 0; i < bufferSize; ++i) {
        monoInput[i] = inputBuf[i * 2]; // Left channel
        monoOutput[i] = outputBuf[i * 2]; // Left channel
    }
    
    jellieTrack.loadFromBuffer(monoInput.data(), bufferSize, sampleRate);
    pnptrTrack.loadFromBuffer(monoOutput.data(), bufferSize, sampleRate);
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
