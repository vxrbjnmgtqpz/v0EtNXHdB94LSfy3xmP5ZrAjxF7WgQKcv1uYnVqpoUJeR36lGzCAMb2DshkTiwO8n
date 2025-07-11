#include "WaveformAnalysisRow.h"
#include "../DSP/PNBTRTrainer.h"

void WaveformAnalysisRow::setTrainer(PNBTRTrainer* trainerPtr)
{
    inputOsc.setTrainer(trainerPtr);
    reconOsc.setTrainer(trainerPtr);
    trainer = trainerPtr;
    if (trainer)
        startTimerHz(2); // update thumbnail at 2 Hz - much more reasonable
    else
        stopTimer();
}

void WaveformAnalysisRow::timerCallback()
{
    updateThumbnail();
}

void WaveformAnalysisRow::updateThumbnail()
{
    if (!trainer) return;
    std::vector<float> inputBuf(bufferSize);
    trainer->getInputBuffer(inputBuf.data(), bufferSize);
    jellieTrack.loadFromBuffer(inputBuf.data(), bufferSize, sampleRate);
}

WaveformAnalysisRow::WaveformAnalysisRow()
    : inputOsc(OscilloscopeComponent::BufferType::AudioInput, "Input Analysis")
    , reconOsc(OscilloscopeComponent::BufferType::Reconstructed, "Reconstruction Analysis")
{
    addAndMakeVisible(inputOsc);
    addAndMakeVisible(reconOsc);
    addAndMakeVisible(jellieTrack);
}

WaveformAnalysisRow::~WaveformAnalysisRow() = default;

void WaveformAnalysisRow::resized()
{
    auto area = getLocalBounds().reduced(4);
    int oscHeight = area.getHeight() / 3;
    inputOsc.setBounds(area.removeFromTop(oscHeight).reduced(0, 2));
    reconOsc.setBounds(area.removeFromTop(oscHeight).reduced(0, 2));
    jellieTrack.setBounds(area.reduced(0, 2));
}

void WaveformAnalysisRow::paint(juce::Graphics& g)
{
    g.setColour(juce::Colours::white);
    g.setFont(13.0f);

    g.drawText("Original Waveform Oscilloscope\n(inputBuffer, real mic data)\nupdateOscilloscope(inputBuffer)",
               inputOsc.getBounds().reduced(6), juce::Justification::topLeft, true);

    g.drawText("Reconstructed Waveform Oscilloscope\n(reconstructedBuffer, after PNBTR)\nupdateOscilloscope(reconstructedBuffer)",
               reconOsc.getBounds().reduced(6), juce::Justification::topLeft, true);

    g.drawText("JELLIE Track (Recorded Input)\n[Live Audio Thumbnail]",
               jellieTrack.getBounds().reduced(6), juce::Justification::topLeft, true);
}
