#include "OscilloscopeRow.h"
#include "../DSP/PNBTRTrainer.h"

void OscilloscopeRow::setTrainer(PNBTRTrainer* trainerPtr)
{
    inputOsc.setTrainer(trainerPtr);
    // TOAST network oscilloscope doesn't use PNBTRTrainer directly - it uses JAM Framework
    outputOsc.setTrainer(trainerPtr);
}

OscilloscopeRow::OscilloscopeRow()
    : inputOsc(OscilloscopeComponent::BufferType::AudioInput, "Input")
    , outputOsc(OscilloscopeComponent::BufferType::Reconstructed, "Output")
{
    addAndMakeVisible(&inputOsc);
    addAndMakeVisible(&toastNetworkOsc);
    addAndMakeVisible(&logPanel);
    addAndMakeVisible(&outputOsc);
}

OscilloscopeRow::~OscilloscopeRow() = default;

void OscilloscopeRow::paint(juce::Graphics& g)
{
    g.setColour(juce::Colours::white);
    g.setFont(13.0f);

    g.drawText("(1) CoreAudio input callback (mic)\n→ (2) JELLIE encode (48kHz→192kHz, 8ch)",
               inputOsc.getBounds().reduced(4), juce::Justification::topLeft, true);

    g.drawText("(3) TOAST Protocol UDP Multicast\n→ (5) Real-time network metrics",
               toastNetworkOsc.getBounds().reduced(4), juce::Justification::topLeft, true);

    g.drawText("(6) Log events, errors, metrics",
               logPanel.getBounds().reduced(4), juce::Justification::topLeft, true);

    g.drawText("(4) PNBTR neural reconstruction (output buffer)",
               outputOsc.getBounds().reduced(4), juce::Justification::topLeft, true);
}

void OscilloscopeRow::resized()
{
    auto area = getLocalBounds();
    auto columnWidth = area.getWidth() / 4;

    inputOsc.setBounds(area.removeFromLeft(columnWidth));
    toastNetworkOsc.setBounds(area.removeFromLeft(columnWidth));
    logPanel.setBounds(area.removeFromLeft(columnWidth));
    outputOsc.setBounds(area);
}
