#include "TitleComponent.h"

TitleComponent::TitleComponent()
{
    titleLabel.setText("PNBTR+JELLIE Training Testbed", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(18.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(titleLabel);
}

TitleComponent::~TitleComponent() = default;

void TitleComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey);
    g.setColour(juce::Colours::lightgrey);
    g.drawRect(getLocalBounds(), 1);
}

void TitleComponent::resized()
{
    titleLabel.setBounds(getLocalBounds());
}
