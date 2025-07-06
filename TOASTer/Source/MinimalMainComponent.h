#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
class MinimalMainComponent : public juce::Component
{
public:
    MinimalMainComponent()
    {
        setSize(400, 300);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::darkgrey);
        g.setColour(juce::Colours::white);
        g.setFont(20.0f);
        g.drawText("TOASTer Test - Minimal Version", getLocalBounds(), juce::Justification::centred, true);
    }

    void resized() override
    {
        // Nothing to resize in minimal version
    }

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MinimalMainComponent)
};
