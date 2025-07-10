#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class TitleComponent : public juce::Component
{
public:
    TitleComponent();
    ~TitleComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    juce::Label titleLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TitleComponent)
};
