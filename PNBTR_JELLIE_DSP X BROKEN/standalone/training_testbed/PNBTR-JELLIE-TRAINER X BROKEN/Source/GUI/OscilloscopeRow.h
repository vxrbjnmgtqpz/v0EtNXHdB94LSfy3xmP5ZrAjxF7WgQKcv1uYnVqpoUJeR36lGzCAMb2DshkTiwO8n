#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "OscilloscopeComponent.h"
#include "LogStatusComponent.h"
#include "../DSP/PNBTRTrainer.h"

class OscilloscopeRow : public juce::Component
{
public:
    void setTrainer(PNBTRTrainer* trainerPtr);
    OscilloscopeRow();
    ~OscilloscopeRow() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    OscilloscopeComponent& getInputOsc()       { return inputOsc; }
    OscilloscopeComponent& getNetworkOsc()     { return networkOsc; }
    OscilloscopeComponent& getOutputOsc()      { return outputOsc; }
    LogStatusComponent& getLogStatus()         { return logPanel; }

private:
    OscilloscopeComponent inputOsc;
    OscilloscopeComponent networkOsc;
    LogStatusComponent logPanel;
    OscilloscopeComponent outputOsc;
    PNBTRTrainer* trainer = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OscilloscopeRow)
};
