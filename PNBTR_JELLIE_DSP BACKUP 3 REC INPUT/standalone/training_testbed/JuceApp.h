#pragma once
#include <JuceHeader.h>

class TrainingTestbedJuceApp : public juce::JUCEApplication
{
public:
    TrainingTestbedJuceApp() = default;
    const juce::String getApplicationName() override { return "PNBTR+JELLIE Training Testbed"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }
    void initialise(const juce::String&) override;
    void shutdown() override;
private:
    std::unique_ptr<juce::DocumentWindow> mainWindow;
};
