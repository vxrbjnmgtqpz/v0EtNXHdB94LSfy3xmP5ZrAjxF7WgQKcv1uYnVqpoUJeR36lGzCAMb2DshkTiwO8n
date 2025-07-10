/*
  ==============================================================================

    Main.cpp
    Created: 9 Jul 2024 10:00:00pm
    Author:  Gemini

  ==============================================================================
*/

#include <JuceHeader.h>

class MainComponent : public juce::Component
{
public:
    MainComponent()
    {
        setSize(800, 600);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);
    }
};


class PnbtrJellieJuceApp : public juce::JUCEApplication
{
public:
    PnbtrJellieJuceApp() {}

    const juce::String getApplicationName() override { return "PnbtrJellieJuceApp"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }

    void initialise(const juce::String&) override
    {
        mainWindow.reset(new MainWindow(getApplicationName(), new MainComponent, this));
    }

    void shutdown() override
    {
        mainWindow.reset();
    }

private:
    class MainWindow : public juce::DocumentWindow
    {
    public:
        MainWindow(const juce::String& name, juce::Component* component, juce::JUCEApplication* app)
            : DocumentWindow(name, juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(ResizableWindow::backgroundColourId), allButtons),
            app(app)
        {
            setUsingNativeTitleBar(true);
            setContentOwned(component, true);
            setResizable(true, true);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }

        void closeButtonPressed() override
        {
            app->systemRequestedQuit();
        }

    private:
        juce::JUCEApplication* app;
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
    };

    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION(PnbtrJellieJuceApp) 