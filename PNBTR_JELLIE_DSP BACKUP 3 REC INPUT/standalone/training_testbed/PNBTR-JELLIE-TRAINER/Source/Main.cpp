/*
  ==============================================================================

    Main.cpp
    Created: Main application entry point for PNBTR+JELLIE Training Testbed

    GPU-native training harness with Metal compute shaders

  ==============================================================================
*/

#include <juce_gui_basics/juce_gui_basics.h>
#include "GUI/MainComponent.h"

//==============================================================================
class PnbtrJellieTrainerApplication : public juce::JUCEApplication
{
public:
    PnbtrJellieTrainerApplication() = default;

    const juce::String getApplicationName() override { return "PNBTR+JELLIE Training Testbed"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }
    bool moreThanOneInstanceAllowed() override { return true; }

    //==============================================================================
    void initialise(const juce::String& commandLine) override
    {
        mainWindow.reset(new MainWindow(getApplicationName()));
    }

    void shutdown() override
    {
        mainWindow.reset();
    }

    //==============================================================================
    void systemRequestedQuit() override
    {
        quit();
    }

    void anotherInstanceStarted(const juce::String& commandLine) override
    {
        // Handle multiple instances if needed
    }

    //==============================================================================
    class MainWindow : public juce::DocumentWindow
    {
    public:
        MainWindow(juce::String name) : DocumentWindow(name,
                                                      juce::Desktop::getInstance().getDefaultLookAndFeel()
                                                          .findColour(juce::ResizableWindow::backgroundColourId),
                                                      DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar(true);
            setContentOwned(new MainComponent(), true);

            #if JUCE_IOS || JUCE_ANDROID
                setFullScreen(true);
            #else
                setResizable(true, true);
                // Force a reasonable size and position
                setBounds(100, 100, 1400, 620);
                centreWithSize(1400, 620);
            #endif

            setVisible(true);
            toFront(true);
        }

        void closeButtonPressed() override
        {
            JUCEApplication::getInstance()->systemRequestedQuit();
        }

    private:
        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
    };

private:
    std::unique_ptr<MainWindow> mainWindow;
};

//==============================================================================
START_JUCE_APPLICATION(PnbtrJellieTrainerApplication) 