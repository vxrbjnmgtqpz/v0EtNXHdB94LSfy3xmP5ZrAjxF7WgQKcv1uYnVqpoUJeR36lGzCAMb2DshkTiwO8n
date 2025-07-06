#include "MainComponent.h"
#include "BasicMIDIPanel.h"
#include "BasicNetworkPanel.h"
#include "BasicTransportPanel.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Create simplified panels without heavy framework dependencies
    transportPanel = new BasicTransportPanel();
    midiPanel = new BasicMIDIPanel();
    networkPanel = new BasicNetworkPanel();
    
    addAndMakeVisible(transportPanel);
    addAndMakeVisible(midiPanel);
    addAndMakeVisible(networkPanel);
    
    setSize(800, 600);
    
    // Start UI update timer
    startTimer(50); // 20 FPS for UI updates
}

MainComponent::~MainComponent()
{
    stopTimer();
    delete transportPanel;
    delete midiPanel;
    delete networkPanel;
}

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff2a2a2a));
        
        g.setColour(juce::Colour(0xff4a4a4a));
        g.drawRect(getLocalBounds(), 2);
        
        g.setColour(juce::Colours::white);
        g.setFont(24.0f);
        g.drawText("TOASTer - MIDI/Audio Testing Tool", 
                   getLocalBounds().removeFromTop(50), 
                   juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(50); // Title area
        
        auto panelHeight = bounds.getHeight() / 3;
        
        transportPanel->setBounds(bounds.removeFromTop(panelHeight).reduced(5));
        midiPanel->setBounds(bounds.removeFromTop(panelHeight).reduced(5));
        networkPanel->setBounds(bounds.reduced(5));
    }
        networkPanel.setBounds(bounds.reduced(5));
    }

private:
    void timerCallback() override
    {
        // Simple periodic updates
        repaint();
    }

    BasicTransportPanel transportPanel;
    BasicMIDIPanel midiPanel;
    BasicNetworkPanel networkPanel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};

//==============================================================================
class TOASTerApplication : public juce::JUCEApplication
{
public:
    TOASTerApplication() = default;

    const juce::String getApplicationName() override { return "TOASTer"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }
    bool moreThanOneInstanceAllowed() override { return true; }

    void initialise(const juce::String&) override
    {
        mainWindow.reset(new MainWindow(getApplicationName()));
    }

    void shutdown() override
    {
        mainWindow = nullptr;
    }

    void systemRequestedQuit() override
    {
        quit();
    }

    class MainWindow : public juce::DocumentWindow
    {
    public:
        MainWindow(juce::String name)
            : DocumentWindow(name,
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
             centreWithSize(getWidth(), getHeight());
            #endif

            setVisible(true);
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
START_JUCE_APPLICATION(TOASTerApplication)
