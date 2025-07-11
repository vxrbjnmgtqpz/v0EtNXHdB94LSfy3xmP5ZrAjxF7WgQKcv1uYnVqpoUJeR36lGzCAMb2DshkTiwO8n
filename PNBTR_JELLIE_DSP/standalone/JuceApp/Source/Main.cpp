/*
  ==============================================================================

    Main.cpp
    Created: 9 Jul 2024 10:00:00pm
    Author:  Gemini

  ==============================================================================
*/

#include <JuceHeader.h>
#include "RingBuffer.h"
#include "MetalBridge.h"

class MainComponent : public juce::AudioAppComponent, public juce::Thread
{
public:
    MainComponent() : Thread("GPU Processing Thread")
    {
        setSize(800, 600);
        
        setAudioChannels(2, 2);

        inputRingBuffer = std::make_unique<RingBuffer<float>>(4096);
        outputRingBuffer = std::make_unique<RingBuffer<float>>(4096);
        metalBridge = std::make_unique<MetalBridge>();
        metalBridge->init();
        
        startThread();
    }

    ~MainComponent() override
    {
        stopThread(1000);
        shutdownAudio();
    }

    void run() override
    {
        while (!threadShouldExit())
        {
            // In a real app, you'd use a more sophisticated way to wait for data
            if (inputRingBuffer->size() >= 512)
            {
                std::vector<float> inputBlock(512);
                inputRingBuffer->read(inputBlock.data(), 512);

                std::vector<float> outputBlock(512);
                metalBridge->process(inputBlock, outputBlock);
                
                outputRingBuffer->write(outputBlock.data(), 512);
            }
            wait(1); // Wait a bit
        }
    }

    void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override
    {
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override
    {
        auto* device = deviceManager.getCurrentAudioDevice();
        auto activeInputChannels = device->getActiveInputChannels();
        auto activeOutputChannels = device->getActiveOutputChannels();
        auto maxInputChannels = activeInputChannels.countNumberOfSetBits();
        auto maxOutputChannels = activeOutputChannels.countNumberOfSetBits();

        // Write input to ring buffer
        for (int channel = 0; channel < maxInputChannels; ++channel)
        {
            if (activeInputChannels[channel])
            {
                inputRingBuffer->write(bufferToFill.buffer->getReadPointer(channel), bufferToFill.numSamples);
            }
        }
        
        // Read from output ring buffer
        if (outputRingBuffer->size() >= bufferToFill.numSamples)
        {
            for (int channel = 0; channel < maxOutputChannels; ++channel)
            {
                 if (activeOutputChannels[channel])
                 {
                    outputRingBuffer->read(bufferToFill.buffer->getWritePointer(channel), bufferToFill.numSamples);
                 }
            }
        }
        else 
        {
            // Not enough data, clear buffer
            bufferToFill.clearActiveBufferRegion();
        }
    }

    void releaseResources() override
    {
    }


    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);
    }

private:
    std::unique_ptr<RingBuffer<float>> inputRingBuffer;
    std::unique_ptr<RingBuffer<float>> outputRingBuffer;
    std::unique_ptr<MetalBridge> metalBridge;
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