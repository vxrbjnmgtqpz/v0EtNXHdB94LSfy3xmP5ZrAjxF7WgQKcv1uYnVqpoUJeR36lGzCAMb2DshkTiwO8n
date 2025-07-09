#include <JuceHeader.h>
#include "training_testbed.h"

class MainComponent : public juce::AudioAppComponent, public juce::Button::Listener {
public:
    MainComponent() {
        addAndMakeVisible(deviceSelector);
        addAndMakeVisible(startButton);
        addAndMakeVisible(stopButton);
        addAndMakeVisible(exportButton);
        addAndMakeVisible(statusBox);

        startButton.setButtonText("Start Capture");
        stopButton.setButtonText("Stop Capture");
        exportButton.setButtonText("Export Training Data");
        startButton.addListener(this);
        stopButton.addListener(this);
        exportButton.addListener(this);

        setSize(600, 400);
        setAudioChannels(2, 0); // 2 input, 0 output
    }

    ~MainComponent() override {
        shutdownAudio();
    }

    void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override {
        // Initialize backend pipeline here
        statusBox.setText("Audio ready. Waiting for capture...", juce::dontSendNotification);
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override {
        // Pass audio to backend pipeline for GPU-native processing
        // Example: backend.processAudio(bufferToFill.buffer->getReadPointer(0), bufferToFill.numSamples);
    }

    void releaseResources() override {
        // Cleanup if needed
    }

    void resized() override {
        deviceSelector.setBounds(10, 10, getWidth() - 20, 40);
        startButton.setBounds(10, 60, 120, 30);
        stopButton.setBounds(140, 60, 120, 30);
        exportButton.setBounds(270, 60, 180, 30);
        statusBox.setBounds(10, 100, getWidth() - 20, getHeight() - 110);
    }

    void buttonClicked(juce::Button* button) override {
        if (button == &startButton) {
            statusBox.setText("Capturing audio...", juce::dontSendNotification);
            // Start backend capture
        } else if (button == &stopButton) {
            statusBox.setText("Capture stopped.", juce::dontSendNotification);
            // Stop backend capture
        } else if (button == &exportButton) {
            statusBox.setText("Exporting training data...", juce::dontSendNotification);
            // Trigger backend export
        }
    }

private:
    juce::AudioDeviceSelectorComponent deviceSelector{0, 2, 0, 2, false, false, true, false};
    juce::TextButton startButton, stopButton, exportButton;
    juce::TextEditor statusBox;
};

class TrainingTestbedJUCEApp : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override { return "PNBTR+JELLIE Training Testbed"; }
    const juce::String getApplicationVersion() override { return "1.0"; }
    void initialise(const juce::String&) override {
        mainWindow.reset(new MainWindow(getApplicationName()));
    }
    void shutdown() override {
        mainWindow = nullptr;
    }
    class MainWindow : public juce::DocumentWindow {
    public:
        MainWindow(juce::String name) : DocumentWindow(name,
            juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId),
            juce::DocumentWindow::allButtons) {
            setUsingNativeTitleBar(true);
            setContentOwned(new MainComponent(), true);
            setResizable(true, true);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }
        void closeButtonPressed() override {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
        }
    };
private:
    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION(TrainingTestbedJUCEApp)
