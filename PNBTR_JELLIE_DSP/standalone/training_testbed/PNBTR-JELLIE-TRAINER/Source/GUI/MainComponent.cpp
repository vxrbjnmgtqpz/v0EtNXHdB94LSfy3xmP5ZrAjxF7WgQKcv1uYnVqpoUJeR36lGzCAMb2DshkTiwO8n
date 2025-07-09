/*
  ==============================================================================

    MainComponent.cpp
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the exact schematic layout with fixed row heights:
    - Title Bar: 40px
    - Row 1 (Oscilloscopes): 200px  
    - Row 2 (Waveform Analysis): 120px
    - Row 3 (Audio Tracks): 80px
    - Row 4 (Metrics Dashboard): 100px
    - Row 5 (Controls): 60px

  ==============================================================================
*/


#include "MainComponent.h"
// JUCE module includes (must be first)
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "MainComponent.h"

using namespace juce;

//==============================================================================


MainComponent::MainComponent()
{
    title = std::make_unique<TitleComponent>();
    transportBar = std::make_unique<ProfessionalTransportController>();
    oscilloscopeRow = std::make_unique<OscilloscopeRow>();
    waveformAnalysisRow = std::make_unique<WaveformAnalysisRow>();
    audioTracksRow = std::make_unique<AudioTracksRow>();
    metricsDashboard = std::make_unique<MetricsDashboard>();
    controlsRow = std::make_unique<ControlsRow>();
    pnbtrTrainer = std::make_unique<PNBTRTrainer>();

    addAndMakeVisible(title.get());
    addAndMakeVisible(transportBar.get());
    addAndMakeVisible(oscilloscopeRow.get());
    addAndMakeVisible(waveformAnalysisRow.get());
    addAndMakeVisible(audioTracksRow.get());
    addAndMakeVisible(metricsDashboard.get());
    addAndMakeVisible(controlsRow.get());

    // Audio device dropdowns
    inputDeviceBox = std::make_unique<juce::ComboBox>("InputDevice");
    outputDeviceBox = std::make_unique<juce::ComboBox>("OutputDevice");
    addAndMakeVisible(inputDeviceBox.get());
    addAndMakeVisible(outputDeviceBox.get());
    inputDeviceBox->onChange = [this] { inputDeviceChanged(); };
    outputDeviceBox->onChange = [this] { outputDeviceChanged(); };

    // Wire transport bar to audio device and DSP
    transportBar->onPlay = [this] { handleTransportPlay(); };
    transportBar->onStop = [this] { handleTransportStop(); };
    transportBar->onRecord = [this] { handleTransportRecord(); };

    // Set up device manager (2 ins, 2 outs by default)
    deviceManager.initialise(2, 2, nullptr, true);
    deviceManager.addAudioCallback(this);
    updateDeviceLists();

    // Connect controls and visualizations to DSP pipeline
    transportBar->setTrainer(pnbtrTrainer.get());
    controlsRow->setTrainer(pnbtrTrainer.get());
    oscilloscopeRow->setTrainer(pnbtrTrainer.get());
    waveformAnalysisRow->setTrainer(pnbtrTrainer.get());
    metricsDashboard->setTrainer(pnbtrTrainer.get());
    audioTracksRow->setTrainer(pnbtrTrainer.get());
    setSize(1280, 720);
}

void MainComponent::handleTransportPlay()
{
    // Start audio device if not running
    auto* device = deviceManager.getCurrentAudioDevice();
    if (!device || !device->isOpen()) {
        juce::Logger::writeToLog("[TRANSPORT] Restarting audio device");
        deviceManager.restartLastAudioDevice();
    }
    if (pnbtrTrainer)
        pnbtrTrainer->startTraining();
    juce::Logger::writeToLog("[TRANSPORT] Play pressed, training started");
}

void MainComponent::handleTransportStop()
{
    // Stop audio device
    juce::Logger::writeToLog("[TRANSPORT] Stop pressed, closing audio device");
    deviceManager.closeAudioDevice();
    if (pnbtrTrainer)
        pnbtrTrainer->stopTraining();
}

void MainComponent::handleTransportRecord()
{
    // Start recording (set recordingActive)
    if (pnbtrTrainer)
        pnbtrTrainer->recordingActive.store(true);
    handleTransportPlay();
}
// Audio device management
void MainComponent::updateDeviceLists()
{
    inputDeviceBox->clear();
    outputDeviceBox->clear();
    auto setup = deviceManager.getAudioDeviceSetup();
    juce::StringArray inputNames = deviceManager.getAvailableDeviceTypes()[0]->getDeviceNames(true);
    juce::StringArray outputNames = deviceManager.getAvailableDeviceTypes()[0]->getDeviceNames(false);
    for (int i = 0; i < inputNames.size(); ++i)
        inputDeviceBox->addItem(inputNames[i], i + 1);
    for (int i = 0; i < outputNames.size(); ++i)
        outputDeviceBox->addItem(outputNames[i], i + 1);
    inputDeviceBox->setSelectedId(inputNames.indexOf(setup.inputDeviceName) + 1, juce::dontSendNotification);
    outputDeviceBox->setSelectedId(outputNames.indexOf(setup.outputDeviceName) + 1, juce::dontSendNotification);
}

void MainComponent::inputDeviceChanged()
{
    auto setup = deviceManager.getAudioDeviceSetup();
    setup.inputDeviceName = inputDeviceBox->getText();
    deviceManager.setAudioDeviceSetup(setup, true);
}

void MainComponent::outputDeviceChanged()
{
    auto setup = deviceManager.getAudioDeviceSetup();
    setup.outputDeviceName = outputDeviceBox->getText();
    deviceManager.setAudioDeviceSetup(setup, true);
}

void MainComponent::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    if (pnbtrTrainer)
        pnbtrTrainer->prepareToPlay(device->getCurrentSampleRate(), device->getCurrentBufferSizeSamples());
}

void MainComponent::audioDeviceStopped()
{
    if (pnbtrTrainer)
        pnbtrTrainer->releaseResources();
}

void MainComponent::audioDeviceIOCallback(const float** inputChannelData, int numInputChannels,
                                          float** outputChannelData, int numOutputChannels, int numSamples)
{
    static int callbackCount = 0;
    if (++callbackCount % 100 == 0) {
        juce::Logger::writeToLog("[AUDIO CALLBACK] Running (" + juce::String(callbackCount) + ")");
    }
    // Defensive: zero output in case of crash or uninitialized buffer
    for (int ch = 0; ch < numOutputChannels; ++ch)
        std::memset(outputChannelData[ch], 0, sizeof(float) * numSamples);

    // Thread-safe audio buffer copy and processing
    juce::AudioBuffer<float> buffer(const_cast<float**>(inputChannelData), numInputChannels, numSamples);
    if (pnbtrTrainer)
    {
        // Try/catch to prevent any crash from killing the audio thread
        try {
            juce::MidiBuffer midi;
            pnbtrTrainer->processBlock(buffer, midi);
        } catch (const std::exception& e) {
            juce::Logger::writeToLog("[AUDIO CALLBACK] Exception: " + juce::String(e.what()));
        } catch (...) {
            juce::Logger::writeToLog("[AUDIO CALLBACK] Unknown exception");
        }
    }
    // Copy processed buffer to output (thread-safe)
    for (int ch = 0; ch < numOutputChannels; ++ch)
    {
        if (ch < buffer.getNumChannels())
            std::memcpy(outputChannelData[ch], buffer.getReadPointer(ch), sizeof(float) * numSamples);
    }
}

MainComponent::~MainComponent() = default;

//==============================================================================
void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::darkgrey);

    // Optional: draw horizontal lines between rows
    const int rowHeights[] = {40, 48, 200, 240, 160, 100, 60};
    int y = rowHeights[0];
    for (int i = 1; i < 7; ++i)
    {
        y += rowHeights[i - 1];
        g.drawLine(0.0f, (float)y, (float)getWidth(), (float)y, 1.0f);
    }
}

void MainComponent::resized()
{
    const int rowHeights[] = {40, 48, 200, 240, 160, 100, 60};
    juce::Rectangle<int> area = getLocalBounds();

    title->setBounds(area.removeFromTop(rowHeights[0]));
    transportBar->setBounds(area.removeFromTop(rowHeights[1]));
    // Place device dropdowns above oscilloscope row
    auto deviceBar = area.removeFromTop(32);
    inputDeviceBox->setBounds(deviceBar.removeFromLeft(getWidth() / 2).reduced(8, 4));
    outputDeviceBox->setBounds(deviceBar.reduced(8, 4));
    oscilloscopeRow->setBounds(area.removeFromTop(rowHeights[2]));
    waveformAnalysisRow->setBounds(area.removeFromTop(rowHeights[3]));
    audioTracksRow->setBounds(area.removeFromTop(rowHeights[4]));
    metricsDashboard->setBounds(area.removeFromTop(rowHeights[5]));
    controlsRow->setBounds(area.removeFromTop(rowHeights[6]));
}
