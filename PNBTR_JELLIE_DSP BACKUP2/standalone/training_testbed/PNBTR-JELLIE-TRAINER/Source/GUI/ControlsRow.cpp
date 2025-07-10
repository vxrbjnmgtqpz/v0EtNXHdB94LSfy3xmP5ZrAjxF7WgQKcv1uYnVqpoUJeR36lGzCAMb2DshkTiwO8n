#include "ControlsRow.h"
#include "../DSP/PNBTRTrainer.h"

//==============================================================================
ControlsRow::ControlsRow()
{
    // Transport controls (GPU-aware pattern from TOASTer)
    startButton = std::make_unique<juce::TextButton>("Start");
    stopButton = std::make_unique<juce::TextButton>("Stop");
    exportButton = std::make_unique<juce::TextButton>("Export WAV");
    
    // Configure transport buttons
    startButton->setColour(juce::TextButton::buttonColourId, juce::Colours::green.withAlpha(0.7f));
    stopButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.7f));
    exportButton->setColour(juce::TextButton::buttonColourId, juce::Colours::blue.withAlpha(0.7f));
    
    // Set up callbacks
    startButton->onClick = [this] { startProcessing(); };
    stopButton->onClick = [this] { stopProcessing(); };
    exportButton->onClick = [this] { exportSession(); };
    
    addAndMakeVisible(*startButton);
    addAndMakeVisible(*stopButton);
    addAndMakeVisible(*exportButton);
    
    // Network parameter sliders (write to GPU config atomically)
    packetLossSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    jitterSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    gainSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxBelow);
    
    // Configure packet loss slider (0-20%)
    packetLossSlider->setRange(0.0, 20.0, 0.1);
    packetLossSlider->setValue(0.0);
    packetLossSlider->setTextValueSuffix("%");
    packetLossSlider->onValueChange = [this] { onPacketLossChanged(); };
    
    // Configure jitter slider (0-50ms)
    jitterSlider->setRange(0.0, 50.0, 0.1);
    jitterSlider->setValue(0.0);
    jitterSlider->setTextValueSuffix("ms");
    jitterSlider->onValueChange = [this] { onJitterChanged(); };
    
    // Configure gain slider (-20 to +20 dB)
    gainSlider->setRange(-20.0, 20.0, 0.1);
    gainSlider->setValue(0.0);
    gainSlider->setTextValueSuffix("dB");
    gainSlider->onValueChange = [this] { onGainChanged(); };
    
    addAndMakeVisible(*packetLossSlider);
    addAndMakeVisible(*jitterSlider);
    addAndMakeVisible(*gainSlider);
    
    // Labels
    packetLossLabel = std::make_unique<juce::Label>("", "Packet Loss");
    jitterLabel = std::make_unique<juce::Label>("", "Jitter");
    gainLabel = std::make_unique<juce::Label>("", "Gain");
    
    packetLossLabel->setJustificationType(juce::Justification::centred);
    jitterLabel->setJustificationType(juce::Justification::centred);
    gainLabel->setJustificationType(juce::Justification::centred);
    
    addAndMakeVisible(*packetLossLabel);
    addAndMakeVisible(*jitterLabel);
    addAndMakeVisible(*gainLabel);
}

ControlsRow::~ControlsRow() = default;

//==============================================================================
void ControlsRow::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey.withAlpha(0.3f));
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void ControlsRow::resized()
{
    auto area = getLocalBounds().reduced(4);
    
    // Transport controls on the left (3 buttons)
    auto transportArea = area.removeFromLeft(300);
    auto buttonWidth = transportArea.getWidth() / 3 - 4;
    
    startButton->setBounds(transportArea.removeFromLeft(buttonWidth));
    transportArea.removeFromLeft(4);
    stopButton->setBounds(transportArea.removeFromLeft(buttonWidth));
    transportArea.removeFromLeft(4);
    exportButton->setBounds(transportArea);
    
    // Network parameter sliders on the right
    area.removeFromLeft(20); // spacing
    auto sliderWidth = area.getWidth() / 3 - 8;
    
    // Packet Loss
    auto packetLossArea = area.removeFromLeft(sliderWidth);
    packetLossLabel->setBounds(packetLossArea.removeFromTop(18));
    packetLossSlider->setBounds(packetLossArea);
    
    area.removeFromLeft(12); // spacing
    
    // Jitter
    auto jitterArea = area.removeFromLeft(sliderWidth);
    jitterLabel->setBounds(jitterArea.removeFromTop(18));
    jitterSlider->setBounds(jitterArea);
    
    area.removeFromLeft(12); // spacing
    
    // Gain
    gainLabel->setBounds(area.removeFromTop(18));
    gainSlider->setBounds(area);
}

//==============================================================================
void ControlsRow::setTrainer(PNBTRTrainer* t)
{
    trainer = t;
}

void ControlsRow::updateTransportState(bool isPlaying, bool isRecording)
{
    startButton->setEnabled(!isPlaying);
    stopButton->setEnabled(isPlaying);
    exportButton->setEnabled(!isPlaying && !isRecording);
}

void ControlsRow::updateNetworkParameters(float packetLoss, float jitter, float gain)
{
    packetLossSlider->setValue(packetLoss, juce::dontSendNotification);
    jitterSlider->setValue(jitter, juce::dontSendNotification);
    gainSlider->setValue(gain, juce::dontSendNotification);
}

//==============================================================================
void ControlsRow::startProcessing()
{
    if (trainer != nullptr) {
        trainer->startTraining();
        // If "Record" is armed, enable recording
        trainer->recordingActive.store(true);
    }
}

void ControlsRow::stopProcessing()
{
    if (trainer != nullptr) {
        trainer->stopTraining();
        trainer->recordingActive.store(false);
    }
}

void ControlsRow::exportSession()
{
    // Export the recorded buffer to a WAV file
    if (trainer == nullptr)
        return;

    // Get the recorded buffer (entire buffer)
    constexpr int numChannels = 2;
    constexpr double sampleRate = 48000.0; // TODO: get from trainer if dynamic
    const size_t bufferLength = 48000 * 10; // 10 seconds, stereo interleaved
    std::vector<float> exportBuffer(bufferLength * numChannels);
    trainer->getRecordedBuffer(exportBuffer.data(), (int)bufferLength, 0);

    // Create a JUCE AudioBuffer for writing
    juce::AudioBuffer<float> audioBuffer(numChannels, (int)bufferLength);
    for (int i = 0; i < (int)bufferLength; ++i) {
        audioBuffer.setSample(0, i, exportBuffer[i * 2]);
        audioBuffer.setSample(1, i, exportBuffer[i * 2 + 1]);
    }

    // Prompt user for file location
    juce::FileChooser chooser("Export Recorded Audio as WAV", juce::File::getSpecialLocation(juce::File::userDesktopDirectory), "*.wav");
    chooser.launchAsync(juce::FileBrowserComponent::saveMode, [audioBuffer = std::move(audioBuffer), sampleRate, numChannels, bufferLength](const juce::FileChooser& fc) mutable {
        auto file = fc.getResult();
        if (file != juce::File{}) {
            juce::WavAudioFormat wavFormat;
            std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
            if (stream != nullptr) {
                std::unique_ptr<juce::AudioFormatWriter> writer(wavFormat.createWriterFor(stream.get(), sampleRate, numChannels, 16, {}, 0));
                if (writer != nullptr) {
                    stream.release(); // Writer now owns the stream
                    writer->writeFromAudioSampleBuffer(audioBuffer, 0, (int)bufferLength);
                }
            }
        }
    });
}

void ControlsRow::onPacketLossChanged()
{
    if (trainer != nullptr)
        trainer->setPacketLossPercentage(static_cast<float>(packetLossSlider->getValue()));
}

void ControlsRow::onJitterChanged()
{
    if (trainer != nullptr)
        trainer->setJitterAmount(static_cast<float>(jitterSlider->getValue()));
}

void ControlsRow::onGainChanged()
{
    if (trainer != nullptr)
        trainer->setGain(static_cast<float>(gainSlider->getValue()));
}
