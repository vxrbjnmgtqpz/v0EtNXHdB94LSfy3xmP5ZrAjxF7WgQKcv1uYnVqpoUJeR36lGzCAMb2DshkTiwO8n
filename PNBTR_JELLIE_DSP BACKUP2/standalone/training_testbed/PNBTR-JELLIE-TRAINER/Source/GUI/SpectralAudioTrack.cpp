#include "SpectralAudioTrack.h"
#include "../DSP/PNBTRTrainer.h"
#include <algorithm>

SpectralAudioTrack::SpectralAudioTrack(TrackType type, const std::string& trackName)
    : trackType(type), trackName(trackName)
{
    // Initialize FFT buffers
    fftBuffer.resize(FFT_SIZE * 2);
    spectrumData.resize(FFT_SIZE / 2);
    
    // Set track-specific colors
    if (trackType == TrackType::JELLIE_INPUT) {
        waveformColour = juce::Colours::cyan;
        spectrumColour = juce::Colours::lightblue;
    } else {
        waveformColour = juce::Colours::orange;
        spectrumColour = juce::Colours::yellow;
    }
    
    // Start visualization updates at 30 FPS
    startTimer(33);
}

SpectralAudioTrack::~SpectralAudioTrack()
{
    stopTimer();
}

void SpectralAudioTrack::paint(juce::Graphics& g)
{
    // Background
    g.setColour(juce::Colours::black);
    g.fillRect(getLocalBounds());
    
    // Border
    g.setColour(juce::Colours::darkgrey);
    g.drawRect(getLocalBounds(), 1);
    
    // Draw each section
    drawHeader(g, headerArea);
    drawWaveform(g, waveformArea);
    drawSpectrum(g, spectrumArea);
    drawControls(g, controlsArea);
    
    // Record arm indicator (overlaid)
    if (recordArmed) {
        drawRecordArmIndicator(g, getLocalBounds());
    }
}

void SpectralAudioTrack::resized()
{
    auto bounds = getLocalBounds();
    
    headerArea = bounds.removeFromTop(20);
    controlsArea = bounds.removeFromBottom(15);
    
    // Split remaining area between waveform and spectrum
    auto remainingHeight = bounds.getHeight();
    waveformArea = bounds.removeFromTop(remainingHeight * 0.6f);
    spectrumArea = bounds;
}

void SpectralAudioTrack::timerCallback()
{
    // Update spectral analysis if we have new audio data
    if (!audioBuffer.empty()) {
        performFFT();
        updateSpectrum();
    }
    
    // Get live audio data from trainer if available
    if (pnbtrTrainer) {
        // Get appropriate audio data based on track type (fixed buffer size for now)
        const int bufferSize = 512;
        std::vector<float> tempBuffer(bufferSize);
        
        if (trackType == TrackType::JELLIE_INPUT) {
            pnbtrTrainer->getInputBuffer(tempBuffer.data(), bufferSize);
            addAudioData(tempBuffer.data(), bufferSize, 48000.0);
        } else if (trackType == TrackType::PNBTR_OUTPUT) {
            // For now, use input buffer as placeholder since getReconstructedBuffer may not exist
            pnbtrTrainer->getInputBuffer(tempBuffer.data(), bufferSize);
            addAudioData(tempBuffer.data(), bufferSize, 48000.0);
        }
    }
    
    repaint();
}

void SpectralAudioTrack::addAudioData(const float* audioData, int numSamples, double sampleRate)
{
    if (!audioData || numSamples <= 0) return;
    
    currentSampleRate = sampleRate;
    
    // Add to live audio buffer (circular buffer)
    for (int i = 0; i < numSamples; ++i) {
        audioBuffer.push_back(audioData[i]);
        
        // Limit buffer size to prevent memory issues
        if (audioBuffer.size() > MAX_BUFFER_SIZE) {
            audioBuffer.erase(audioBuffer.begin());
        }
    }
    
    // If recording, add to recording buffer
    if (recording) {
        for (int i = 0; i < numSamples; ++i) {
            recordingBuffer.push_back(audioData[i]);
        }
    }
}

void SpectralAudioTrack::clearAudioData()
{
    audioBuffer.clear();
    recordingBuffer.clear();
    std::lock_guard<std::mutex> lock(spectrumMutex);
    std::fill(spectrumData.begin(), spectrumData.end(), 0.0f);
}

void SpectralAudioTrack::setRecordArmed(bool armed)
{
    recordArmed = armed;
    if (!armed && recording) {
        stopRecording();
    }
}

void SpectralAudioTrack::startRecording()
{
    if (recordArmed) {
        recording = true;
        recordingBuffer.clear();
    }
}

void SpectralAudioTrack::stopRecording()
{
    recording = false;
}

void SpectralAudioTrack::performFFT()
{
    if (audioBuffer.size() < FFT_SIZE) return;
    
    // Copy the most recent samples to FFT buffer
    std::copy(audioBuffer.end() - FFT_SIZE, audioBuffer.end(), fftBuffer.begin());
    
    // Apply windowing
    window.multiplyWithWindowingTable(fftBuffer.data(), FFT_SIZE);
    
    // Perform FFT
    fft.performFrequencyOnlyForwardTransform(fftBuffer.data());
}

void SpectralAudioTrack::updateSpectrum()
{
    std::lock_guard<std::mutex> lock(spectrumMutex);
    
    // Convert FFT output to spectrum data
    for (int i = 0; i < FFT_SIZE / 2; ++i) {
        float magnitude = fftBuffer[i];
        
        // Convert to dB scale
        if (magnitude > 0.0f) {
            spectrumData[i] = 20.0f * log10f(magnitude);
        } else {
            spectrumData[i] = -100.0f; // Noise floor
        }
        
        // Smooth the spectrum (simple exponential smoothing)
        spectrumData[i] = spectrumData[i] * 0.3f + spectrumData[i] * 0.7f;
    }
}

bool SpectralAudioTrack::exportToWAV(const juce::File& file)
{
    if (recordingBuffer.empty()) return false;
    
    // Create audio format writer
    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer;
    
    juce::FileOutputStream* outputStream = new juce::FileOutputStream(file);
    if (outputStream->openedOk()) {
        writer.reset(wavFormat.createWriterFor(outputStream, currentSampleRate, 1, 16, {}, 0));
        if (writer) {
            // Convert vector to AudioBuffer for writing
            juce::AudioBuffer<float> buffer(1, static_cast<int>(recordingBuffer.size()));
            buffer.copyFrom(0, 0, recordingBuffer.data(), static_cast<int>(recordingBuffer.size()));
            
            writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
            writer.reset();
            return true;
        }
    }
    
    delete outputStream;
    return false;
}

void SpectralAudioTrack::drawHeader(juce::Graphics& g, juce::Rectangle<int> area)
{
    g.setColour(juce::Colours::white);
    g.setFont(12.0f);
    
    std::string headerText = trackName;
    if (trackType == TrackType::JELLIE_INPUT) {
        headerText += " (Input - JELLIE Encoding)";
    } else {
        headerText += " (Output - PNBTR Reconstruction)";
    }
    
    g.drawText(headerText, area, juce::Justification::centredLeft, true);
    
    // Recording status
    if (recording) {
        g.setColour(juce::Colours::red);
        g.fillEllipse(area.getRight() - 50, area.getY() + 3, 10, 10);
        g.setColour(juce::Colours::white);
        g.drawText("REC", area.getRight() - 35, area.getY(), 30, area.getHeight(), juce::Justification::centredLeft);
    } else if (recordArmed) {
        g.setColour(juce::Colours::red);
        g.drawEllipse(area.getRight() - 50, area.getY() + 3, 10, 10, 1);
        g.setColour(juce::Colours::lightgrey);
        g.drawText("ARM", area.getRight() - 35, area.getY(), 30, area.getHeight(), juce::Justification::centredLeft);
    }
}

void SpectralAudioTrack::drawWaveform(juce::Graphics& g, juce::Rectangle<int> area)
{
    if (audioBuffer.empty()) {
        g.setColour(juce::Colours::grey);
        g.setFont(11.0f);
        g.drawText("No audio data", area, juce::Justification::centred);
        return;
    }
    
    // Draw waveform
    g.setColour(waveformColour);
    
    float width = static_cast<float>(area.getWidth());
    float height = static_cast<float>(area.getHeight());
    float centerY = area.getY() + height * 0.5f;
    
    juce::Path waveformPath;
    bool firstPoint = true;
    
    // Sample the audio buffer for display
    int samplesPerPixel = std::max(1, static_cast<int>(audioBuffer.size()) / static_cast<int>(width));
    
    for (int x = 0; x < static_cast<int>(width); ++x) {
        int sampleIndex = x * samplesPerPixel;
        if (sampleIndex >= static_cast<int>(audioBuffer.size())) break;
        
        // Find min/max in this pixel column
        float minVal = audioBuffer[sampleIndex];
        float maxVal = audioBuffer[sampleIndex];
        
        for (int i = 0; i < samplesPerPixel && (sampleIndex + i) < static_cast<int>(audioBuffer.size()); ++i) {
            float sample = audioBuffer[sampleIndex + i];
            minVal = std::min(minVal, sample);
            maxVal = std::max(maxVal, sample);
        }
        
        float y1 = centerY - (maxVal * height * 0.4f);
        float y2 = centerY - (minVal * height * 0.4f);
        
        g.drawLine(area.getX() + x, y1, area.getX() + x, y2, 1.0f);
    }
}

void SpectralAudioTrack::drawSpectrum(juce::Graphics& g, juce::Rectangle<int> area)
{
    std::lock_guard<std::mutex> lock(spectrumMutex);
    
    if (spectrumData.empty()) {
        g.setColour(juce::Colours::grey);
        g.setFont(9.0f);
        g.drawText("No spectrum data", area, juce::Justification::centred);
        return;
    }
    
    g.setColour(spectrumColour);
    
    float width = static_cast<float>(area.getWidth());
    float height = static_cast<float>(area.getHeight());
    
    juce::Path spectrumPath;
    spectrumPath.startNewSubPath(area.getX(), area.getBottom());
    
    // Draw spectrum
    for (int i = 0; i < static_cast<int>(spectrumData.size()) && i < static_cast<int>(width); ++i) {
        float frequency = (i * currentSampleRate) / (2.0f * spectrumData.size());
        float x = mapFrequencyToX(frequency, width);
        float magnitude = spectrumData[i];
        float y = mapAmplitudeToY(magnitude, height);
        
        spectrumPath.lineTo(area.getX() + x, area.getY() + y);
    }
    
    spectrumPath.lineTo(area.getRight(), area.getBottom());
    spectrumPath.closeSubPath();
    
    g.setColour(spectrumColour.withAlpha(0.3f));
    g.fillPath(spectrumPath);
    
    g.setColour(spectrumColour);
    g.strokePath(spectrumPath, juce::PathStrokeType(1.0f));
}

void SpectralAudioTrack::drawControls(juce::Graphics& g, juce::Rectangle<int> area)
{
    g.setColour(juce::Colours::lightgrey);
    g.setFont(8.0f);
    
    std::string info = "Sample Rate: " + std::to_string(static_cast<int>(currentSampleRate)) + " Hz";
    info += " | Buffer: " + std::to_string(audioBuffer.size()) + " samples";
    
    if (recording) {
        info += " | Recording: " + std::to_string(recordingBuffer.size()) + " samples";
    }
    
    g.drawText(info, area, juce::Justification::centredLeft);
}

void SpectralAudioTrack::drawRecordArmIndicator(juce::Graphics& g, juce::Rectangle<int> area)
{
    // Subtle red tint when record armed
    g.setColour(recordArmColour.withAlpha(0.1f));
    g.fillRect(area);
    
    // Pulsing border when recording
    if (recording) {
        float alpha = 0.5f + 0.3f * sinf(juce::Time::getMillisecondCounter() * 0.01f);
        g.setColour(recordArmColour.withAlpha(alpha));
        g.drawRect(area, 2);
    }
}

float SpectralAudioTrack::mapFrequencyToX(float frequency, float width) const
{
    // Logarithmic frequency mapping (like professional spectrum analyzers)
    float minFreq = 20.0f;
    float maxFreq = static_cast<float>(currentSampleRate * 0.5);
    
    if (frequency <= minFreq) return 0.0f;
    if (frequency >= maxFreq) return width;
    
    float logMin = log10f(minFreq);
    float logMax = log10f(maxFreq);
    float logFreq = log10f(frequency);
    
    return width * (logFreq - logMin) / (logMax - logMin);
}

float SpectralAudioTrack::mapAmplitudeToY(float amplitude, float height) const
{
    // Map dB range (-100 to 0) to height
    float dbMin = -100.0f;
    float dbMax = 0.0f;
    
    amplitude = std::clamp(amplitude, dbMin, dbMax);
    float normalized = (amplitude - dbMin) / (dbMax - dbMin);
    
    return height * (1.0f - normalized);
} 