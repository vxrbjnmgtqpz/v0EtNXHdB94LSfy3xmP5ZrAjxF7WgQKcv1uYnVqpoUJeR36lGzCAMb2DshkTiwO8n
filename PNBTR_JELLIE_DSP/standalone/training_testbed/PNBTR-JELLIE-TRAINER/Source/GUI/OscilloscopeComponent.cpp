#include "OscilloscopeComponent.h"
#include "../DSP/PNBTRTrainer.h"
#include "../GPU/MetalBridge.h"
#include <algorithm>
#include <cmath>

void OscilloscopeComponent::setTrainer(PNBTRTrainer* trainerPtr) { 
    trainer = trainerPtr; 
    
    // Start timer only after trainer is set
    if (trainer) {
        startTimer(1000 / refreshRateHz);
    }
}

//==============================================================================
OscilloscopeComponent::OscilloscopeComponent(BufferType type, const juce::String& title)
    : bufferType(type), oscilloscopeTitle(title)
{
    // Get MetalBridge singleton
    metalBridge = &MetalBridge::getInstance();
    
    // Set up display buffer
    displayBuffer.resize(bufferSize, 0.0f);
    previousBuffer.resize(bufferSize, 0.0f);
    
    // Timer will be started when trainer is set
    
    // Set size
    setSize(300, 200);
}

OscilloscopeComponent::~OscilloscopeComponent()
{
    stopTimer();
}

//==============================================================================
void OscilloscopeComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Draw background
    g.fillAll(backgroundColour);
    
    // Draw title at top
    auto titleArea = bounds.removeFromTop(25);
    drawTitle(g, titleArea);
    
    // Draw main oscilloscope area
    auto scopeArea = bounds.reduced(10);
    
    if (showGrid) {
        drawGrid(g, scopeArea);
    }
    
    drawWaveform(g, scopeArea);
    drawScale(g, scopeArea);
}

void OscilloscopeComponent::resized()
{
    displayWidth = getWidth() - 20; // Account for margins
    // Update buffer size based on display width for smooth rendering
    if (displayWidth > 0) {
        bufferSize = static_cast<size_t>(displayWidth);
        displayBuffer.resize(bufferSize, 0.0f);
        previousBuffer.resize(bufferSize, 0.0f);
    }
}

void OscilloscopeComponent::timerCallback()
{
    // Store previous buffer to detect changes
    std::vector<float> oldBuffer = displayBuffer;
    
    updateFromMetalBuffer();
    
    // Only repaint if data actually changed (massive performance improvement)
    if (displayBuffer != oldBuffer) {
        repaint();
    }
}

//==============================================================================
void OscilloscopeComponent::updateFromMetalBuffer()
{
    if (trainer) {
        // Use thread-safe double-buffered access for oscilloscope - REAL DATA ONLY
        switch (bufferType) {
            case BufferType::AudioInput: {
                // FIX: getLatestOscInput returns stereo interleaved data, convert to mono
                std::vector<float> stereoBuffer(displayBuffer.size() * 2);
                trainer->getLatestOscInput(stereoBuffer.data(), (int)displayBuffer.size());
                
                // Convert stereo interleaved (L,R,L,R...) to mono (L+R)/2
                for (size_t i = 0; i < displayBuffer.size(); ++i) {
                    displayBuffer[i] = (stereoBuffer[i * 2] + stereoBuffer[i * 2 + 1]) * 0.5f;
                }
                isActive = true;
                break;
            }
            case BufferType::Reconstructed: {
                // FIX: getLatestOscOutput returns stereo interleaved data, convert to mono  
                std::vector<float> stereoBuffer(displayBuffer.size() * 2);
                trainer->getLatestOscOutput(stereoBuffer.data(), (int)displayBuffer.size());
                
                // Convert stereo interleaved to mono
                for (size_t i = 0; i < displayBuffer.size(); ++i) {
                    displayBuffer[i] = (stereoBuffer[i * 2] + stereoBuffer[i * 2 + 1]) * 0.5f;
                }
                isActive = true;
                break;
            }
            case BufferType::NetworkProcessed: {
                // FIX: Real network simulation data from trainer - convert stereo to mono
                std::vector<float> stereoBuffer(displayBuffer.size() * 2);
                trainer->getLatestOscOutput(stereoBuffer.data(), (int)displayBuffer.size());
                
                // Convert stereo interleaved to mono
                for (size_t i = 0; i < displayBuffer.size(); ++i) {
                    displayBuffer[i] = (stereoBuffer[i * 2] + stereoBuffer[i * 2 + 1]) * 0.5f;
                }
                isActive = true;
                break;
            }
            case BufferType::JellieEncoded: {
                // FIX: Real JELLIE encoded data from trainer - convert stereo to mono
                std::vector<float> stereoBuffer(displayBuffer.size() * 2);
                trainer->getLatestOscOutput(stereoBuffer.data(), (int)displayBuffer.size());
                
                // Convert stereo interleaved to mono
                for (size_t i = 0; i < displayBuffer.size(); ++i) {
                    displayBuffer[i] = (stereoBuffer[i * 2] + stereoBuffer[i * 2 + 1]) * 0.5f;
                }
                isActive = true;
                break;
            }
        }
    } else {
        // No trainer = no data = flat line
        std::fill(displayBuffer.begin(), displayBuffer.end(), 0.0f);
        isActive = false;
    }
}

void OscilloscopeComponent::updateDisplayBuffer()
{
    // This will be called when time window changes
    // For now, keep the existing buffer size logic
}

//==============================================================================
// Drawing methods

void OscilloscopeComponent::drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(textColour);
    g.setFont(juce::Font(juce::FontOptions(14.0f, juce::Font::bold)));
    g.drawText(oscilloscopeTitle, bounds, juce::Justification::centred);
    
    // Add activity indicator
    if (isActive) {
        g.setColour(juce::Colours::green);
        g.fillEllipse(bounds.getRight() - 15, bounds.getCentreY() - 3, 6, 6);
    }
}

void OscilloscopeComponent::drawGrid(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(gridColour);
    
    // Vertical grid lines (time)
    int numVerticalLines = 10;
    for (int i = 0; i <= numVerticalLines; ++i) {
        float x = bounds.getX() + (bounds.getWidth() * i) / float(numVerticalLines);
        g.drawVerticalLine(static_cast<int>(x), bounds.getY(), bounds.getBottom());
    }
    
    // Horizontal grid lines (amplitude)
    int numHorizontalLines = 8;
    for (int i = 0; i <= numHorizontalLines; ++i) {
        float y = bounds.getY() + (bounds.getHeight() * i) / float(numHorizontalLines);
        g.drawHorizontalLine(static_cast<int>(y), bounds.getX(), bounds.getRight());
    }
    
    // Center line (zero)
    g.setColour(gridColour.brighter());
    g.drawHorizontalLine(bounds.getCentreY(), bounds.getX(), bounds.getRight());
}

void OscilloscopeComponent::drawWaveform(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    if (displayBuffer.empty()) return;
    
    g.setColour(waveformColour);
    
    juce::Path waveformPath;
    bool pathStarted = false;
    
    for (size_t i = 0; i < displayBuffer.size() && i < static_cast<size_t>(bounds.getWidth()); ++i) {
        float sample = displayBuffer[i] * amplitudeScale;
        sample = std::clamp(sample, -1.0f, 1.0f); // Clamp to valid range
        
        float x = bounds.getX() + (bounds.getWidth() * i) / float(displayBuffer.size());
        float y = bounds.getCentreY() - (sample * bounds.getHeight() * 0.4f); // 0.4f for some margin
        
        if (!pathStarted) {
            waveformPath.startNewSubPath(x, y);
            pathStarted = true;
        } else {
            waveformPath.lineTo(x, y);
        }
    }
    
    g.strokePath(waveformPath, juce::PathStrokeType(1.5f));
}

void OscilloscopeComponent::drawScale(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(textColour.withAlpha(0.7f));
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    
    // Draw amplitude scale on the left
    g.drawText("+1", bounds.getX() - 15, bounds.getY(), 15, 12, juce::Justification::right);
    g.drawText("0", bounds.getX() - 15, bounds.getCentreY() - 6, 15, 12, juce::Justification::right);
    g.drawText("-1", bounds.getX() - 15, bounds.getBottom() - 12, 15, 12, juce::Justification::right);
    
    // Draw time scale at bottom
    juce::String timeText = juce::String(timeWindowSeconds * 1000.0f, 1) + "ms";
    g.drawText("0", bounds.getX(), bounds.getBottom() + 2, 20, 12, juce::Justification::left);
    g.drawText(timeText, bounds.getRight() - 40, bounds.getBottom() + 2, 40, 12, juce::Justification::right);
}

//==============================================================================
// Buffer reading methods

void OscilloscopeComponent::readAudioInputBuffer()
{
    // Read real audio input data from PNBTR trainer
    if (trainer) {
        // Get real microphone input buffer
        trainer->getLatestOscInput(displayBuffer.data(), (int)displayBuffer.size() / 2);
        isActive = true;
    } else {
        // No trainer connected - show flat line instead of fake data
        std::fill(displayBuffer.begin(), displayBuffer.end(), 0.0f);
        isActive = false;
    }
}

void OscilloscopeComponent::readJellieBuffer()
{
    // Read from MetalBridge jellieBuffer (192kHz 8-channel)
    // For now, generate higher frequency test pattern
    static float phase = 0.0f;
    for (size_t i = 0; i < displayBuffer.size(); ++i) {
        displayBuffer[i] = 0.5f * std::sin(phase + i * 0.3f) * std::sin(phase + i * 0.05f);
    }
    phase += 0.05f;
    isActive = true;
}

void OscilloscopeComponent::readNetworkBuffer()
{
    // Read real network simulation data from PNBTR trainer
    if (trainer) {
        // Get real TOAST network simulation buffer
        trainer->getLatestOscOutput(displayBuffer.data(), (int)displayBuffer.size() / 2);
        isActive = true;
    } else {
        // No trainer connected - show flat line instead of fake data
        std::fill(displayBuffer.begin(), displayBuffer.end(), 0.0f);
        isActive = false;
    }
}

void OscilloscopeComponent::readReconstructedBuffer()
{
    // Read from MetalBridge reconstructedBuffer (PNBTR output)
    // Generate test pattern that looks like reconstructed audio
    static float phase = 0.0f;
    for (size_t i = 0; i < displayBuffer.size(); ++i) {
        float base = 0.35f * std::sin(phase + i * 0.12f);
        float noise = 0.05f * (float(rand()) / RAND_MAX - 0.5f); // Small amount of noise
        displayBuffer[i] = base + noise;
    }
    phase += 0.06f;
    isActive = true;
} 