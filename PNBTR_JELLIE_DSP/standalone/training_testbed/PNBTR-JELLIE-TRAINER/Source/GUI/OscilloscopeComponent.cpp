#include "OscilloscopeComponent.h"
#include "../DSP/PNBTRTrainer.h"
#include "../GPU/MetalBridgeInterface.h"
#include <algorithm>
#include <cmath>

void OscilloscopeComponent::setTrainer(PNBTRTrainer* trainerPtr) { trainer = trainerPtr; }
#include "OscilloscopeComponent.h"
#include "../GPU/MetalBridgeInterface.h"
#include <algorithm>
#include <cmath>

//==============================================================================
OscilloscopeComponent::OscilloscopeComponent(BufferType type, const juce::String& title)
    : bufferType(type), oscilloscopeTitle(title)
{
    // Get MetalBridge singleton
    metalBridge = &MetalBridgeInterface::getInstance();
    
    // Set up display buffer
    displayBuffer.resize(bufferSize, 0.0f);
    previousBuffer.resize(bufferSize, 0.0f);
    
    // Start timer for real-time updates
    startTimer(1000 / refreshRateHz);
    
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
    updateFromMetalBuffer();
    repaint();
}

//==============================================================================
void OscilloscopeComponent::updateFromMetalBuffer()
{
    if (trainer) {
        // Use real DSP buffers if trainer is set
        switch (bufferType) {
            case BufferType::AudioInput:
                trainer->getInputBuffer(displayBuffer.data(), (int)displayBuffer.size());
                isActive = true;
                break;
            case BufferType::Reconstructed:
                trainer->getOutputBuffer(displayBuffer.data(), (int)displayBuffer.size());
                isActive = true;
                break;
            default:
                // For other types, fall back to MetalBridge/test pattern for now
                if (metalBridge) {
                    switch (bufferType) {
                        case BufferType::JellieEncoded:
                            readJellieBuffer(); break;
                        case BufferType::NetworkProcessed:
                            readNetworkBuffer(); break;
                        default: break;
                    }
                }
                break;
        }
    } else if (metalBridge) {
        // Fallback: test pattern/MetalBridge
        switch (bufferType) {
            case BufferType::AudioInput:
                readAudioInputBuffer(); break;
            case BufferType::JellieEncoded:
                readJellieBuffer(); break;
            case BufferType::NetworkProcessed:
                readNetworkBuffer(); break;
            case BufferType::Reconstructed:
                readReconstructedBuffer(); break;
        }
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
    // Read from MetalBridge audioInputBuffer
    // For now, generate test pattern until Metal integration is complete
    static float phase = 0.0f;
    for (size_t i = 0; i < displayBuffer.size(); ++i) {
        displayBuffer[i] = 0.3f * std::sin(phase + i * 0.1f);
    }
    phase += 0.1f;
    isActive = true;
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
    // Read from MetalBridge networkBuffer (with packet loss simulation)
    // Generate test pattern with occasional dropouts
    static float phase = 0.0f;
    static int dropoutCounter = 0;
    
    for (size_t i = 0; i < displayBuffer.size(); ++i) {
        if (dropoutCounter > 0) {
            displayBuffer[i] = 0.0f; // Simulate packet loss
            dropoutCounter--;
        } else {
            displayBuffer[i] = 0.4f * std::sin(phase + i * 0.15f);
            if ((i % 50) == 0 && (rand() % 100) < 5) { // 5% chance of dropout
                dropoutCounter = 5; // 5 sample dropout
            }
        }
    }
    phase += 0.08f;
    isActive = true;
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