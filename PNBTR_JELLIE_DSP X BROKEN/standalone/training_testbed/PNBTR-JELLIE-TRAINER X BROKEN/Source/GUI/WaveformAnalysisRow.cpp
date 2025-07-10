#include "WaveformAnalysisRow.h"
#include "../DSP/PNBTRTrainer.h"
#include "../GPU/MetalBridge.h"
#include <cmath>

//==============================================================================
WaveformAnalysisRow::WaveformAnalysisRow()
    : originalSpectralImage(juce::Image::ARGB, 
                           SpectralAnalysisConfig::TEXTURE_WIDTH, 
                           SpectralAnalysisConfig::TEXTURE_HEIGHT, 
                           true)
    , reconstructedSpectralImage(juce::Image::ARGB, 
                                SpectralAnalysisConfig::TEXTURE_WIDTH, 
                                SpectralAnalysisConfig::TEXTURE_HEIGHT, 
                                true)
{
    // 🎮 VIDEO GAME ENGINE RULE: TRANSPORT BAR PRIORITY #1
    // Metal initialization DISABLED to maintain game loop stability
    // See: VIDEO_GAME_ENGINE_RULES.md for proper deferred init protocol
    // TODO: Re-enable with async Metal init after transport bar testing
    spectralBridge = nullptr; // Disabled for game loop stability
    metalInitialized = true; // Skip Metal init to prevent main thread blocking
}

WaveformAnalysisRow::~WaveformAnalysisRow() = default;

//==============================================================================
void WaveformAnalysisRow::setTrainer(PNBTRTrainer* trainerPtr)
{
    trainer = trainerPtr;
    if (trainer)
        startTimerHz(30); // 30 Hz for smooth GPU-native updates
    else
        stopTimer();
}

void WaveformAnalysisRow::timerCallback()
{
    // SIMPLIFIED: Just basic updates, no Metal for now
    // updateGPUSpectralData(); // Disabled until transport bar is stable
}

void WaveformAnalysisRow::updateGPUSpectralData()
{
    if (!trainer || !spectralBridge || !spectralBridge->isInitialized()) return;
    
    // Get audio data from trainer
    std::vector<float> originalData(spectralBufferSize);
    std::vector<float> reconstructedData(spectralBufferSize * 2); // Stereo
    
    trainer->getRecordedBuffer(originalData.data(), spectralBufferSize, 0);
    trainer->getLatestOscOutput(reconstructedData.data(), spectralBufferSize);
    
    // Convert stereo to mono for reconstructed data
    std::vector<float> reconstructedMono(spectralBufferSize);
    for (int i = 0; i < spectralBufferSize; ++i) {
        reconstructedMono[i] = (reconstructedData[i * 2] + reconstructedData[i * 2 + 1]) * 0.5f;
    }
    
    // Update pulse animation
    pulseTime += 0.033f; // ~30 FPS
    spectralConfig.pulse = std::sin(pulseTime * 2.0f) * 0.5f + 0.5f;
    spectralBridge->setConfig(spectralConfig);
    
    // Process through GPU-native Metal pipeline
    spectralBridge->processAudioBuffer(originalData.data(), spectralBufferSize, true);
    spectralBridge->processAudioBuffer(reconstructedMono.data(), spectralBufferSize, false);
    
    // Update spectral textures
    spectralBridge->updateSpectralTexture(true);
    spectralBridge->updateSpectralTexture(false);
    
    // Render Metal textures to JUCE images
    spectralBridge->renderToJUCEImage(originalSpectralImage, true);
    spectralBridge->renderToJUCEImage(reconstructedSpectralImage, false);
    
    repaint();
}

//==============================================================================
void WaveformAnalysisRow::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    
    auto bounds = getLocalBounds().reduced(8);
    int halfWidth = bounds.getWidth() / 2 - 4;
    
    // Left: Original GPU-native spectral waveform
    auto leftBounds = bounds.removeFromLeft(halfWidth);
    drawGPUSpectralWaveform(g, leftBounds, true);
    
    bounds.removeFromLeft(8); // Gap between panels
    
    // Right: Reconstructed GPU-native spectral waveform  
    drawGPUSpectralWaveform(g, bounds, false);
}

void WaveformAnalysisRow::drawGPUSpectralWaveform(juce::Graphics& g, juce::Rectangle<int> bounds, bool isOriginal)
{
    // Draw border and title
    g.setColour(juce::Colours::darkgrey);
    g.drawRect(bounds, 1);
    
    auto titleArea = bounds.removeFromTop(20);
    g.setColour(juce::Colours::white);
    g.setFont(14.0f);
    g.drawText(isOriginal ? "Original GPU Spectral Waveform" : "Reconstructed GPU Spectral Waveform", 
               titleArea, juce::Justification::centred);
    
    auto waveformArea = bounds.reduced(4);
    if (waveformArea.getWidth() <= 0 || waveformArea.getHeight() <= 0) return;
    
    // Render GPU-native Metal texture
    renderMetalTextureToGraphics(g, waveformArea, isOriginal);
    
    // Draw center line
    g.setColour(juce::Colours::grey.withAlpha(0.3f));
    float centerY = waveformArea.getCentreY();
    g.drawHorizontalLine((int)centerY, (float)waveformArea.getX(), 
                        (float)(waveformArea.getX() + waveformArea.getWidth()));
}

void WaveformAnalysisRow::renderMetalTextureToGraphics(juce::Graphics& g, juce::Rectangle<int> bounds, bool isOriginal)
{
    if (!spectralBridge || !spectralBridge->isInitialized()) {
        // Fallback: Draw simple status
        g.setColour(juce::Colours::red);
        g.drawText("GPU Spectral Analysis Not Available", bounds, juce::Justification::centred);
        return;
    }
    
    // Get the appropriate JUCE image (converted from Metal texture)
    const juce::Image& spectralImage = isOriginal ? originalSpectralImage : reconstructedSpectralImage;
    
    if (spectralImage.isValid()) {
        // Scale and draw the GPU-rendered spectral image
        g.drawImage(spectralImage, bounds.toFloat(), juce::RectanglePlacement::stretchToFit);
        
        // Add GPU-native branding
        g.setColour(juce::Colours::lime.withAlpha(0.7f));
        g.setFont(10.0f);
        g.drawText("GPU-Native Metal FFT", bounds.removeFromBottom(15), juce::Justification::centredRight);
    } else {
        // Draw placeholder
        g.setColour(juce::Colours::orange);
        g.drawText("Initializing GPU Pipeline...", bounds, juce::Justification::centred);
    }
}

void WaveformAnalysisRow::resized()
{
    // No child components to resize - GPU textures handle their own sizing
}
