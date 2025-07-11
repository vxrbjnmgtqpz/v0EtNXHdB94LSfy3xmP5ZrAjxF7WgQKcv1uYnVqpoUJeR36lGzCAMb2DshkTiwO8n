/*
  ==============================================================================

    MetalDrawableTrack.h
    Created: GPU-native waveform rendering component

    Renders live waveform data entirely on the GPU using Metal compute shaders:
    - No CPU buffer reads
    - No AudioVisualiserComponent redraw loops  
    - Just mapped memory, GPU compute, and fast Metal draw calls
    - Eliminates latency and copy bottlenecks

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#else
typedef struct objc_object* id;
#endif

class MetalAudioTrack;

//==============================================================================
class MetalDrawableTrack : public juce::Component, private juce::Timer
{
public:
    MetalDrawableTrack(id device, id sourceBuffer, int bufferLength);
    ~MetalDrawableTrack() override;

    //==============================================================================
    // Component interface
    void paint(juce::Graphics& g) override;
    void resized() override;

    //==============================================================================
    // Configuration
    void setRefreshRate(int hz) { startTimerHz(hz); }
    void setWaveformColor(juce::Colour color) { waveformColor = color; }
    void setSamplesPerPixel(int samples) { samplesPerPixel = samples; }
    
    //==============================================================================
    // GPU rendering control
    void enableGPURendering(bool enable) { useGPURendering = enable; }
    bool isGPURenderingEnabled() const { return useGPURendering; }

private:
    //==============================================================================
    // Timer callback for updates
    void timerCallback() override;
    
    //==============================================================================
    // GPU rendering pipeline
    void dispatchWaveformShader();
    void renderWaveformTexture();
    void copyTextureToImage();
    
    //==============================================================================
    // Fallback CPU rendering
    void renderWaveformCPU();
    
    //==============================================================================
    // Metal objects
    id metalDevice;
    id audioBuffer;
    id commandQueue;
    id waveformTexture;
    id computePipeline;
    
    //==============================================================================
    // Configuration
    int bufferSize;
    int samplesPerPixel = 128;
    bool useGPURendering = true;
    juce::Colour waveformColor = juce::Colours::lime;
    
    //==============================================================================
    // CPU fallback rendering
    juce::Image previewImage;
    std::vector<float> cpuWaveformData;
    
    //==============================================================================
    // Internal methods
    bool initializeMetalPipeline();
    void createWaveformTexture();
    void loadComputeShader();
    void updatePreviewImage();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MetalDrawableTrack)
}; 