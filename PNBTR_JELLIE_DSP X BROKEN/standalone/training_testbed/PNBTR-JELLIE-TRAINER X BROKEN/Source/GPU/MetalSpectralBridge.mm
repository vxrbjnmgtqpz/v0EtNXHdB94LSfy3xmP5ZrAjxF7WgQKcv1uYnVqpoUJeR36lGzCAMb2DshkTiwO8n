/*
  ==============================================================================

    MetalSpectralBridge.mm
    Created: GPU-native Metal bridge implementation for spectral analysis

  ==============================================================================
*/

#include "MetalSpectralBridge.h"
#include "MetalBridge.h"
#include <iostream>

// Metal shader struct definitions (must match the Metal shader structs)
struct FFTUniforms {
    uint32_t frameOffset;
    uint32_t windowSize;
    float sampleRate;
    uint32_t armed;
};

struct VisualUniforms {
    float lowColor[4];
    float midColor[4];
    float highColor[4];
    uint32_t binCount;
    float maxMagnitude;
    float logScale;
    uint32_t armed;
    float pulse;
};

//==============================================================================
MetalSpectralBridge::MetalSpectralBridge(MetalBridge& bridge)
    : metalBridge(bridge)
    , originalSpectralData(SpectralAnalysisConfig::SPECTRUM_BINS, 0.0f)
    , reconstructedSpectralData(SpectralAnalysisConfig::SPECTRUM_BINS, 0.0f)
{
}

MetalSpectralBridge::~MetalSpectralBridge()
{
    cleanup();
}

//==============================================================================
bool MetalSpectralBridge::initialize()
{
    if (initialized) return true;

    // Get Metal device and command queue from main bridge
    device = metalBridge.getDevice();
    commandQueue = metalBridge.getCommandQueue();
    
    if (!device || !commandQueue) {
        std::cerr << "[MetalSpectralBridge] Failed to get Metal device/queue" << std::endl;
        return false;
    }

    // Create Metal library with spectral shaders
    NSError* error = nil;
    NSString* shaderSource = @R"(
        #include <metal_stdlib>
        using namespace metal;

        constant uint FFT_SIZE = 1024;

        // Include the FFT compute kernel
        kernel void fftComputeKernel(
            constant FFTUniforms& uniforms [[buffer(0)]],
            device const float* audioBuffer [[buffer(1)]],
            device float* spectrumBins [[buffer(2)]],
            device float* realPart [[buffer(3)]],
            device float* imagPart [[buffer(4)]],
            uint threadID [[thread_position_in_grid]])
        {
            if (uniforms.armed == 0) return;
            
            const uint half = FFT_SIZE / 2;
            if (threadID >= half) return;

            // Initialize working buffers
            if (threadID < uniforms.windowSize) {
                realPart[threadID] = audioBuffer[uniforms.frameOffset + threadID];
                imagPart[threadID] = 0.0;
            } else {
                realPart[threadID] = 0.0;
                imagPart[threadID] = 0.0;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Simple magnitude calculation (placeholder for full FFT)
            if (threadID < half) {
                float re = realPart[threadID];
                float im = imagPart[threadID];
                spectrumBins[threadID] = sqrt(re * re + im * im);
            }
        }

        // Spectrum visualization kernel
        kernel void spectrumVisualKernel(
            constant VisualUniforms& uniforms [[buffer(0)]],
            device const float* bins [[buffer(1)]],
            texture2d<float, access::write> outTexture [[texture(0)]],
            uint2 gid [[thread_position_in_grid]])
        {
            if (uniforms.armed == 0) {
                outTexture.write(float4(0.1, 0.1, 0.1, 1.0), gid);
                return;
            }
            
            uint textureWidth = outTexture.get_width();
            uint textureHeight = outTexture.get_height();
            
            float xNorm = float(gid.x) / float(textureWidth);
            float yNorm = float(gid.y) / float(textureHeight);
            
            // Map to frequency bin
            float logX = pow(xNorm, uniforms.logScale);
            uint bin = min(uint(logX * uniforms.binCount), uniforms.binCount - 1);
            
            // Get magnitude
            float magnitude = bins[bin] / uniforms.maxMagnitude;
            magnitude = clamp(magnitude, 0.0, 1.0);
            magnitude *= (1.0 + uniforms.pulse * 0.3);
            
            // DJ-style color mapping
            float4 baseColor;
            if (xNorm < 0.33) {
                float t = xNorm * 3.0;
                baseColor = mix(uniforms.lowColor, float4(1.0, 0.5, 0.0, 1.0), t);
            } else if (xNorm < 0.66) {
                float t = (xNorm - 0.33) * 3.0;
                baseColor = mix(float4(1.0, 0.5, 0.0, 1.0), uniforms.midColor, t);
            } else {
                float t = (xNorm - 0.66) * 3.0;
                baseColor = mix(uniforms.midColor, uniforms.highColor, t);
            }
            
            // Render spectrum bars
            float barHeight = magnitude;
            float pixelHeight = 1.0 - yNorm;
            
            if (pixelHeight <= barHeight) {
                float intensity = pixelHeight / barHeight;
                float4 color = baseColor * intensity;
                color.rgb += magnitude * 0.2;
                outTexture.write(color, gid);
            } else {
                float4 bgColor = float4(0.05, 0.05, 0.1, 1.0);
                outTexture.write(bgColor, gid);
            }
        }

        // Smoothing kernel
        kernel void smoothingKernel(
            device const float* inputBins [[buffer(0)]],
            device float* outputBins [[buffer(1)]],
            device float* smoothingBuffer [[buffer(2)]],
            constant float& smoothingFactor [[buffer(3)]],
            uint binID [[thread_position_in_grid]])
        {
            if (binID >= 512) return;
            
            float currentValue = inputBins[binID];
            float smoothedValue = smoothingBuffer[binID];
            
            smoothedValue = smoothingFactor * currentValue + (1.0 - smoothingFactor) * smoothedValue;
            
            smoothingBuffer[binID] = smoothedValue;
            outputBins[binID] = smoothedValue;
        }
    )";

    library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        std::cerr << "[MetalSpectralBridge] Failed to create shader library: " 
                  << error.localizedDescription.UTF8String << std::endl;
        return false;
    }

    // Create compute pipelines
    if (!createComputePipelines()) {
        std::cerr << "[MetalSpectralBridge] Failed to create compute pipelines" << std::endl;
        return false;
    }

    // Create buffers
    if (!createBuffers()) {
        std::cerr << "[MetalSpectralBridge] Failed to create buffers" << std::endl;
        return false;
    }

    // Create textures
    if (!createTextures()) {
        std::cerr << "[MetalSpectralBridge] Failed to create textures" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "[MetalSpectralBridge] Successfully initialized GPU-native spectral analysis" << std::endl;
    return true;
}

void MetalSpectralBridge::cleanup()
{
    if (!initialized) return;

    // Release Metal resources - ARC-compatible
    fftComputePipeline = nil;
    spectrumVisualPipeline = nil;
    smoothingPipeline = nil;
    
    originalAudioBuffer = nil;
    originalRealBuffer = nil;
    originalImagBuffer = nil;
    originalSpectrumBuffer = nil;
    originalSmoothingBuffer = nil;
    
    reconstructedAudioBuffer = nil;
    reconstructedRealBuffer = nil;
    reconstructedImagBuffer = nil;
    reconstructedSpectrumBuffer = nil;
    reconstructedSmoothingBuffer = nil;
    
    fftUniformsBuffer = nil;
    visualUniformsBuffer = nil;
    smoothingFactorBuffer = nil;
    
    originalSpectralTexture = nil;
    reconstructedSpectralTexture = nil;
    spectralTextureDescriptor = nil;
    
    library = nil;

    initialized = false;
}

//==============================================================================
bool MetalSpectralBridge::createComputePipelines()
{
    NSError* error = nil;
    
    // FFT Compute Pipeline
    id<MTLFunction> fftFunction = [library newFunctionWithName:@"fftComputeKernel"];
    if (!fftFunction) {
        std::cerr << "[MetalSpectralBridge] Failed to create FFT function" << std::endl;
        return false;
    }
    
    fftComputePipeline = [device newComputePipelineStateWithFunction:fftFunction error:&error];
    if (!fftComputePipeline) {
        std::cerr << "[MetalSpectralBridge] Failed to create FFT pipeline: " 
                  << error.localizedDescription.UTF8String << std::endl;
        return false;
    }
    
    // Spectrum Visual Pipeline
    id<MTLFunction> visualFunction = [library newFunctionWithName:@"spectrumVisualKernel"];
    if (!visualFunction) {
        std::cerr << "[MetalSpectralBridge] Failed to create visual function" << std::endl;
        return false;
    }
    
    spectrumVisualPipeline = [device newComputePipelineStateWithFunction:visualFunction error:&error];
    if (!spectrumVisualPipeline) {
        std::cerr << "[MetalSpectralBridge] Failed to create visual pipeline: " 
                  << error.localizedDescription.UTF8String << std::endl;
        return false;
    }
    
    // Smoothing Pipeline
    id<MTLFunction> smoothingFunction = [library newFunctionWithName:@"smoothingKernel"];
    if (!smoothingFunction) {
        std::cerr << "[MetalSpectralBridge] Failed to create smoothing function" << std::endl;
        return false;
    }
    
    smoothingPipeline = [device newComputePipelineStateWithFunction:smoothingFunction error:&error];
    if (!smoothingPipeline) {
        std::cerr << "[MetalSpectralBridge] Failed to create smoothing pipeline: " 
                  << error.localizedDescription.UTF8String << std::endl;
        return false;
    }
    
    return true;
}

bool MetalSpectralBridge::createBuffers()
{
    size_t audioBufferSize = SpectralAnalysisConfig::FFT_SIZE * sizeof(float);
    size_t spectrumBufferSize = SpectralAnalysisConfig::SPECTRUM_BINS * sizeof(float);
    
    // Original audio buffers
    originalAudioBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    originalRealBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    originalImagBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    originalSpectrumBuffer = [device newBufferWithLength:spectrumBufferSize options:MTLResourceStorageModeShared];
    originalSmoothingBuffer = [device newBufferWithLength:spectrumBufferSize options:MTLResourceStorageModeShared];
    
    // Reconstructed audio buffers
    reconstructedAudioBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    reconstructedRealBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    reconstructedImagBuffer = [device newBufferWithLength:audioBufferSize options:MTLResourceStorageModeShared];
    reconstructedSpectrumBuffer = [device newBufferWithLength:spectrumBufferSize options:MTLResourceStorageModeShared];
    reconstructedSmoothingBuffer = [device newBufferWithLength:spectrumBufferSize options:MTLResourceStorageModeShared];
    
    // Uniform buffers
    fftUniformsBuffer = [device newBufferWithLength:sizeof(FFTUniforms) options:MTLResourceStorageModeShared];
    visualUniformsBuffer = [device newBufferWithLength:sizeof(VisualUniforms) options:MTLResourceStorageModeShared];
    smoothingFactorBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    
    // Set initial smoothing factor
    *(float*)smoothingFactorBuffer.contents = config.smoothingFactor;
    
    return (originalAudioBuffer && originalRealBuffer && originalImagBuffer && 
            originalSpectrumBuffer && originalSmoothingBuffer &&
            reconstructedAudioBuffer && reconstructedRealBuffer && reconstructedImagBuffer &&
            reconstructedSpectrumBuffer && reconstructedSmoothingBuffer &&
            fftUniformsBuffer && visualUniformsBuffer && smoothingFactorBuffer);
}

bool MetalSpectralBridge::createTextures()
{
    spectralTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:SpectralAnalysisConfig::TEXTURE_WIDTH
                                                                                  height:SpectralAnalysisConfig::TEXTURE_HEIGHT
                                                                               mipmapped:NO];
    spectralTextureDescriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    
    originalSpectralTexture = [device newTextureWithDescriptor:spectralTextureDescriptor];
    reconstructedSpectralTexture = [device newTextureWithDescriptor:spectralTextureDescriptor];
    
    return (originalSpectralTexture && reconstructedSpectralTexture);
}

//==============================================================================
void MetalSpectralBridge::processAudioBuffer(const float* audioData, size_t numSamples, bool isOriginal)
{
    if (!initialized) return;
    
    updateAudioBuffer(audioData, numSamples, isOriginal);
    updateUniforms();
    dispatchFFTKernel(isOriginal);
    dispatchSmoothingKernel(isOriginal);
    copySpectrumToTexture(isOriginal);
}

void MetalSpectralBridge::updateSpectralTexture(bool isOriginal)
{
    if (!initialized) return;
    
    updateUniforms();
    dispatchVisualKernel(isOriginal);
}

void MetalSpectralBridge::updateAudioBuffer(const float* audioData, size_t numSamples, bool isOriginal)
{
    id<MTLBuffer> targetBuffer = isOriginal ? originalAudioBuffer : reconstructedAudioBuffer;
    
    size_t copySize = std::min(numSamples, SpectralAnalysisConfig::FFT_SIZE) * sizeof(float);
    memcpy(targetBuffer.contents, audioData, copySize);
    
    // Zero-pad if necessary
    if (numSamples < SpectralAnalysisConfig::FFT_SIZE) {
        size_t remainingBytes = (SpectralAnalysisConfig::FFT_SIZE - numSamples) * sizeof(float);
        memset((char*)targetBuffer.contents + copySize, 0, remainingBytes);
    }
}

void MetalSpectralBridge::updateUniforms()
{
    // Update FFT uniforms
    FFTUniforms* fftUniforms = (FFTUniforms*)fftUniformsBuffer.contents;
    fftUniforms->frameOffset = 0;
    fftUniforms->windowSize = SpectralAnalysisConfig::FFT_SIZE;
    fftUniforms->sampleRate = 48000.0f;
    fftUniforms->armed = config.armed ? 1 : 0;
    
    // Update visual uniforms
    VisualUniforms* visualUniforms = (VisualUniforms*)visualUniformsBuffer.contents;
    visualUniforms->lowColor[0] = config.lowColor.getFloatRed();
    visualUniforms->lowColor[1] = config.lowColor.getFloatGreen();
    visualUniforms->lowColor[2] = config.lowColor.getFloatBlue();
    visualUniforms->lowColor[3] = 1.0f;
    
    visualUniforms->midColor[0] = config.midColor.getFloatRed();
    visualUniforms->midColor[1] = config.midColor.getFloatGreen();
    visualUniforms->midColor[2] = config.midColor.getFloatBlue();
    visualUniforms->midColor[3] = 1.0f;
    
    visualUniforms->highColor[0] = config.highColor.getFloatRed();
    visualUniforms->highColor[1] = config.highColor.getFloatGreen();
    visualUniforms->highColor[2] = config.highColor.getFloatBlue();
    visualUniforms->highColor[3] = 1.0f;
    
    visualUniforms->binCount = SpectralAnalysisConfig::SPECTRUM_BINS;
    visualUniforms->maxMagnitude = config.maxMagnitude;
    visualUniforms->logScale = config.logScale;
    visualUniforms->armed = config.armed ? 1 : 0;
    visualUniforms->pulse = config.pulse;
}

void MetalSpectralBridge::dispatchFFTKernel(bool isOriginal)
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:fftComputePipeline];
    [encoder setBuffer:fftUniformsBuffer offset:0 atIndex:0];
    [encoder setBuffer:(isOriginal ? originalAudioBuffer : reconstructedAudioBuffer) offset:0 atIndex:1];
    [encoder setBuffer:(isOriginal ? originalSpectrumBuffer : reconstructedSpectrumBuffer) offset:0 atIndex:2];
    [encoder setBuffer:(isOriginal ? originalRealBuffer : reconstructedRealBuffer) offset:0 atIndex:3];
    [encoder setBuffer:(isOriginal ? originalImagBuffer : reconstructedImagBuffer) offset:0 atIndex:4];
    
    MTLSize threadGroupSize = MTLSizeMake(fftComputePipeline.maxTotalThreadsPerThreadgroup, 1, 1);
    MTLSize threadGroups = MTLSizeMake((SpectralAnalysisConfig::SPECTRUM_BINS + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1);
    
    [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
}

void MetalSpectralBridge::dispatchSmoothingKernel(bool isOriginal)
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:smoothingPipeline];
    [encoder setBuffer:(isOriginal ? originalSpectrumBuffer : reconstructedSpectrumBuffer) offset:0 atIndex:0];
    [encoder setBuffer:(isOriginal ? originalSpectrumBuffer : reconstructedSpectrumBuffer) offset:0 atIndex:1];
    [encoder setBuffer:(isOriginal ? originalSmoothingBuffer : reconstructedSmoothingBuffer) offset:0 atIndex:2];
    [encoder setBuffer:smoothingFactorBuffer offset:0 atIndex:3];
    
    MTLSize threadGroupSize = MTLSizeMake(smoothingPipeline.maxTotalThreadsPerThreadgroup, 1, 1);
    MTLSize threadGroups = MTLSizeMake((SpectralAnalysisConfig::SPECTRUM_BINS + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1);
    
    [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
}

void MetalSpectralBridge::dispatchVisualKernel(bool isOriginal)
{
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:spectrumVisualPipeline];
    [encoder setBuffer:visualUniformsBuffer offset:0 atIndex:0];
    [encoder setBuffer:(isOriginal ? originalSpectrumBuffer : reconstructedSpectrumBuffer) offset:0 atIndex:1];
    [encoder setTexture:(isOriginal ? originalSpectralTexture : reconstructedSpectralTexture) atIndex:0];
    
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadGroups = MTLSizeMake(
        (SpectralAnalysisConfig::TEXTURE_WIDTH + threadGroupSize.width - 1) / threadGroupSize.width,
        (SpectralAnalysisConfig::TEXTURE_HEIGHT + threadGroupSize.height - 1) / threadGroupSize.height,
        1
    );
    
    [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
}

void MetalSpectralBridge::copySpectrumToTexture(bool isOriginal)
{
    // Copy spectrum data from GPU buffer to CPU for JUCE integration
    id<MTLBuffer> spectrumBuffer = isOriginal ? originalSpectrumBuffer : reconstructedSpectrumBuffer;
    std::vector<float>& targetData = isOriginal ? originalSpectralData : reconstructedSpectralData;
    
    float* bufferData = (float*)spectrumBuffer.contents;
    std::copy(bufferData, bufferData + SpectralAnalysisConfig::SPECTRUM_BINS, targetData.begin());
}

//==============================================================================
void MetalSpectralBridge::setConfig(const SpectralAnalysisConfig& newConfig)
{
    config = newConfig;
    if (initialized) {
        updateUniforms();
    }
}

const float* MetalSpectralBridge::getSpectralBins(bool isOriginal) const
{
    return isOriginal ? originalSpectralData.data() : reconstructedSpectralData.data();
}

id<MTLTexture> MetalSpectralBridge::getSpectralTexture(bool isOriginal) const
{
    return isOriginal ? originalSpectralTexture : reconstructedSpectralTexture;
}

void MetalSpectralBridge::renderToJUCEImage(juce::Image& image, bool isOriginal)
{
    if (!initialized) return;
    
    // Convert Metal texture to JUCE image
    id<MTLTexture> texture = getSpectralTexture(isOriginal);
    if (!texture) return;
    
    // Get texture data
    MTLRegion region = MTLRegionMake2D(0, 0, texture.width, texture.height);
    size_t bytesPerRow = texture.width * 4; // RGBA
    size_t dataSize = bytesPerRow * texture.height;
    
    std::vector<uint8_t> textureData(dataSize);
    [texture getBytes:textureData.data() bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    
    // Convert to JUCE image
    juce::Image::BitmapData bitmapData(image, juce::Image::BitmapData::writeOnly);
    
    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            size_t textureIndex = (y * texture.width + x) * 4;
            if (textureIndex + 3 < textureData.size()) {
                uint8_t r = textureData[textureIndex];
                uint8_t g = textureData[textureIndex + 1];
                uint8_t b = textureData[textureIndex + 2];
                uint8_t a = textureData[textureIndex + 3];
                
                juce::Colour pixelColour(r, g, b, a);
                bitmapData.setPixelColour(x, y, pixelColour);
            }
        }
    }
} 