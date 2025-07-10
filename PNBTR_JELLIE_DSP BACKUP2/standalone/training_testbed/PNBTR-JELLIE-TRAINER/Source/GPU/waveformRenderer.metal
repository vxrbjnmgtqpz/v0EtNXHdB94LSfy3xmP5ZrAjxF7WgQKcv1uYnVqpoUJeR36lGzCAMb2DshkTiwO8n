//==============================================================================
//
//  waveformRenderer.metal
//  Created: GPU-native waveform rendering compute shader
//
//  Renders live waveform data entirely on the GPU:
//  - Reads audio samples from shared Metal buffer
//  - Calculates peak/trough per pixel column
//  - Writes waveform lines into 1D texture
//  - Eliminates CPU roundtrip for visualization
//
//==============================================================================

#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Waveform rendering kernel
// Reads a window of samples from audioBuffer, finds peak/trough per pixel,
// and writes waveform amplitude into waveformTexture
//==============================================================================

kernel void waveformRenderer(const device float* audioBuffer [[buffer(0)]],
                            texture1d<float, access::write> waveformTexture [[texture(0)]],
                            constant uint& samplesPerPixel [[buffer(1)]],
                            constant uint& bufferLength [[buffer(2)]],
                            uint id [[thread_position_in_grid]])
{
    // Ensure we don't exceed texture bounds
    if (id >= waveformTexture.get_width()) {
        return;
    }
    
    // Calculate sample range for this pixel
    const uint startSample = id * samplesPerPixel;
    const uint endSample = min(startSample + samplesPerPixel, bufferLength);
    
    // Find peak and trough in this sample range
    float maxSample = -1.0f;
    float minSample = 1.0f;
    
    for (uint i = startSample; i < endSample; ++i) {
        const float sample = audioBuffer[i];
        maxSample = fmax(maxSample, sample);
        minSample = fmin(minSample, sample);
    }
    
    // Calculate waveform amplitude (peak-to-peak)
    const float amplitude = (maxSample - minSample) * 0.5f;
    
    // Write amplitude to texture
    waveformTexture.write(amplitude, id);
}

//==============================================================================
// Stereo waveform rendering kernel
// Renders stereo audio with left/right channel separation
//==============================================================================

kernel void stereoWaveformRenderer(const device float* audioBuffer [[buffer(0)]],
                                  texture2d<float, access::write> waveformTexture [[texture(0)]],
                                  constant uint& samplesPerPixel [[buffer(1)]],
                                  constant uint& bufferLength [[buffer(2)]],
                                  uint2 id [[thread_position_in_grid]])
{
    // Ensure we don't exceed texture bounds
    if (id.x >= waveformTexture.get_width() || id.y >= waveformTexture.get_height()) {
        return;
    }
    
    const uint channel = id.y; // 0 = left, 1 = right
    const uint pixelX = id.x;
    
    // Calculate sample range for this pixel
    const uint startSample = pixelX * samplesPerPixel * 2; // Stereo interleaved
    const uint endSample = min(startSample + samplesPerPixel * 2, bufferLength);
    
    // Find peak and trough for this channel
    float maxSample = -1.0f;
    float minSample = 1.0f;
    
    for (uint i = startSample + channel; i < endSample; i += 2) {
        const float sample = audioBuffer[i];
        maxSample = fmax(maxSample, sample);
        minSample = fmin(minSample, sample);
    }
    
    // Calculate waveform amplitude
    const float amplitude = (maxSample - minSample) * 0.5f;
    
    // Write amplitude to texture
    waveformTexture.write(amplitude, id);
}

//==============================================================================
// Colored waveform rendering kernel
// Renders waveform with color coding based on amplitude
//==============================================================================

kernel void coloredWaveformRenderer(const device float* audioBuffer [[buffer(0)]],
                                   texture2d<half, access::write> waveformTexture [[texture(0)]],
                                   constant uint& samplesPerPixel [[buffer(1)]],
                                   constant uint& bufferLength [[buffer(2)]],
                                   constant float4& baseColor [[buffer(3)]],
                                   uint2 id [[thread_position_in_grid]])
{
    // Ensure we don't exceed texture bounds
    if (id.x >= waveformTexture.get_width() || id.y >= waveformTexture.get_height()) {
        return;
    }
    
    // Calculate sample range for this pixel
    const uint startSample = id.x * samplesPerPixel;
    const uint endSample = min(startSample + samplesPerPixel, bufferLength);
    
    // Find peak and trough in this sample range
    float maxSample = -1.0f;
    float minSample = 1.0f;
    float rms = 0.0f;
    
    for (uint i = startSample; i < endSample; ++i) {
        const float sample = audioBuffer[i];
        maxSample = fmax(maxSample, sample);
        minSample = fmin(minSample, sample);
        rms += sample * sample;
    }
    
    // Calculate metrics
    rms = sqrt(rms / float(endSample - startSample));
    
    // Calculate intensity based on RMS and clipping
    half intensity = half(0.3f + 0.7f * rms);
    
    // Red tint for clipping
    if (maxSample > 0.95f || minSample < -0.95f) {
        intensity = half(1.0f); // Full intensity for clipping
    }
    
    // Write intensity to texture (using single channel)
    waveformTexture.write(intensity, id);
}

//==============================================================================
// Packet loss visualization kernel
// Renders waveform with packet loss indicators
//==============================================================================

kernel void packetLossWaveformRenderer(const device float* audioBuffer [[buffer(0)]],
                                      const device bool* packetLossMap [[buffer(1)]],
                                      texture2d<half, access::write> waveformTexture [[texture(0)]],
                                      constant uint& samplesPerPixel [[buffer(2)]],
                                      constant uint& bufferLength [[buffer(3)]],
                                      constant uint& packetSize [[buffer(4)]],
                                      uint2 id [[thread_position_in_grid]])
{
    // Ensure we don't exceed texture bounds
    if (id.x >= waveformTexture.get_width() || id.y >= waveformTexture.get_height()) {
        return;
    }
    
    // Calculate sample range for this pixel
    const uint startSample = id.x * samplesPerPixel;
    const uint endSample = min(startSample + samplesPerPixel, bufferLength);
    
    // Find peak and trough in this sample range
    float maxSample = -1.0f;
    float minSample = 1.0f;
    bool hasPacketLoss = false;
    
    for (uint i = startSample; i < endSample; ++i) {
        const float sample = audioBuffer[i];
        maxSample = fmax(maxSample, sample);
        minSample = fmin(minSample, sample);
        
        // Check if this sample is in a lost packet
        const uint packetIndex = i / packetSize;
        if (packetLossMap[packetIndex]) {
            hasPacketLoss = true;
        }
    }
    
    // Calculate waveform amplitude
    const float amplitude = (maxSample - minSample) * 0.5f;
    
    // Encode packet loss information in the intensity
    half intensity;
    
    if (hasPacketLoss) {
        // High intensity for packet loss areas
        intensity = half(amplitude + 0.5f);
    } else {
        // Normal intensity for good audio
        intensity = half(amplitude);
    }
    
    // Write intensity to texture
    waveformTexture.write(intensity, id);
} 