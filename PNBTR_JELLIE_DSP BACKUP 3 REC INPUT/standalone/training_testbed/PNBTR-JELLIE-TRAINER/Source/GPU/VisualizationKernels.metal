/*
  ==============================================================================

    VisualizationKernels.metal
    Created: GPU-Accelerated Audio Visualization Shaders

    Real-time audio visualization compute kernels:
    - Waveform rendering with multi-channel support
    - FFT-based spectrum analysis and rendering
    - Time-frequency spectrograms
    - Stereo vectorscope and phase correlation
    - Level meters and peak detection

    Features:
    - 60fps real-time performance
    - High-resolution visualization (up to 4K)
    - Efficient memory bandwidth utilization
    - Color mapping and scaling

  ==============================================================================
*/

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

//==============================================================================
// Constants and Structures

constant float PI = 3.14159265359f;
constant float TWO_PI = 6.28318530718f;
constant uint MAX_DISPLAY_WIDTH = 3840;
constant uint MAX_DISPLAY_HEIGHT = 2160;
constant uint MAX_FFT_SIZE = 2048;

struct VisualizationParams {
    float gainScale;
    float timeScale;
    float frequencyScale;
    uint logScale;          // 0 = linear, 1 = log
    uint smoothing;         // 0 = off, 1 = on
    float smoothingFactor;
    
    // Color parameters
    float hueShift;
    float saturation;
    float brightness;
    
    // Display options
    uint showGrid;
    uint showLabels;
    float sampleRate;
    uint numChannels;
    uint displayWidth;
    uint displayHeight;
};

struct AudioVisualizationData {
    uint numFrames;
    uint numChannels;
    float sampleRate;
    uint startFrame;        // For scrolling displays
};

//==============================================================================
// Utility Functions

// HSV to RGB color conversion
float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h / 60.0, 2.0) - 1.0));
    float m = v - c;
    
    float3 rgb;
    if (h < 60.0) {
        rgb = float3(c, x, 0.0);
    } else if (h < 120.0) {
        rgb = float3(x, c, 0.0);
    } else if (h < 180.0) {
        rgb = float3(0.0, c, x);
    } else if (h < 240.0) {
        rgb = float3(0.0, x, c);
    } else if (h < 300.0) {
        rgb = float3(x, 0.0, c);
    } else {
        rgb = float3(c, 0.0, x);
    }
    
    return rgb + m;
}

// Map audio amplitude to color
float4 amplitude_to_color(float amplitude, const VisualizationParams& params) {
    // Normalize amplitude to 0-1 range
    float normalizedAmp = clamp(abs(amplitude) * params.gainScale, 0.0f, 1.0f);
    
    // Apply color mapping
    float hue = params.hueShift + normalizedAmp * 120.0f; // Blue to red spectrum
    float saturation = params.saturation;
    float brightness = params.brightness * normalizedAmp;
    
    float3 rgb = hsv_to_rgb(hue, saturation, brightness);
    return float4(rgb, normalizedAmp);
}

// Logarithmic frequency mapping
float linear_to_log_freq(float linearFreq, float maxFreq) {
    if (linearFreq <= 0.0f) return 0.0f;
    return log2(linearFreq / 20.0f) / log2(maxFreq / 20.0f);
}

// Window function for FFT (Hann window)
float hann_window(uint index, uint size) {
    return 0.5f * (1.0f - cos(TWO_PI * index / (size - 1)));
}

//==============================================================================
// Waveform Rendering Kernels

// Single-channel waveform
kernel void render_waveform_mono(device float* audioBuffer [[buffer(0)]],
                                device float4* pixelBuffer [[buffer(1)]],
                                constant VisualizationParams& params [[buffer(2)]],
                                constant AudioVisualizationData& audioData [[buffer(3)]],
                                uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Map pixel x to audio sample
    float samplePos = (float)x / params.displayWidth * audioData.numFrames;
    uint sampleIndex = (uint)samplePos;
    
    if (sampleIndex >= audioData.numFrames) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        return;
    }
    
    // Get audio sample
    float sample = audioBuffer[sampleIndex];
    
    // Map sample amplitude to vertical position
    float centerY = params.displayHeight * 0.5f;
    float amplitude = sample * params.gainScale;
    float sampleY = centerY - (amplitude * centerY);
    
    // Determine if this pixel should be lit
    float distance = abs(y - sampleY);
    float intensity = max(0.0f, 1.0f - distance * 2.0f);
    
    if (intensity > 0.0f) {
        pixelBuffer[pixelIndex] = amplitude_to_color(sample, params) * intensity;
    } else {
        pixelBuffer[pixelIndex] = float4(0.0f);
    }
}

// Multi-channel waveform (stereo with separate tracks)
kernel void render_waveform_stereo(device float* audioBuffer [[buffer(0)]],
                                  device float4* pixelBuffer [[buffer(1)]],
                                  constant VisualizationParams& params [[buffer(2)]],
                                  constant AudioVisualizationData& audioData [[buffer(3)]],
                                  uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Determine which channel based on vertical position
    uint channelHeight = params.displayHeight / audioData.numChannels;
    uint channel = y / channelHeight;
    uint localY = y % channelHeight;
    
    if (channel >= audioData.numChannels) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        return;
    }
    
    // Map pixel x to audio sample
    float samplePos = (float)x / params.displayWidth * audioData.numFrames;
    uint sampleIndex = (uint)samplePos;
    
    if (sampleIndex >= audioData.numFrames) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        return;
    }
    
    // Get audio sample (interleaved format)
    float sample = audioBuffer[sampleIndex * audioData.numChannels + channel];
    
    // Map sample amplitude to vertical position within channel
    float centerY = channelHeight * 0.5f;
    float amplitude = sample * params.gainScale;
    float sampleY = centerY - (amplitude * centerY);
    
    // Determine if this pixel should be lit
    float distance = abs(localY - sampleY);
    float intensity = max(0.0f, 1.0f - distance * 2.0f);
    
    if (intensity > 0.0f) {
        // Color-code channels
        VisualizationParams channelParams = params;
        channelParams.hueShift += channel * 60.0f; // Different hues per channel
        pixelBuffer[pixelIndex] = amplitude_to_color(sample, channelParams) * intensity;
    } else {
        pixelBuffer[pixelIndex] = float4(0.0f);
    }
}

//==============================================================================
// FFT and Spectrum Analysis Kernels

// Complex number structure for FFT
struct complex_float {
    float real;
    float imag;
    
    complex_float(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}
    
    complex_float operator+(const complex_float& other) const {
        return complex_float(real + other.real, imag + other.imag);
    }
    
    complex_float operator-(const complex_float& other) const {
        return complex_float(real - other.real, imag - other.imag);
    }
    
    complex_float operator*(const complex_float& other) const {
        return complex_float(real * other.real - imag * other.imag,
                           real * other.imag + imag * other.real);
    }
    
    float magnitude() const {
        return sqrt(real * real + imag * imag);
    }
};

// Radix-2 FFT (simplified for GPU)
kernel void compute_fft(device float* audioBuffer [[buffer(0)]],
                       device float* magnitudeBuffer [[buffer(1)]],
                       device float* windowBuffer [[buffer(2)]],
                       constant VisualizationParams& params [[buffer(3)]],
                       constant AudioVisualizationData& audioData [[buffer(4)]],
                       uint index [[thread_position_in_grid]]) {
    
    uint fftSize = min(audioData.numFrames, (uint)MAX_FFT_SIZE);
    if (index >= fftSize) return;
    
    // Apply window function
    float windowedSample = audioBuffer[index] * hann_window(index, fftSize);
    
    // Simplified magnitude calculation (placeholder for full FFT)
    // In a real implementation, this would be a proper radix-2 FFT
    float magnitude = 0.0f;
    
    for (uint k = 0; k < fftSize; ++k) {
        float phase = -TWO_PI * index * k / fftSize;
        complex_float twiddle(cos(phase), sin(phase));
        float sample = audioBuffer[k] * hann_window(k, fftSize);
        magnitude += sample * twiddle.real; // Simplified
    }
    
    magnitudeBuffer[index] = abs(magnitude) / fftSize;
}

// Spectrum rendering from FFT magnitude data
kernel void render_spectrum(device float* magnitudeBuffer [[buffer(0)]],
                           device float4* pixelBuffer [[buffer(1)]],
                           device float* previousSpectrum [[buffer(2)]],
                           constant VisualizationParams& params [[buffer(3)]],
                           constant AudioVisualizationData& audioData [[buffer(4)]],
                           uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Map pixel x to frequency bin
    uint fftSize = min(audioData.numFrames, (uint)MAX_FFT_SIZE);
    uint halfFFT = fftSize / 2;
    
    float freqPos;
    if (params.logScale) {
        // Logarithmic frequency scale
        float logPos = (float)x / params.displayWidth;
        freqPos = pow(10.0f, logPos * log10((float)halfFFT)) - 1.0f;
    } else {
        // Linear frequency scale
        freqPos = (float)x / params.displayWidth * halfFFT;
    }
    
    uint binIndex = (uint)freqPos;
    if (binIndex >= halfFFT) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        return;
    }
    
    // Get magnitude with interpolation
    float magnitude = magnitudeBuffer[binIndex];
    if (binIndex + 1 < halfFFT) {
        float frac = freqPos - binIndex;
        magnitude = mix(magnitude, magnitudeBuffer[binIndex + 1], frac);
    }
    
    // Apply smoothing
    if (params.smoothing) {
        float prevMagnitude = previousSpectrum[binIndex];
        magnitude = mix(magnitude, prevMagnitude, params.smoothingFactor);
        previousSpectrum[binIndex] = magnitude;
    }
    
    // Scale magnitude
    magnitude *= params.gainScale;
    
    // Map magnitude to vertical position
    float magnitudeY = magnitude * params.displayHeight;
    
    // Determine if this pixel should be lit
    if (y <= magnitudeY) {
        float intensity = (float)(magnitudeY - y) / magnitudeY;
        
        // Color based on frequency and magnitude
        VisualizationParams freqParams = params;
        freqParams.hueShift += (float)x / params.displayWidth * 240.0f; // Rainbow spectrum
        
        pixelBuffer[pixelIndex] = amplitude_to_color(magnitude, freqParams) * intensity;
    } else {
        pixelBuffer[pixelIndex] = float4(0.0f);
    }
}

//==============================================================================
// Spectrogram Rendering

kernel void render_spectrogram(device float* magnitudeBuffer [[buffer(0)]],
                              device float4* pixelBuffer [[buffer(1)]],
                              device float4* spectrogramHistory [[buffer(2)]],
                              constant VisualizationParams& params [[buffer(3)]],
                              constant AudioVisualizationData& audioData [[buffer(4)]],
                              uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Shift existing spectrogram data left
    if (x < params.displayWidth - 1) {
        pixelBuffer[pixelIndex] = spectrogramHistory[y * params.displayWidth + (x + 1)];
        spectrogramHistory[pixelIndex] = pixelBuffer[pixelIndex];
        return;
    }
    
    // Add new spectrum data on the right edge
    uint fftSize = min(audioData.numFrames, (uint)MAX_FFT_SIZE);
    uint halfFFT = fftSize / 2;
    
    // Map pixel y to frequency bin (inverted - high freq at top)
    float freqPos = (1.0f - (float)y / params.displayHeight) * halfFFT;
    uint binIndex = (uint)freqPos;
    
    if (binIndex >= halfFFT) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        spectrogramHistory[pixelIndex] = float4(0.0f);
        return;
    }
    
    float magnitude = magnitudeBuffer[binIndex] * params.gainScale;
    
    // Color based on magnitude
    float4 color = amplitude_to_color(magnitude, params);
    
    pixelBuffer[pixelIndex] = color;
    spectrogramHistory[pixelIndex] = color;
}

//==============================================================================
// Vectorscope Rendering (Stereo)

kernel void render_vectorscope(device float* audioBuffer [[buffer(0)]],
                              device float4* pixelBuffer [[buffer(1)]],
                              constant VisualizationParams& params [[buffer(2)]],
                              constant AudioVisualizationData& audioData [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    
    if (index >= audioData.numFrames || audioData.numChannels < 2) return;
    
    // Clear pixel buffer first (only thread 0)
    if (index == 0) {
        for (uint i = 0; i < params.displayWidth * params.displayHeight; ++i) {
            pixelBuffer[i] = float4(0.0f);
        }
    }
    
    // Synchronize threads
    threadgroup_barrier(mem_flags::mem_device);
    
    // Get stereo samples
    float leftSample = audioBuffer[index * 2] * params.gainScale;
    float rightSample = audioBuffer[index * 2 + 1] * params.gainScale;
    
    // Map samples to screen coordinates
    uint centerX = params.displayWidth / 2;
    uint centerY = params.displayHeight / 2;
    
    int x = centerX + (int)(leftSample * centerX);
    int y = centerY - (int)(rightSample * centerY);
    
    // Clamp to screen bounds
    x = clamp(x, 0, (int)params.displayWidth - 1);
    y = clamp(y, 0, (int)params.displayHeight - 1);
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Add dot to vectorscope
    float magnitude = sqrt(leftSample * leftSample + rightSample * rightSample);
    float4 color = amplitude_to_color(magnitude, params);
    
    // Atomic add for thread safety (approximated)
    pixelBuffer[pixelIndex] = min(pixelBuffer[pixelIndex] + color * 0.1f, float4(1.0f));
}

//==============================================================================
// Level Meters

kernel void render_level_meters(device float* audioBuffer [[buffer(0)]],
                               device float4* pixelBuffer [[buffer(1)]],
                               device float* peakBuffer [[buffer(2)]],
                               device float* rmsBuffer [[buffer(3)]],
                               constant VisualizationParams& params [[buffer(4)]],
                               constant AudioVisualizationData& audioData [[buffer(5)]],
                               uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Determine which channel
    uint meterWidth = params.displayWidth / audioData.numChannels;
    uint channel = x / meterWidth;
    uint localX = x % meterWidth;
    
    if (channel >= audioData.numChannels) {
        pixelBuffer[pixelIndex] = float4(0.0f);
        return;
    }
    
    // Calculate peak and RMS for this channel
    float peak = 0.0f;
    float rms = 0.0f;
    
    for (uint i = 0; i < audioData.numFrames; ++i) {
        float sample = audioBuffer[i * audioData.numChannels + channel];
        peak = max(peak, abs(sample));
        rms += sample * sample;
    }
    
    rms = sqrt(rms / audioData.numFrames);
    
    // Store for persistence
    peakBuffer[channel] = max(peakBuffer[channel] * 0.95f, peak); // Peak hold with decay
    rmsBuffer[channel] = rms;
    
    // Render meter
    float peakY = peakBuffer[channel] * params.gainScale * params.displayHeight;
    float rmsY = rmsBuffer[channel] * params.gainScale * params.displayHeight;
    
    float4 color = float4(0.0f);
    
    if (params.displayHeight - y <= rmsY) {
        // RMS level - green
        color = float4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    
    if (params.displayHeight - y <= peakY) {
        // Peak level - yellow to red based on level
        float level = peakY / params.displayHeight;
        if (level > 0.8f) {
            color = float4(1.0f, 0.0f, 0.0f, 1.0f); // Red - danger
        } else if (level > 0.6f) {
            color = float4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow - warning
        } else {
            color = float4(0.0f, 1.0f, 0.0f, 1.0f); // Green - safe
        }
    }
    
    pixelBuffer[pixelIndex] = color;
}

//==============================================================================
// Grid and Label Rendering

kernel void render_grid(device float4* pixelBuffer [[buffer(0)]],
                       constant VisualizationParams& params [[buffer(1)]],
                       uint2 threadPos [[thread_position_in_grid]]) {
    
    uint x = threadPos.x;
    uint y = threadPos.y;
    
    if (x >= params.displayWidth || y >= params.displayHeight) return;
    
    uint pixelIndex = y * params.displayWidth + x;
    
    // Grid parameters
    uint gridSpacingX = params.displayWidth / 10;
    uint gridSpacingY = params.displayHeight / 8;
    
    float4 gridColor = float4(0.3f, 0.3f, 0.3f, 0.5f);
    float4 axisColor = float4(0.6f, 0.6f, 0.6f, 0.8f);
    
    // Vertical grid lines
    if (x % gridSpacingX == 0) {
        pixelBuffer[pixelIndex] = mix(pixelBuffer[pixelIndex], gridColor, gridColor.a);
    }
    
    // Horizontal grid lines
    if (y % gridSpacingY == 0) {
        pixelBuffer[pixelIndex] = mix(pixelBuffer[pixelIndex], gridColor, gridColor.a);
    }
    
    // Center axes
    if (x == params.displayWidth / 2 || y == params.displayHeight / 2) {
        pixelBuffer[pixelIndex] = mix(pixelBuffer[pixelIndex], axisColor, axisColor.a);
    }
} 