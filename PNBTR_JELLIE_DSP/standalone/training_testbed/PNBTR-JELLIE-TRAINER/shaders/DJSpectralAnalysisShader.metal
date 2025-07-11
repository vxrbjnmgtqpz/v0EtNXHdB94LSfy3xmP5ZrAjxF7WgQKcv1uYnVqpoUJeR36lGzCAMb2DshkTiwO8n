#include <metal_stdlib>
using namespace metal;

//==============================================================================
// DJ Spectral Analysis Shader
// Stage 3: Real-time FFT with color mapping
//==============================================================================

struct SpectralUniforms {
    uint bufferSize;
    float sampleRate;
};

kernel void DJSpectralAnalysisShader(device const float* inputBuffer [[buffer(0)]],
                                     device float* spectrumBuffer [[buffer(1)]],
                                     constant SpectralUniforms& uniforms [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]) {
    
    // Simple spectral analysis (simplified FFT for real-time performance)
    // This is a basic implementation - real FFT would be more complex
    
    if (index >= uniforms.bufferSize / 2) return; // Only calculate positive frequencies
    
    float frequency = (index * uniforms.sampleRate) / uniforms.bufferSize;
    float magnitude = 0.0f;
    
    // Simple DFT calculation for frequency bin
    float real = 0.0f;
    float imag = 0.0f;
    
    for (int n = 0; n < uniforms.bufferSize; n++) {
        float angle = -2.0f * M_PI_F * index * n / uniforms.bufferSize;
        real += inputBuffer[n] * cos(angle);
        imag += inputBuffer[n] * sin(angle);
    }
    
    // Calculate magnitude
    magnitude = sqrt(real * real + imag * imag) / uniforms.bufferSize;
    
    // Apply logarithmic scaling for DJ-style visualization
    magnitude = 20.0f * log10(max(magnitude, 0.0001f));
    
    // Normalize to 0-1 range for visualization
    magnitude = clamp((magnitude + 60.0f) / 60.0f, 0.0f, 1.0f);
    
    spectrumBuffer[index] = magnitude;
} 