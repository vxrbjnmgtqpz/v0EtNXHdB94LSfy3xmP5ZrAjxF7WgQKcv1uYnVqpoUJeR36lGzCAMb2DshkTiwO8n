#include <metal_stdlib>
using namespace metal;

//==============================================================================
// JDAT Encoder Shader
// JAM Digital Audio Tape - High-fidelity audio encoding
//==============================================================================

struct JELLIEUniforms {
    float sampleRate;
    bool enableJDAT;
};

kernel void JELLIEPreprocessShader(device const float* inputBuffer [[buffer(0)]],
                                   device float* outputBuffer [[buffer(1)]],
                                   constant JELLIEUniforms& uniforms [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
    
    if (!uniforms.enableJDAT) {
        outputBuffer[index] = inputBuffer[index];
        return;
    }
    
    // Get input sample
    float sample = inputBuffer[index];
    
    // JDAT processing: High-fidelity digital tape simulation
    // Apply analog warmth modeling
    float processed = sample;
    
    // Soft saturation (tape saturation modeling)
    if (processed > 0.7f) {
        processed = 0.7f + 0.3f * tanh((processed - 0.7f) * 3.0f);
    } else if (processed < -0.7f) {
        processed = -0.7f + 0.3f * tanh((processed + 0.7f) * 3.0f);
    }
    
    // High-frequency preservation (digital tape characteristic)
    float highFreq = sample - processed;
    processed += highFreq * 0.1f;
    
    // Final clipping protection
    processed = clamp(processed, -1.0f, 1.0f);
    
    outputBuffer[index] = processed;
} 