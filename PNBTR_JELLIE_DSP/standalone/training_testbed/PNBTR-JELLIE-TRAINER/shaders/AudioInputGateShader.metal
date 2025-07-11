#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Audio Input Gate Shader
// Stage 2: Noise suppression and signal detection
//==============================================================================

struct GateParams {
    float threshold;
    float ratio;
    float attack;
    float release;
};

kernel void AudioInputGateShader(device const float* inputBuffer [[buffer(0)]],
                                 device float* outputBuffer [[buffer(1)]],
                                 constant GateParams& params [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
    
    float sample = inputBuffer[index];
    float absSample = abs(sample);
    
    // Noise gate with smooth attack/release
    if (absSample < params.threshold) {
        // Below threshold - apply gate reduction
        sample *= 0.1f; // -20dB reduction
    } else {
        // Above threshold - apply gentle compression
        if (absSample > params.threshold) {
            float overThreshold = absSample - params.threshold;
            float compressedOver = overThreshold / params.ratio;
            float newAbs = params.threshold + compressedOver;
            sample = (sample >= 0.0f) ? newAbs : -newAbs;
        }
    }
    
    // Soft clipping for safety
    sample = clamp(sample, -1.0f, 1.0f);
    
    outputBuffer[index] = sample;
} 