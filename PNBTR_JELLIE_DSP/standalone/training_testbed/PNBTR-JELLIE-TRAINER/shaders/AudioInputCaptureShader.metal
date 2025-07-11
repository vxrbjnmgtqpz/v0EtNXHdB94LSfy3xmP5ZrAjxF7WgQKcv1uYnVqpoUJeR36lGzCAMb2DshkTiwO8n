#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Audio Input Capture Shader
// Stage 1: Record-armed audio capture with gain control
//==============================================================================

kernel void AudioInputCaptureShader(device const float* inputBuffer [[buffer(0)]],
                                    device float* outputBuffer [[buffer(1)]],
                                    constant float& gainParam [[buffer(2)]],
                                    constant bool& recordArmed [[buffer(3)]],
                                    uint index [[thread_position_in_grid]]) {
    
    // Only process if record armed
    if (!recordArmed) {
        outputBuffer[index] = 0.0f;
        return;
    }
    
    // Apply gain and copy to output
    float sample = inputBuffer[index];
    sample *= gainParam;
    
    // Soft clipping to prevent distortion
    if (sample > 1.0f) {
        sample = 1.0f;
    } else if (sample < -1.0f) {
        sample = -1.0f;
    }
    
    outputBuffer[index] = sample;
} 