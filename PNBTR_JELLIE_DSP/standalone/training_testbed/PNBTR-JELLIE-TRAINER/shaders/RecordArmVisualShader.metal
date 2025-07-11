#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Record Arm Visual Shader
// Stage 4: Animated record-arm feedback
//==============================================================================

struct RecordArmUniforms {
    bool isRecordArmed;
    float time;
};

kernel void RecordArmVisualShader(device const float* inputBuffer [[buffer(0)]],
                                 device float* visualBuffer [[buffer(1)]],
                                 constant RecordArmUniforms& uniforms [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
    
    float inputSample = inputBuffer[index];
    float visualSample = inputSample;
    
    if (uniforms.isRecordArmed) {
        // Record armed - add visual enhancement
        float amplitude = abs(inputSample);
        
        // Add pulsing effect based on time and amplitude
        float pulse = sin(uniforms.time * 4.0f) * 0.1f + 0.9f; // 4Hz pulse
        
        // Enhance signal when record armed
        visualSample *= pulse;
        
        // Add subtle "recording" indicator by modulating amplitude
        if (amplitude > 0.1f) {
            // Add harmonic enhancement for visual feedback
            visualSample += inputSample * 0.2f * sin(uniforms.time * 8.0f);
        }
        
        // Apply record arm "glow" effect
        visualSample *= 1.2f; // Slight boost when armed
    } else {
        // Not record armed - reduce visual prominence
        visualSample *= 0.3f; // Dim the signal
    }
    
    // Clamp to prevent visual artifacts
    visualSample = clamp(visualSample, -1.0f, 1.0f);
    
    visualBuffer[index] = visualSample;
} 