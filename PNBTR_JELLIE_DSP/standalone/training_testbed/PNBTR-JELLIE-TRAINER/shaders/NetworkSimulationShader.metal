#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Network Simulation Shader
// Stage 6: Packet loss and jitter simulation
//==============================================================================

struct NetworkParams {
    float packetLoss; // 0-100
    float jitter; // ms
    uint randomSeed;
};

kernel void NetworkSimulationShader(device const float* inputBuffer [[buffer(0)]],
                                    device float* outputBuffer [[buffer(1)]],
                                    constant NetworkParams& params [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
    
    float sample = inputBuffer[index];
    
    // Simulate packet loss using simple deterministic pattern
    // Use index-based pseudo-random for consistent results
    uint seed = index * 1103515245u + params.randomSeed;
    float random = (float)(seed % 1000) / 1000.0f;
    
    // Apply packet loss
    if (random < (params.packetLoss / 100.0f)) {
        // Packet lost - zero the sample
        sample = 0.0f;
    } else {
        // Packet survived - apply jitter effects
        // Simple jitter simulation: slight amplitude and timing variations
        float jitterAmount = params.jitter / 1000.0f; // Convert ms to fraction
        float jitterMod = sin(index * 0.01f) * jitterAmount;
        
        // Apply jitter as amplitude modulation
        sample *= (1.0f + jitterMod * 0.1f);
        
        // Clamp to prevent overflow
        sample = clamp(sample, -1.0f, 1.0f);
    }
    
    outputBuffer[index] = sample;
} 