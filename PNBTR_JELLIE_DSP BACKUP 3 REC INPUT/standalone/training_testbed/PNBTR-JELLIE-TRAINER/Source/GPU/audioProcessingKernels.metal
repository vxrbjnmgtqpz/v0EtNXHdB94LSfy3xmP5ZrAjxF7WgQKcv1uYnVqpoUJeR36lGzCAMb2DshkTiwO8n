#include <metal_stdlib>
using namespace metal;

//==============================================================================
// JELLIE Encoding Kernel
// Input: Raw audio samples
// Output: JELLIE-encoded data (4x expansion for 192kHz, 8-channel)
//==============================================================================
kernel void jellie_encode_kernel(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& numSamples [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= numSamples) return;
    
    // JELLIE encoding: Upsample to 192kHz (4x) and expand to 8 channels
    float sample = input[id];
    
    // Simple upsample with linear interpolation
    for (uint i = 0; i < 4; i++) {
        uint outputIdx = id * 4 + i;
        output[outputIdx] = sample * (1.0f - i * 0.25f) + 
                           (id < numSamples - 1 ? input[id + 1] : sample) * (i * 0.25f);
    }
}

//==============================================================================
// Network Loss Simulation Kernel
// Input: JELLIE-encoded data
// Output: Network-degraded data with packet loss and jitter
//==============================================================================
kernel void network_simulate_kernel(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   constant uint& numSamples [[buffer(2)]],
                                   constant float& packetLossRate [[buffer(3)]],
                                   constant float& jitterAmount [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= numSamples) return;
    
    // Simple deterministic packet loss simulation
    // In real implementation, would use proper random number generation
    uint packetId = id / 64; // 64 samples per packet
    bool packetLost = ((packetId * 1337) % 100) < uint(packetLossRate * 100);
    
    if (packetLost) {
        // Zero out lost packets
        output[id] = 0.0f;
    } else {
        // Add jitter noise
        float jitterNoise = sin(float(id) * 0.1f) * jitterAmount * 0.01f;
        output[id] = input[id] + jitterNoise;
    }
}

//==============================================================================
// PNBTR Reconstruction Kernel
// Input: Network-degraded JELLIE data
// Output: Reconstructed audio samples
//==============================================================================
kernel void pnbtr_reconstruct_kernel(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& numSamples [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    if (id >= numSamples * 4) return;
    
    uint outputIdx = id / 4;
    if (outputIdx >= numSamples) return;
    
    // PNBTR reconstruction: Downsample and reconstruct
    float sample = input[id];
    
    // Simple gap filling for zero samples (lost packets)
    if (sample == 0.0f && id > 0 && id < numSamples * 4 - 1) {
        // Linear interpolation between non-zero neighbors
        float prev = input[id - 1];
        float next = input[id + 1];
        
        // Find next non-zero sample
        for (uint i = 1; i < 16 && id + i < numSamples * 4; i++) {
            if (input[id + i] != 0.0f) {
                next = input[id + i];
                break;
            }
        }
        
        sample = (prev + next) * 0.5f;
    }
    
    // Accumulate downsampled result
    atomic_fetch_add_explicit((device atomic<float>*)&output[outputIdx], 
                             sample * 0.25f, 
                             memory_order_relaxed);
}

//==============================================================================
// Audio Metrics Calculation Kernel
// Input: Original and reconstructed audio
// Output: SNR, latency, gap quality metrics
//==============================================================================
struct AudioMetrics {
    float snr;
    float latency;
    float gapQuality;
    float processingTime;
};

kernel void calculate_metrics_kernel(device const float* original [[buffer(0)]],
                                   device const float* reconstructed [[buffer(1)]],
                                   device AudioMetrics* metrics [[buffer(2)]],
                                   constant uint& numSamples [[buffer(3)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= numSamples) return;
    
    // Calculate per-sample metrics
    float orig = original[id];
    float recon = reconstructed[id];
    float error = orig - recon;
    
    // Signal power and noise power
    float signalPower = orig * orig;
    float noisePower = error * error;
    
    // Contribute to overall SNR calculation
    atomic_fetch_add_explicit((device atomic<float>*)&metrics->snr, 
                             signalPower / max(noisePower, 1e-10f), 
                             memory_order_relaxed);
    
    // Simple latency detection (phase correlation)
    if (id > 0 && id < numSamples - 1) {
        float origSlope = original[id + 1] - original[id - 1];
        float reconSlope = reconstructed[id + 1] - reconstructed[id - 1];
        
        if (abs(origSlope) > 0.01f && abs(reconSlope) > 0.01f) {
            float correlation = origSlope * reconSlope;
            atomic_fetch_add_explicit((device atomic<float>*)&metrics->latency, 
                                     correlation > 0 ? 0.0f : 1.0f, 
                                     memory_order_relaxed);
        }
    }
    
    // Gap quality (smoothness of reconstruction)
    if (id > 1 && id < numSamples - 1) {
        float derivative = abs(recon - 2.0f * reconstructed[id - 1] + reconstructed[id - 2]);
        atomic_fetch_add_explicit((device atomic<float>*)&metrics->gapQuality, 
                                 derivative, 
                                 memory_order_relaxed);
    }
}

 