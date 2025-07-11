#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Metrics Compute Shader
// Real-time audio metrics calculation (SNR, THD, latency)
//==============================================================================

struct AudioMetrics {
    float snr_db;
    float thd_percent; 
    float latency_ms;
    float reconstruction_rate_percent;
};

kernel void MetricsComputeShader(device const float* inputBuffer [[buffer(0)]],
                                device const float* outputBuffer [[buffer(1)]],
                                device AudioMetrics* metricsBuffer [[buffer(2)]],
                                constant int& bufferSize [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    
    // Only first thread calculates metrics to avoid race conditions
    if (index != 0) return;
    
    float inputRMS = 0.0f;
    float outputRMS = 0.0f;
    float noiseRMS = 0.0f;
    int nonZeroSamples = 0;
    
    // Calculate RMS values
    for (int i = 0; i < bufferSize; i++) {
        float inputSample = inputBuffer[i];
        float outputSample = outputBuffer[i];
        
        inputRMS += inputSample * inputSample;
        outputRMS += outputSample * outputSample;
        
        // Calculate noise (difference between input and output)
        float noise = inputSample - outputSample;
        noiseRMS += noise * noise;
        
        // Count non-zero samples for reconstruction rate
        if (abs(outputSample) > 0.001f) {
            nonZeroSamples++;
        }
    }
    
    inputRMS = sqrt(inputRMS / bufferSize);
    outputRMS = sqrt(outputRMS / bufferSize);
    noiseRMS = sqrt(noiseRMS / bufferSize);
    
    // Calculate SNR (Signal-to-Noise Ratio)
    float snr = 20.0f * log10(max(outputRMS / max(noiseRMS, 0.0001f), 0.0001f));
    
    // Simple THD estimate (placeholder)
    float thd = (noiseRMS / max(outputRMS, 0.0001f)) * 100.0f;
    
    // Reconstruction rate (percentage of non-zero samples)
    float reconstructionRate = ((float)nonZeroSamples / bufferSize) * 100.0f;
    
    // Estimated latency (based on buffer size and sample rate)
    float latency = (bufferSize / 48000.0f) * 1000.0f; // Assume 48kHz
    
    // Store results
    metricsBuffer->snr_db = clamp(snr, -60.0f, 100.0f);
    metricsBuffer->thd_percent = clamp(thd, 0.0f, 100.0f);
    metricsBuffer->latency_ms = latency;
    metricsBuffer->reconstruction_rate_percent = reconstructionRate;
} 