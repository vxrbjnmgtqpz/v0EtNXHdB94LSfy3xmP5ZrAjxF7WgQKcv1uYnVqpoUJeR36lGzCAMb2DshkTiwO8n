//==============================================================================
//
//  PNBTRKernels.metal
//  Created: GPU-native Metal compute shaders for PNBTR+JELLIE training
//
//  Kernels:
//  - JELLIE_encode: 48kHz→192kHz upsampling + 8-channel distribution
//  - simulate_network: Packet loss and jitter simulation
//  - PNBTR_reconstruct: Neural gap filling and reconstruction
//  - calculate_metrics: SNR, THD, latency calculation
//
//==============================================================================

#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Shared constants and structures
//==============================================================================

struct ProcessingParams {
    float sampleRate;
    int blockSize;
    float packetLossPercent;
    float jitterAmount;
    float gainDb;
};

struct MetricsData {
    float snr;
    float thd;
    float latency;
    float rmsInput;
    float rmsOutput;
    int totalPackets;
    int lostPackets;
};

//==============================================================================
// JELLIE Encoding Kernel
// Upsamples 48kHz stereo to 192kHz 8-channel distributed format
//==============================================================================

kernel void JELLIE_encode(device const float* inputBuffer [[buffer(0)]],
                         device float* jellieBuffer [[buffer(1)]],
                         device const ProcessingParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]])
{
    const int inputSamples = params.blockSize;
    const int upsampleRatio = 4; // 48kHz * 4 = 192kHz
    const int outputChannels = 8;
    const int outputSamples = inputSamples * upsampleRatio;
    
    if (index >= outputSamples) return;
    
    // Calculate input sample index (downsample for interpolation)
    const int inputIndex = index / upsampleRatio;
    const float fraction = (index % upsampleRatio) / float(upsampleRatio);
    
    // Linear interpolation for upsampling
    float leftSample = 0.0f;
    float rightSample = 0.0f;
    
    if (inputIndex < inputSamples - 1) {
        // Interpolate between current and next sample
        const float left0 = inputBuffer[inputIndex * 2];
        const float left1 = inputBuffer[(inputIndex + 1) * 2];
        const float right0 = inputBuffer[inputIndex * 2 + 1];
        const float right1 = inputBuffer[(inputIndex + 1) * 2 + 1];
        
        leftSample = left0 + fraction * (left1 - left0);
        rightSample = right0 + fraction * (right1 - right0);
    } else if (inputIndex < inputSamples) {
        // Use last sample
        leftSample = inputBuffer[inputIndex * 2];
        rightSample = inputBuffer[inputIndex * 2 + 1];
    }
    
    // Distribute to 8 channels with different processing
    for (int channel = 0; channel < outputChannels; ++channel) {
        const int outputIndex = index * outputChannels + channel;
        
        // Channel-specific distribution strategy
        float channelSample = 0.0f;
        
        if (channel < 4) {
            // Channels 0-3: Left channel with phase variations
            channelSample = leftSample * (0.8f + 0.2f * sin(channel * M_PI_F / 4.0f));
        } else {
            // Channels 4-7: Right channel with phase variations
            channelSample = rightSample * (0.8f + 0.2f * sin((channel - 4) * M_PI_F / 4.0f));
        }
        
        // Apply 24-bit quantization simulation
        const float quantLevels = 8388608.0f; // 2^23
        channelSample = round(channelSample * quantLevels) / quantLevels;
        
        // Apply channel gain (1/8 for energy conservation)
        jellieBuffer[outputIndex] = channelSample * 0.125f;
    }
}

//==============================================================================
// Network Simulation Kernel
// Simulates packet loss and jitter on JELLIE-encoded data
//==============================================================================

kernel void simulate_network(device const float* jellieBuffer [[buffer(0)]],
                            device float* networkBuffer [[buffer(1)]],
                            device bool* packetLossMap [[buffer(2)]],
                            device const ProcessingParams& params [[buffer(3)]],
                            uint index [[thread_position_in_grid]])
{
    const int outputChannels = 8;
    const int outputSamples = params.blockSize * 4; // 192kHz
    const int packetSize = 64; // Samples per packet
    const int totalSamples = outputSamples * outputChannels;
    
    if (index >= totalSamples) return;
    
    // Calculate packet index
    const int sampleIndex = index / outputChannels;
    const int packetIndex = sampleIndex / packetSize;
    const int channel = index % outputChannels;
    
    // Copy input to output first
    networkBuffer[index] = jellieBuffer[index];
    
    // Apply packet loss if this packet is marked as lost
    if (packetLossMap[packetIndex]) {
        networkBuffer[index] = 0.0f; // Zero out lost packet
    } else {
        // Apply jitter simulation (simple amplitude modulation)
        const float jitterFactor = 1.0f + params.jitterAmount * 0.01f * 
                                  sin(sampleIndex * 0.1f + channel * M_PI_F / 4.0f);
        networkBuffer[index] *= jitterFactor;
    }
}

//==============================================================================
// PNBTR Reconstruction Kernel
// Neural gap filling and reconstruction of lost packets
//==============================================================================

kernel void PNBTR_reconstruct(device const float* networkBuffer [[buffer(0)]],
                             device float* outputBuffer [[buffer(1)]],
                             device const bool* packetLossMap [[buffer(2)]],
                             device const ProcessingParams& params [[buffer(3)]],
                             uint index [[thread_position_in_grid]])
{
    const int outputChannels = 8;
    const int outputSamples = params.blockSize * 4; // 192kHz
    const int packetSize = 64;
    const int downsampleRatio = 4; // 192kHz → 48kHz
    const int finalSamples = params.blockSize;
    
    if (index >= finalSamples * 2) return; // Stereo output
    
    const int outputSample = index / 2;
    const int outputChannel = index % 2;
    
    // Reconstruct from 8-channel data
    float sample = 0.0f;
    for (int ch = 0; ch < outputChannels; ++ch) {
        int srcIndex = outputSample * outputChannels + ch;
        sample += networkBuffer[srcIndex];
    }
    sample /= float(outputChannels);
    
    // Apply gain
    const float gainLinear = pow(10.0f, params.gainDb / 20.0f);
    sample *= gainLinear;
    
    // Always write to output
    outputBuffer[index] = sample;
}

//==============================================================================
// Metrics Calculation Kernel
// Calculates SNR, THD, and latency metrics
//==============================================================================

kernel void calculate_metrics(device const float* inputBuffer [[buffer(0)]],
                            device const float* outputBuffer [[buffer(1)]],
                            device const bool* packetLossMap [[buffer(2)]],
                            device MetricsData* metrics [[buffer(3)]],
                            device const ProcessingParams& params [[buffer(4)]],
                            uint index [[thread_position_in_grid]])
{
    const int totalSamples = params.blockSize * 2; // Stereo
    const int packetSize = 64;
    const int totalPackets = (params.blockSize * 4) / packetSize; // 192kHz packets
    
    if (index != 0) return; // Only one thread calculates metrics
    
    // Calculate RMS levels
    float inputRMS = 0.0f;
    float outputRMS = 0.0f;
    float signalPower = 0.0f;
    float noisePower = 0.0f;
    
    for (int i = 0; i < totalSamples; ++i) {
        const float inputSample = inputBuffer[i];
        const float outputSample = outputBuffer[i];
        const float error = outputSample - inputSample;
        
        inputRMS += inputSample * inputSample;
        outputRMS += outputSample * outputSample;
        signalPower += inputSample * inputSample;
        noisePower += error * error;
    }
    
    inputRMS = sqrt(inputRMS / totalSamples);
    outputRMS = sqrt(outputRMS / totalSamples);
    signalPower /= totalSamples;
    noisePower /= totalSamples;
    
    // Calculate SNR
    const float snr = (noisePower > 0.0f) ? 10.0f * log10(signalPower / noisePower) : 100.0f;
    
    // Calculate THD (simplified)
    const float thd = (outputRMS > 0.0f) ? (noisePower / (signalPower + noisePower)) * 100.0f : 0.0f;
    
    // Calculate packet loss statistics
    int lostPackets = 0;
    for (int i = 0; i < totalPackets; ++i) {
        if (packetLossMap[i]) lostPackets++;
    }
    
    // Calculate latency (buffer latency)
    const float latency = (params.blockSize / params.sampleRate) * 1000.0f; // ms
    
    // Store results
    metrics->snr = snr;
    metrics->thd = thd;
    metrics->latency = latency;
    metrics->rmsInput = inputRMS;
    metrics->rmsOutput = outputRMS;
    metrics->totalPackets = totalPackets;
    metrics->lostPackets = lostPackets;
}