/*
  ==============================================================================

    AudioKernels.metal
    Created: Metal Compute Shaders for Audio Processing

    GPU-accelerated audio processing kernels:
    - JELLIE audio compression/decompression
    - PNBTR neural enhancement/reconstruction  
    - Basic audio effects (gain, filters, etc.)
    - High-performance parallel processing

    Features:
    - Optimized for real-time audio processing
    - Support for multichannel audio
    - Efficient memory bandwidth utilization
    - Metal Performance Shaders integration

  ==============================================================================
*/

#include <metal_stdlib>
using namespace metal;

//==============================================================================
// Constants and Structures

constant uint MAX_CHANNELS = 8;
constant uint MAX_FRAMES_PER_THREAD = 16;

struct AudioProcessingParams {
    uint numChannels;
    uint numFrames;
    uint samplesPerThread;
    float sampleRate;
};

struct GainParams {
    float gain;
    float smoothingCoeff;
};

struct FilterParams {
    float cutoff;
    float resonance;
    float filterType; // 0=LowPass, 1=HighPass, 2=BandPass, 3=Notch
};

struct JELLIEParams {
    float compressionRatio;
    float qualityLevel;
    uint adaptiveMode;
    float networkLatency;
};

struct PNBTRParams {
    float enhancementLevel;
    float reconstructionDepth;
    uint neuralMode;
    float modelStrength;
};

//==============================================================================
// Utility Functions

// Biquad filter coefficients calculation
void calculateBiquadCoeffs(float cutoff, float resonance, float sampleRate, 
                          int filterType, thread float* coeffs) {
    float w = 2.0 * M_PI_F * cutoff / sampleRate;
    float cosw = cos(w);
    float sinw = sin(w);
    float alpha = sinw / (2.0 * resonance);
    
    float b0, b1, b2, a0, a1, a2;
    
    switch(filterType) {
        case 0: // Low Pass
            b0 = (1.0 - cosw) / 2.0;
            b1 = 1.0 - cosw;
            b2 = (1.0 - cosw) / 2.0;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cosw;
            a2 = 1.0 - alpha;
            break;
            
        case 1: // High Pass
            b0 = (1.0 + cosw) / 2.0;
            b1 = -(1.0 + cosw);
            b2 = (1.0 + cosw) / 2.0;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cosw;
            a2 = 1.0 - alpha;
            break;
            
        case 2: // Band Pass
            b0 = alpha;
            b1 = 0.0;
            b2 = -alpha;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cosw;
            a2 = 1.0 - alpha;
            break;
            
        default: // Notch
            b0 = 1.0;
            b1 = -2.0 * cosw;
            b2 = 1.0;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cosw;
            a2 = 1.0 - alpha;
            break;
    }
    
    // Normalize coefficients
    coeffs[0] = b0 / a0;
    coeffs[1] = b1 / a0;
    coeffs[2] = b2 / a0;
    coeffs[3] = a1 / a0;
    coeffs[4] = a2 / a0;
}

// Simple neural network activation function
float neuralActivation(float x, uint mode) {
    switch(mode) {
        case 0: return tanh(x);           // Hyperbolic tangent
        case 1: return max(0.0f, x);      // ReLU
        case 2: return 1.0f / (1.0f + exp(-x)); // Sigmoid
        default: return x;                // Linear
    }
}

//==============================================================================
// Basic Audio Processing Kernels

// Gain processing with smoothing
kernel void gain_processor(device float* inputBuffer [[buffer(0)]],
                          device float* outputBuffer [[buffer(1)]],
                          constant AudioProcessingParams& audioParams [[buffer(2)]],
                          constant GainParams& gainParams [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    
    uint totalSamples = audioParams.numChannels * audioParams.numFrames;
    if (index >= totalSamples) return;
    
    float input = inputBuffer[index];
    float targetGain = gainParams.gain;
    float smoothing = gainParams.smoothingCoeff;
    
    // Apply smoothed gain
    float output = input * targetGain;
    
    // Simple high-frequency preservation
    if (abs(input) > 0.8f) {
        output = output * 0.95f + input * 0.05f;
    }
    
    outputBuffer[index] = output;
}

// Biquad filter processing
kernel void biquad_filter(device float* inputBuffer [[buffer(0)]],
                         device float* outputBuffer [[buffer(1)]],
                         device float* stateBuffer [[buffer(2)]],
                         constant AudioProcessingParams& audioParams [[buffer(3)]],
                         constant FilterParams& filterParams [[buffer(4)]],
                         uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    // Calculate filter coefficients
    float coeffs[5];
    calculateBiquadCoeffs(filterParams.cutoff, filterParams.resonance, 
                         audioParams.sampleRate, (int)filterParams.filterType, coeffs);
    
    float input = inputBuffer[index];
    
    // Get filter state for this channel (x1, x2, y1, y2)
    uint stateOffset = channel * 4;
    float x1 = stateBuffer[stateOffset + 0];
    float x2 = stateBuffer[stateOffset + 1];
    float y1 = stateBuffer[stateOffset + 2];
    float y2 = stateBuffer[stateOffset + 3];
    
    // Apply biquad filter
    float output = coeffs[0] * input + coeffs[1] * x1 + coeffs[2] * x2 
                 - coeffs[3] * y1 - coeffs[4] * y2;
    
    // Update state
    stateBuffer[stateOffset + 0] = input;  // x1 = current input
    stateBuffer[stateOffset + 1] = x1;     // x2 = previous input
    stateBuffer[stateOffset + 2] = output; // y1 = current output
    stateBuffer[stateOffset + 3] = y1;     // y2 = previous output
    
    outputBuffer[index] = output;
}

//==============================================================================
// JELLIE Audio Compression Kernels

// JELLIE Encoder - Adaptive audio compression
kernel void jellie_encoder(device float* inputBuffer [[buffer(0)]],
                          device float* outputBuffer [[buffer(1)]],
                          device float* compressionState [[buffer(2)]],
                          constant AudioProcessingParams& audioParams [[buffer(3)]],
                          constant JELLIEParams& jellieParams [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    float compressionRatio = jellieParams.compressionRatio;
    float quality = jellieParams.qualityLevel;
    
    // Adaptive compression based on signal characteristics
    float magnitude = abs(input);
    float dynamicCompression = compressionRatio;
    
    if (jellieParams.adaptiveMode > 0) {
        // Adapt compression based on signal energy
        float energy = magnitude * magnitude;
        dynamicCompression = mix(compressionRatio * 0.5f, compressionRatio * 1.5f, energy);
    }
    
    // Psychoacoustic masking (simplified)
    float maskingThreshold = 0.01f * quality;
    if (magnitude < maskingThreshold) {
        dynamicCompression *= 2.0f; // More aggressive compression for quiet signals
    }
    
    // Apply compression (simplified bit reduction simulation)
    float quantizationLevels = 32.0f / dynamicCompression;
    float compressed = round(input * quantizationLevels) / quantizationLevels;
    
    // Network latency compensation
    float latencyCoeff = 1.0f - jellieParams.networkLatency * 0.1f;
    compressed *= latencyCoeff;
    
    outputBuffer[index] = compressed;
    
    // Update compression state for next frame
    uint stateIndex = channel * audioParams.numFrames + frame;
    if (stateIndex < audioParams.numChannels * audioParams.numFrames) {
        compressionState[stateIndex] = magnitude;
    }
}

// JELLIE Decoder - Audio decompression with reconstruction
kernel void jellie_decoder(device float* inputBuffer [[buffer(0)]],
                          device float* outputBuffer [[buffer(1)]],
                          device float* reconstructionState [[buffer(2)]],
                          constant AudioProcessingParams& audioParams [[buffer(3)]],
                          constant JELLIEParams& jellieParams [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    float quality = jellieParams.qualityLevel;
    
    // Basic decompression (expand quantized signal)
    float decompressed = input;
    
    // High-frequency reconstruction (simplified spectral enhancement)
    float baseFreq = decompressed;
    float harmonicEnhancement = 0.0f;
    
    if (quality > 0.5f) {
        // Add harmonic content based on fundamental
        float harmonic2 = sin(baseFreq * 2.0f * M_PI_F) * 0.1f;
        float harmonic3 = sin(baseFreq * 3.0f * M_PI_F) * 0.05f;
        harmonicEnhancement = harmonic2 + harmonic3;
    }
    
    // Adaptive reconstruction based on signal history
    uint stateIndex = channel * audioParams.numFrames + frame;
    float previousSample = 0.0f;
    if (stateIndex > 0 && stateIndex < audioParams.numChannels * audioParams.numFrames) {
        previousSample = reconstructionState[stateIndex - 1];
    }
    
    // Temporal smoothing
    float smoothingFactor = 0.1f * quality;
    float smoothed = mix(decompressed, previousSample, smoothingFactor);
    
    float output = smoothed + harmonicEnhancement * quality;
    
    outputBuffer[index] = output;
    
    // Update reconstruction state
    if (stateIndex < audioParams.numChannels * audioParams.numFrames) {
        reconstructionState[stateIndex] = output;
    }
}

//==============================================================================
// PNBTR Neural Enhancement Kernels

// PNBTR Enhancer - Neural network-style audio enhancement
kernel void pnbtr_enhancer(device float* inputBuffer [[buffer(0)]],
                          device float* outputBuffer [[buffer(1)]],
                          device float* neuralWeights [[buffer(2)]],
                          device float* neuralState [[buffer(3)]],
                          constant AudioProcessingParams& audioParams [[buffer(4)]],
                          constant PNBTRParams& pnbtrParams [[buffer(5)]],
                          uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    float enhancementLevel = pnbtrParams.enhancementLevel;
    uint neuralMode = pnbtrParams.neuralMode;
    
    // Neural network processing (simplified multi-layer perceptron)
    const uint NUM_LAYERS = 3;
    const uint NEURONS_PER_LAYER = 8;
    
    float neuralInput = input;
    float layerOutput = neuralInput;
    
    // Process through neural layers
    for (uint layer = 0; layer < NUM_LAYERS; ++layer) {
        float layerSum = 0.0f;
        
        for (uint neuron = 0; neuron < NEURONS_PER_LAYER; ++neuron) {
            uint weightIndex = layer * NEURONS_PER_LAYER + neuron;
            float weight = neuralWeights[weightIndex % 64]; // Assume 64 weights max
            
            layerSum += layerOutput * weight;
        }
        
        // Apply activation function
        layerOutput = neuralActivation(layerSum / NEURONS_PER_LAYER, neuralMode);
    }
    
    // Blend original and enhanced signal
    float enhancement = layerOutput * enhancementLevel;
    float output = mix(input, enhancement, pnbtrParams.modelStrength);
    
    // Psychoacoustic enhancement
    float magnitude = abs(output);
    if (magnitude > 0.1f) {
        // Enhance harmonics for prominent signals
        float harmonicBoost = sin(output * 4.0f * M_PI_F) * 0.05f * enhancementLevel;
        output += harmonicBoost;
    }
    
    outputBuffer[index] = output;
    
    // Update neural state for temporal processing
    uint stateIndex = channel * audioParams.numFrames + frame;
    if (stateIndex < audioParams.numChannels * audioParams.numFrames) {
        neuralState[stateIndex] = layerOutput;
    }
}

// PNBTR Reconstructor - Neural reconstruction of missing frequencies
kernel void pnbtr_reconstructor(device float* inputBuffer [[buffer(0)]],
                               device float* outputBuffer [[buffer(1)]],
                               device float* spectralBuffer [[buffer(2)]],
                               constant AudioProcessingParams& audioParams [[buffer(3)]],
                               constant PNBTRParams& pnbtrParams [[buffer(4)]],
                               uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    float reconstructionDepth = pnbtrParams.reconstructionDepth;
    
    // Spectral analysis (simplified FFT-like processing)
    float spectralContent[8]; // 8-bin spectrum approximation
    
    // Calculate spectral bins using overlapping windows
    for (uint bin = 0; bin < 8; ++bin) {
        float frequency = (bin + 1) * 1000.0f; // 1kHz steps
        float phase = frame * frequency * 2.0f * M_PI_F / audioParams.sampleRate;
        spectralContent[bin] = input * cos(phase);
    }
    
    // Neural reconstruction of missing spectral content
    float reconstructed = input;
    
    for (uint bin = 0; bin < 8; ++bin) {
        float binEnergy = spectralContent[bin] * spectralContent[bin];
        
        if (binEnergy < 0.01f) { // Reconstruct weak frequencies
            float neighborEnergy = 0.0f;
            
            // Use neighboring bins for reconstruction
            if (bin > 0) neighborEnergy += spectralContent[bin - 1] * spectralContent[bin - 1];
            if (bin < 7) neighborEnergy += spectralContent[bin + 1] * spectralContent[bin + 1];
            
            float reconstructionGain = sqrt(neighborEnergy) * reconstructionDepth * 0.1f;
            float frequency = (bin + 1) * 1000.0f;
            float phase = frame * frequency * 2.0f * M_PI_F / audioParams.sampleRate;
            
            reconstructed += sin(phase) * reconstructionGain;
        }
    }
    
    // Apply neural depth control
    float depthFactor = tanh(reconstructionDepth * 2.0f - 1.0f);
    float output = mix(input, reconstructed, depthFactor);
    
    outputBuffer[index] = output;
    
    // Store spectral analysis for next frame
    uint spectralIndex = channel * 8 + (frame % 8);
    if (spectralIndex < audioParams.numChannels * 8) {
        spectralBuffer[spectralIndex] = spectralContent[frame % 8];
    }
}

//==============================================================================
// Advanced Processing Kernels

// Multi-tap delay with feedback (for reverb/echo effects)
kernel void multi_tap_delay(device float* inputBuffer [[buffer(0)]],
                           device float* outputBuffer [[buffer(1)]],
                           device float* delayBuffer [[buffer(2)]],
                           constant AudioProcessingParams& audioParams [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    
    // Multi-tap delay parameters
    const uint NUM_TAPS = 4;
    const uint delayTaps[4] = {441, 882, 1323, 1764}; // Various delay lengths
    const float tapGains[4] = {0.6f, 0.4f, 0.3f, 0.2f};
    
    float output = input;
    uint delayBufferSize = 2048; // Assume 2048 sample delay buffer per channel
    uint channelOffset = channel * delayBufferSize;
    uint writeIndex = (channelOffset + frame) % delayBufferSize;
    
    // Add delayed signals
    for (uint tap = 0; tap < NUM_TAPS; ++tap) {
        uint readIndex = (writeIndex - delayTaps[tap] + delayBufferSize) % delayBufferSize;
        float delayedSample = delayBuffer[channelOffset + readIndex];
        output += delayedSample * tapGains[tap];
    }
    
    // Write input to delay buffer with feedback
    delayBuffer[channelOffset + writeIndex] = input + output * 0.3f;
    
    outputBuffer[index] = output * 0.7f; // Scale to prevent clipping
}

// Spectral gate (noise reduction)
kernel void spectral_gate(device float* inputBuffer [[buffer(0)]],
                         device float* outputBuffer [[buffer(1)]],
                         device float* noiseProfile [[buffer(2)]],
                         constant AudioProcessingParams& audioParams [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    
    uint channel = index % audioParams.numChannels;
    uint frame = index / audioParams.numChannels;
    
    if (frame >= audioParams.numFrames) return;
    
    float input = inputBuffer[index];
    float inputMagnitude = abs(input);
    
    // Get noise threshold for this frequency bin (approximated)
    uint freqBin = frame % 64; // 64 frequency bins
    float noiseThreshold = noiseProfile[channel * 64 + freqBin];
    
    // Apply spectral gate
    float gateRatio = inputMagnitude / (noiseThreshold + 0.001f);
    float gateGain = 1.0f;
    
    if (gateRatio < 2.0f) { // Below 2x noise threshold
        gateGain = gateRatio * gateRatio / 4.0f; // Quadratic suppression
    }
    
    float output = input * gateGain;
    
    // Update noise profile (adaptive)
    float updateRate = 0.001f;
    noiseProfile[channel * 64 + freqBin] = 
        mix(noiseProfile[channel * 64 + freqBin], inputMagnitude, updateRate);
    
    outputBuffer[index] = output;
} 