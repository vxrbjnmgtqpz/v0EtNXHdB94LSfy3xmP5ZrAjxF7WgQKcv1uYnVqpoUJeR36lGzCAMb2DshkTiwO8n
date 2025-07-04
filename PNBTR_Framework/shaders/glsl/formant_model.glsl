#version 450

#define SAMPLE_WINDOW 512              // ~10.6 ms @ 48kHz
#define NUM_FORMANTS 3
#define PREDICTION_LENGTH 1024         // ~21 ms

struct Formant {
    float centerFreq;   // Hz
    float bandwidth;    // Hz
    float amplitude;    // linear gain
};

// Input buffer: recent audio history
layout(std430, binding = 0) readonly buffer RecentSamples {
    float recentSamples[SAMPLE_WINDOW];
};

layout(std430, binding = 1) writeonly buffer PredictedSamples {
    float predictedSamples[PREDICTION_LENGTH];
};

layout(std430, binding = 2) writeonly buffer OutputFormants {
    Formant outputFormants[NUM_FORMANTS];
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid > 0) return; // Formant estimation runs once for now

    // Step 1: crude formant extraction via bandpass scanning
    const float sampleRate = 48000.0;
    float energy[NUM_FORMANTS];
    float centerFreqs[NUM_FORMANTS] = float[](700.0, 1200.0, 2600.0); // F1–F3 approx for human voice
    float bandwidths[NUM_FORMANTS] = float[](100.0, 150.0, 200.0);

    for (uint f = 0; f < NUM_FORMANTS; ++f) {
        energy[f] = 0.0;
        float fc = centerFreqs[f];
        float bw = bandwidths[f];
        float filtered = 0.0;
        for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
            float t = float(i) / sampleRate;
            filtered += recentSamples[i] * sin(2.0 * 3.14159265 * fc * t) * exp(-bw * t);
        }
        energy[f] = abs(filtered);
    }

    // Normalize and output estimated formants
    for (uint i = 0; i < NUM_FORMANTS; ++i) {
        outputFormants[i].centerFreq = centerFreqs[i];
        outputFormants[i].bandwidth = bandwidths[i];
        outputFormants[i].amplitude = energy[i];
    }

    // Step 2: synthesize using formant sine waves
    for (uint i = 0; i < PREDICTION_LENGTH; ++i) {
        float t = float(i) / sampleRate;
        float sum = 0.0;
        for (uint f = 0; f < NUM_FORMANTS; ++f) {
            float phase = 2.0 * 3.14159265 * outputFormants[f].centerFreq * t;
            float decay = exp(-outputFormants[f].bandwidth * t);
            sum += outputFormants[f].amplitude * decay * sin(phase);
        }
        predictedSamples[i] = sum;
    }
}
