#version 450

#define SAMPLE_WINDOW 240            // 5 ms @ 48kHz
#define MIN_LAG 20                   // ~2.4 kHz
#define MAX_LAG 480                  // ~100 Hz
#define SAMPLE_RATE 48000.0

// Output pitch and phase structure
struct PitchResult {
    float frequency;
    float cyclePhase;
};

// Input buffer: most recent waveform samples
layout(std430, binding = 0) readonly buffer RecentSamples {
    float recentSamples[SAMPLE_WINDOW];
};

layout(std430, binding = 1) writeonly buffer OutputBuffer {
    PitchResult output[1];
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid > 0) return; // Single-thread phase estimator

    float bestSum = 0.0;
    int bestLag = MIN_LAG;

    // Autocorrelation-based lag finder
    for (int lag = MIN_LAG; lag < MAX_LAG; ++lag) {
        float sum = 0.0;
        for (int i = 0; i < SAMPLE_WINDOW - lag; ++i) {
            sum += recentSamples[i] * recentSamples[i + lag];
        }
        if (sum > bestSum) {
            bestSum = sum;
            bestLag = lag;
        }
    }

    // Convert lag to frequency
    float freq = SAMPLE_RATE / float(bestLag);

    // Estimate current phase in detected cycle (normalized 0–1)
    float zeroCrossingIdx = 0.0;
    for (int i = 1; i < SAMPLE_WINDOW; ++i) {
        if (recentSamples[i - 1] < 0.0 && recentSamples[i] >= 0.0) {
            zeroCrossingIdx = float(i);
            break;
        }
    }
    float phase = mod(zeroCrossingIdx, float(bestLag)) / float(bestLag);

    output[0].frequency = freq;
    output[0].cyclePhase = phase;
}
