#include <metal_stdlib>
using namespace metal;

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
kernel void pitch_cycle(
    device const float* recentSamples   [[ buffer(0) ]],
    device PitchResult* output          [[ buffer(1) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
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
    float phase = fmod(zeroCrossingIdx, float(bestLag)) / float(bestLag);

    output[0].frequency = freq;
    output[0].cyclePhase = phase;
}
