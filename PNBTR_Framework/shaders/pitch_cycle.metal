#include <metal_stdlib>
using namespace metal;

#define SAMPLE_WINDOW 512
#define MAX_HARMONICS 8
#define CYCLE_PROFILE_RES 64
#define SAMPLE_RATE 48000.0

using namespace metal;

struct PitchCycleResult {
    float baseFreq;
    float phaseOffset;
    float harmonicAmp[MAX_HARMONICS];
    float cycleProfile[CYCLE_PROFILE_RES];
};

kernel void pitch_cycle(
    device const float* recentSamples   [[ buffer(0) ]],
    device PitchCycleResult* result     [[ buffer(1) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
    if (tid > 0) return; // single-thread extraction

    float autocorr[256] = {0.0};
    int bestLag = 1;
    float bestScore = 0.0;

    // Autocorrelation search
    for (int lag = 20; lag < 256; ++lag) {
        float score = 0.0;
        for (int i = 0; i < SAMPLE_WINDOW - lag; ++i) {
            score += recentSamples[i] * recentSamples[i + lag];
        }
        if (score > bestScore) {
            bestScore = score;
            bestLag = lag;
        }
    }

    float baseFreq = SAMPLE_RATE / float(bestLag);
    result->baseFreq = baseFreq;

    // Estimate phase offset using zero-crossing
    float zc = 0.0;
    for (int i = 1; i < SAMPLE_WINDOW; ++i) {
        if (recentSamples[i - 1] < 0.0 && recentSamples[i] >= 0.0) {
            zc = float(i);
            break;
        }
    }
    result->phaseOffset = fmod(zc, float(bestLag)) / float(bestLag);

    // Harmonic weights (simple projection)
    for (int h = 0; h < MAX_HARMONICS; ++h) {
        float freq = baseFreq * float(h + 1);
        float sum = 0.0;
        for (int i = 0; i < SAMPLE_WINDOW; ++i) {
            float t = float(i) / SAMPLE_RATE;
            sum += recentSamples[i] * sin(2.0 * M_PI_F * freq * t);
        }
        result->harmonicAmp[h] = fabs(sum) / float(SAMPLE_WINDOW);
    }

    // Build cycle profile (normalized waveform over one cycle)
    for (int j = 0; j < CYCLE_PROFILE_RES; ++j) {
        float phase = float(j) / float(CYCLE_PROFILE_RES);
        float t = phase / baseFreq;
        float value = 0.0;

        for (int h = 0; h < MAX_HARMONICS; ++h) {
            float freq = baseFreq * float(h + 1);
            float amp = result->harmonicAmp[h];
            value += amp * sin(2.0 * M_PI_F * freq * t);
        }

        result->cycleProfile[j] = value;
    }
}
