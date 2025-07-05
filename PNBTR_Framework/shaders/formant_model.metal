#include <metal_stdlib>
using namespace metal;

#define SAMPLE_WINDOW 512
#define NUM_FORMANTS 3
#define FORMANT_SEARCH_BINS 48
#define PREDICTION_LENGTH 2400
#define SAMPLE_RATE 48000.0

using namespace metal;

struct Formant {
    float centerFreq;
    float bandwidth;
    float amplitude;
};

kernel void formant_model(
    device const float* recentSamples      [[ buffer(0) ]],
    device float* predictedSamples         [[ buffer(1) ]],
    device Formant* formantOut             [[ buffer(2) ]],
    device const float* mlFormantAmps      [[ buffer(3) ]], // optional: ML-inferred amp per bin
    constant float& useMLFactor            [[ buffer(4) ]], // 0.0 = ignore ML, 1.0 = full ML
    uint tid                               [[ thread_position_in_grid ]]
) {
    if (tid > 0) return;

    float bestFreqs[NUM_FORMANTS] = {0.0};
    float bestAmps[NUM_FORMANTS] = {0.0};
    float bestBandwidths[NUM_FORMANTS] = {100.0, 150.0, 200.0};

    float binWidth = 4000.0 / float(FORMANT_SEARCH_BINS);

    // Step 1: Bandpass sweep (simple sine projection)
    for (uint bin = 0; bin < FORMANT_SEARCH_BINS; ++bin) {
        float freq = bin * binWidth + 100.0;
        float amp = 0.0;
        for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
            float t = float(i) / SAMPLE_RATE;
            amp += recentSamples[i] * sin(2.0 * M_PI_F * freq * t);
        }
        amp = fabs(amp / float(SAMPLE_WINDOW));

        // Insert into top-N formants if higher than current
        for (int f = 0; f < NUM_FORMANTS; ++f) {
            if (amp > bestAmps[f]) {
                for (int j = NUM_FORMANTS - 1; j > f; --j) {
                    bestAmps[j] = bestAmps[j - 1];
                    bestFreqs[j] = bestFreqs[j - 1];
                }
                bestAmps[f] = amp;
                bestFreqs[f] = freq;
                break;
            }
        }
    }

    // Step 2: Blend with ML if provided
    for (int f = 0; f < NUM_FORMANTS; ++f) {
        float mlAmp = mlFormantAmps[f];
        float finalAmp = mix(bestAmps[f], mlAmp, useMLFactor);

        formantOut[f].centerFreq = bestFreqs[f];
        formantOut[f].bandwidth = bestBandwidths[f];
        formantOut[f].amplitude = finalAmp;
    }

    // Step 3: Synthesize output
    for (uint i = 0; i < PREDICTION_LENGTH; ++i) {
        float t = float(i) / SAMPLE_RATE;
        float sum = 0.0;
        for (uint f = 0; f < NUM_FORMANTS; ++f) {
            float phase = 2.0 * M_PI_F * formantOut[f].centerFreq * t;
            float decay = exp(-formantOut[f].bandwidth * t);
            sum += formantOut[f].amplitude * decay * sin(phase);
        }
        predictedSamples[i] = sum;
    }
}
