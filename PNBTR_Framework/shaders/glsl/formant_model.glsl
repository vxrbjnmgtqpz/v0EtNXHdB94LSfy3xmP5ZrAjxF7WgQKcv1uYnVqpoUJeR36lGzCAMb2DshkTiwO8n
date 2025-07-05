#version 450

#define SAMPLE_WINDOW 512
#define NUM_FORMANTS 3
#define FORMANT_SEARCH_BINS 48
#define PREDICTION_LENGTH 2400
#define SAMPLE_RATE 48000.0

struct Formant {
    float centerFreq;
    float bandwidth;
    float amplitude;
};

layout(std430, binding = 0) readonly buffer RecentSamples {
    float recentSamples[SAMPLE_WINDOW];
};

layout(std430, binding = 1) writeonly buffer PredictedSamples {
    float predictedSamples[PREDICTION_LENGTH];
};

layout(std430, binding = 2) writeonly buffer FormantOut {
    Formant formantOut[NUM_FORMANTS];
};

layout(std430, binding = 3) readonly buffer MlFormantAmps {
    float mlFormantAmps[NUM_FORMANTS]; // optional: ML-inferred amp per bin
};

layout(std430, binding = 4) readonly buffer UseMLFactor {
    float useMLFactor; // 0.0 = ignore ML, 1.0 = full ML
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid > 0) return;

    float bestFreqs[NUM_FORMANTS];
    float bestAmps[NUM_FORMANTS];
    float bestBandwidths[NUM_FORMANTS] = float[](100.0, 150.0, 200.0);

    // Initialize arrays
    for (int i = 0; i < NUM_FORMANTS; ++i) {
        bestFreqs[i] = 0.0;
        bestAmps[i] = 0.0;
    }

    float binWidth = 4000.0 / float(FORMANT_SEARCH_BINS);

    // Step 1: Bandpass sweep (simple sine projection)
    for (uint bin = 0; bin < FORMANT_SEARCH_BINS; ++bin) {
        float freq = bin * binWidth + 100.0;
        float amp = 0.0;
        for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
            float t = float(i) / SAMPLE_RATE;
            amp += recentSamples[i] * sin(2.0 * 3.14159265359 * freq * t);
        }
        amp = abs(amp / float(SAMPLE_WINDOW));

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
            float phase = 2.0 * 3.14159265359 * formantOut[f].centerFreq * t;
            float decay = exp(-formantOut[f].bandwidth * t);
            sum += formantOut[f].amplitude * decay * sin(phase);
        }
        predictedSamples[i] = sum;
    }
}
