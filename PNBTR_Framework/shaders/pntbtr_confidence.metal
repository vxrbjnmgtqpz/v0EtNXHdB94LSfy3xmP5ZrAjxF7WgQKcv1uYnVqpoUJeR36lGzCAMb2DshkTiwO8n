#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400
#define BLOCK_SIZE 48    // 1 ms @ 48kHz
#define NUM_BLOCKS (PREDICTION_LENGTH / BLOCK_SIZE)

using namespace metal;

struct ConfidenceWeights {
    float energyWeight;
    float slopeWeight;
    float spectralWeight;
    float spectralExpected[BLOCK_SIZE];  // Optional: learned spectral average
};

kernel void pntbtr_confidence(
    device const float* predictedWaveform   [[ buffer(0) ]],
    device const float* fftMagPerSample     [[ buffer(1) ]], // optional, same size as waveform
    constant ConfidenceWeights& weights     [[ buffer(2) ]],
    device float* confidenceOut             [[ buffer(3) ]],
    uint blockID                            [[ thread_position_in_grid ]]
) {
    if (blockID >= NUM_BLOCKS) return;

    uint start = blockID * BLOCK_SIZE;
    float energy = 0.0;
    float maxSlope = 0.0;
    float spectralDiff = 0.0;

    for (uint i = 1; i < BLOCK_SIZE; ++i) {
        uint idx = start + i;
        float x0 = predictedWaveform[idx - 1];
        float x1 = predictedWaveform[idx];
        float slope = fabs(x1 - x0);
        float absVal = fabs(x1);

        energy += absVal;
        if (slope > maxSlope) maxSlope = slope;

        float mag = fftMagPerSample[idx];
        float expected = weights.spectralExpected[i];
        spectralDiff += fabs(mag - expected);
    }

    float normEnergy = energy / float(BLOCK_SIZE);
    float normSlope = maxSlope;
    float normSpectral = spectralDiff / float(BLOCK_SIZE);

    // Weighted inverse: lower energy/slope/spectral deviation → higher confidence
    float confidence =
        (weights.energyWeight   * clamp(normEnergy, 0.0, 1.0)) +
        (weights.slopeWeight    * clamp(1.0 - normSlope, 0.0, 1.0)) +
        (weights.spectralWeight * clamp(1.0 - normSpectral, 0.0, 1.0));

    confidenceOut[blockID] = clamp(confidence, 0.0, 1.0);
}
