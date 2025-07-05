#version 450

#define PREDICTION_LENGTH 2400
#define BLOCK_SIZE 48    // 1 ms @ 48kHz
#define NUM_BLOCKS (PREDICTION_LENGTH / BLOCK_SIZE)

struct ConfidenceWeights {
    float energyWeight;
    float slopeWeight;
    float spectralWeight;
    float spectralExpected[BLOCK_SIZE];  // Optional: learned spectral average
};

layout(std430, binding = 0) readonly buffer PredictedWaveform {
    float predictedWaveform[PREDICTION_LENGTH];
};

layout(std430, binding = 1) readonly buffer FftMagPerSample {
    float fftMagPerSample[PREDICTION_LENGTH]; // optional, same size as waveform
};

layout(std430, binding = 2) readonly buffer Weights {
    ConfidenceWeights weights;
};

layout(std430, binding = 3) writeonly buffer ConfidenceOut {
    float confidenceOut[NUM_BLOCKS];
};

layout(local_size_x = 50, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint blockID = gl_GlobalInvocationID.x;
    if (blockID >= NUM_BLOCKS) return;

    uint start = blockID * BLOCK_SIZE;
    float energy = 0.0;
    float maxSlope = 0.0;
    float spectralDiff = 0.0;

    for (uint i = 1; i < BLOCK_SIZE; ++i) {
        uint idx = start + i;
        float x0 = predictedWaveform[idx - 1];
        float x1 = predictedWaveform[idx];
        float slope = abs(x1 - x0);
        float absVal = abs(x1);

        energy += absVal;
        if (slope > maxSlope) maxSlope = slope;

        float mag = fftMagPerSample[idx];
        float expected = weights.spectralExpected[i];
        spectralDiff += abs(mag - expected);
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
