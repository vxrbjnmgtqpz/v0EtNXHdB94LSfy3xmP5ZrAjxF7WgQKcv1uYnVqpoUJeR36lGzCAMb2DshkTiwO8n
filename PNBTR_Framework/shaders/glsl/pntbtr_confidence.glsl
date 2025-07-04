#version 450

#define PREDICTION_LENGTH 2400
#define CONFIDENCE_BLOCK 48          // 1ms @ 48kHz
#define NOISE_FLOOR_THRESHOLD 0.001  // Used to reject silence
#define MAX_SLOPE_THRESHOLD 0.3      // Used to flag sharp edges

layout(std430, binding = 0) readonly buffer PredictedWaveform {
    float predictedWaveform[PREDICTION_LENGTH];
};

layout(std430, binding = 1) writeonly buffer ConfidenceProfile {
    float confidenceProfile[PREDICTION_LENGTH / CONFIDENCE_BLOCK];
};

layout(local_size_x = 50, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint blockIdx = gl_GlobalInvocationID.x;
    uint start = blockIdx * CONFIDENCE_BLOCK;
    if (start >= PREDICTION_LENGTH) return;

    float energy = 0.0;
    float maxSlope = 0.0;

    for (uint i = 1; i < CONFIDENCE_BLOCK; ++i) {
        uint idx = start + i;
        if (idx >= PREDICTION_LENGTH) break;

        float x0 = predictedWaveform[idx - 1];
        float x1 = predictedWaveform[idx];

        float diff = x1 - x0;
        energy += abs(x1);
        if (abs(diff) > maxSlope) {
            maxSlope = abs(diff);
        }
    }

    float normEnergy = energy / float(CONFIDENCE_BLOCK);
    float slopePenalty = clamp(maxSlope / MAX_SLOPE_THRESHOLD, 0.0, 1.0);

    float confidence = normEnergy > NOISE_FLOOR_THRESHOLD
                     ? 1.0 - slopePenalty
                     : 0.0; // if silence, confidence is zero

    confidenceProfile[blockIdx] = confidence;
}
