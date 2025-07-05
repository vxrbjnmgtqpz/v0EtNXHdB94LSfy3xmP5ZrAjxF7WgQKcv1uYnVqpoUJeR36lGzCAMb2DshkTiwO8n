#version 450

#define PREDICTION_LENGTH 2400

// Inputs:
// - basePrediction[]: deterministic guess (from LPC, spectral, etc.)
// - residualDelta[]: result of pre-run neural model (injected by CPU/CoreML runtime)
// - correctionScale: dynamic mix scalar (per-model confidence, or session-tuned)

layout(std430, binding = 0) readonly buffer BasePrediction {
    float basePrediction[PREDICTION_LENGTH];
};

layout(std430, binding = 1) readonly buffer ResidualDelta {
    float residualDelta[PREDICTION_LENGTH];
};

layout(std430, binding = 2) readonly buffer CorrectionScale {
    float correctionScale;
};

layout(std430, binding = 3) writeonly buffer CorrectedPrediction {
    float correctedPrediction[PREDICTION_LENGTH];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= PREDICTION_LENGTH) return;

    float base   = basePrediction[tid];
    float delta  = residualDelta[tid];

    // Apply residual correction with optional weighting
    float corrected = base + (correctionScale * delta);
    correctedPrediction[tid] = corrected;
}
