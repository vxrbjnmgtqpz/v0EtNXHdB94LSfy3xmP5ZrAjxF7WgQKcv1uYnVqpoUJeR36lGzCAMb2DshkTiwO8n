#version 450

#define PREDICTION_LENGTH 2400
#define SAMPLE_RATE 48000.0

// Parameters passed in per-call
struct MicroParams {
    float baseIntensity;      // e.g. 0.003
    float modFreq1;           // Hz, shimmer component 1
    float modFreq2;           // Hz, shimmer component 2
    float grainJitter;        // 0.0 to 1.0
};

layout(std430, binding = 0) readonly buffer BaseInput {
    float baseInput[PREDICTION_LENGTH];
};

layout(std430, binding = 1) readonly buffer ConfidenceMap {
    float confidenceMap[PREDICTION_LENGTH]; // 1 float per sample (0–1)
};

layout(std430, binding = 2) readonly buffer Params {
    MicroParams params;
};

layout(std430, binding = 3) writeonly buffer EnrichedOutput {
    float enrichedOutput[PREDICTION_LENGTH];
};

layout(std430, binding = 4) readonly buffer SeedBuffer {
    float seed;  // stream ID / randomness seed
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= PREDICTION_LENGTH) return;

    float t = float(tid) / SAMPLE_RATE;
    float base = baseInput[tid];
    float conf = clamp(confidenceMap[tid], 0.0, 1.0);

    // Generate shimmer modulation using two sine components
    float mod1 = sin(2.0 * 3.14159265359 * params.modFreq1 * t + seed);
    float mod2 = sin(2.0 * 3.14159265359 * params.modFreq2 * t + seed * 1.3);

    // Compute jittered grain noise
    float jitter = fract(sin(t * 1000.0 + seed) * 43758.5453);
    float noise = (jitter - 0.5) * 2.0 * params.grainJitter;

    // Combine shimmer and noise with confidence gating
    float shimmer = (mod1 + mod2) * 0.5;
    float dynamicGrain = (shimmer + noise) * params.baseIntensity * conf;

    enrichedOutput[tid] = base + dynamicGrain;
}
