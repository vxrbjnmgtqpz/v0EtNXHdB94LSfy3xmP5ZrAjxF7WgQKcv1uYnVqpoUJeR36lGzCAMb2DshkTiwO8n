#version 450

#define PREDICTION_LENGTH 2400    // 50 ms @ 48kHz
#define NOISE_SCALE 0.003         // Intensity of micro-noise
#define MOD_FREQ1 1800.0          // Hz
#define MOD_FREQ2 3200.0          // Hz
#define SAMPLE_RATE 48000.0

layout(std430, binding = 0) readonly buffer BasePrediction {
    float basePrediction[PREDICTION_LENGTH];
};

layout(std430, binding = 1) writeonly buffer EnrichedOutput {
    float enrichedOutput[PREDICTION_LENGTH];
};

layout(std430, binding = 2) readonly buffer SeedBuffer {
    float seed;   // Random seed or stream ID
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= PREDICTION_LENGTH) return;

    float t = float(tid) / SAMPLE_RATE;

    // Subtle modulations — pseudo-randomized high-freq tremble
    float mod1 = sin(2.0 * 3.14159265 * MOD_FREQ1 * t + seed);
    float mod2 = sin(2.0 * 3.14159265 * MOD_FREQ2 * t + seed * 1.7);

    // Combine with slight high-frequency noise
    float noise = fract(sin((t + seed) * 12345.678) * 54321.123);
    noise = (noise - 0.5) * 2.0 * NOISE_SCALE;

    // Blend
    float base = basePrediction[tid];
    float modulated = base + (mod1 + mod2) * NOISE_SCALE + noise;

    enrichedOutput[tid] = modulated;
}
