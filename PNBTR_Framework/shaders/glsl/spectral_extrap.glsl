#version 450

#define FFT_SIZE 512
#define FRAME_SHIFT 0.01            // 10ms step
#define SAMPLE_RATE 48000.0

// Input: windowed buffer (assume Hann window already applied)
layout(std430, binding = 0) readonly buffer FftInput {
    vec2 fftInput[FFT_SIZE]; // FFT result: real/imag
};

layout(std430, binding = 1) buffer FftPrevious {
    vec2 fftPrevious[FFT_SIZE]; // Last frame's real/imag (for phase delta)
};

layout(std430, binding = 2) writeonly buffer FftOutput {
    vec2 fftOutput[FFT_SIZE]; // Final extrapolated FFT frame
};

layout(std430, binding = 3) readonly buffer StepFactor {
    float stepFactor; // Step increment (1.0 = one full frame)
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= FFT_SIZE) return;

    // Extract
    vec2 curr = fftInput[idx];
    vec2 prev = fftPrevious[idx];

    // Calculate magnitudes
    float mag = length(curr);

    // Calculate phase rotation
    float phaseCurr = atan(curr.y, curr.x);
    float phasePrev = atan(prev.y, prev.x);
    float deltaPhase = phaseCurr - phasePrev;

    // Predict next phase
    float predPhase = phaseCurr + (deltaPhase * stepFactor);

    // Reconstruct new FFT bin with rotated phase and same magnitude
    fftOutput[idx] = vec2(mag * cos(predPhase), mag * sin(predPhase));
}
