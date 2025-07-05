#include <metal_stdlib>
using namespace metal;

#define FFT_SIZE 512
#define FRAME_SHIFT 0.01            // 10ms step
#define SAMPLE_RATE 48000.0

using namespace metal;

// Input: windowed buffer (assume Hann window already applied)
kernel void spectral_extrap(
    device const float2* fftInput           [[ buffer(0) ]], // FFT result: real/imag
    device float2* fftPrevious              [[ buffer(1) ]], // Last frame's real/imag (for phase delta)
    device float2* fftOutput                [[ buffer(2) ]], // Final extrapolated FFT frame
    constant float& stepFactor              [[ buffer(3) ]], // Step increment (1.0 = one full frame)
    uint idx                                [[ thread_position_in_grid ]]
) {
    if (idx >= FFT_SIZE) return;

    // Extract
    float2 curr = fftInput[idx];
    float2 prev = fftPrevious[idx];

    // Calculate magnitudes
    float mag = length(curr);

    // Calculate phase rotation
    float phaseCurr = atan2(curr.y, curr.x);
    float phasePrev = atan2(prev.y, prev.x);
    float deltaPhase = phaseCurr - phasePrev;

    // Predict next phase
    float predPhase = phaseCurr + (deltaPhase * stepFactor);

    // Reconstruct new FFT bin with rotated phase and same magnitude
    fftOutput[idx] = float2(mag * cos(predPhase), mag * sin(predPhase));
}
