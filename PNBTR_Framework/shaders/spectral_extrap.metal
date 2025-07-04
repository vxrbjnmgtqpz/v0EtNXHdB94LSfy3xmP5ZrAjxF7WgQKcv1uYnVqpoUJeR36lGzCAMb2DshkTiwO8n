#include <metal_stdlib>
using namespace metal;

#define WINDOW_SIZE 512              // ~10.6 ms @ 48kHz
#define PREDICT_FRAMES 3            // Predict 3 future spectral frames (~30 ms total)
#define FFT_SIZE (WINDOW_SIZE / 2)

using namespace metal;

// I/O Buffers
kernel void spectral_extrap(
    device const float* recentSamples   [[ buffer(0) ]],
    device float* predictedSamples      [[ buffer(1) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
    // Use Hann window
    float windowed[WINDOW_SIZE];
    for (uint i = 0; i < WINDOW_SIZE; ++i) {
        float hann = 0.5 * (1.0 - cos(2.0 * M_PI_F * float(i) / float(WINDOW_SIZE - 1)));
        windowed[i] = recentSamples[i] * hann;
    }

    // Compute simple DFT (for clarity, not FFT-optimized)
    float mag[FFT_SIZE];
    float phase[FFT_SIZE];
    for (uint k = 0; k < FFT_SIZE; ++k) {
        float real = 0.0;
        float imag = 0.0;
        for (uint n = 0; n < WINDOW_SIZE; ++n) {
            float angle = 2.0 * M_PI_F * float(k * n) / float(WINDOW_SIZE);
            real += windowed[n] * cos(angle);
            imag -= windowed[n] * sin(angle);
        }
        mag[k] = sqrt(real * real + imag * imag);
        phase[k] = atan2(imag, real);
    }

    // Simple spectral extrapolation: hold mag, shift phase linearly
    for (uint f = 0; f < PREDICT_FRAMES; ++f) {
        for (uint n = 0; n < WINDOW_SIZE; ++n) {
            float sample = 0.0;
            for (uint k = 0; k < FFT_SIZE; ++k) {
                float shiftedPhase = phase[k] + float(f) * 2.0 * M_PI_F * float(k) / float(WINDOW_SIZE);
                sample += mag[k] * cos((2.0 * M_PI_F * float(k) * float(n) / float(WINDOW_SIZE)) + shiftedPhase);
            }
            sample /= float(FFT_SIZE); // Normalize
            uint outIdx = f * WINDOW_SIZE + n;
            predictedSamples[outIdx] = sample;
        }
    }
}
