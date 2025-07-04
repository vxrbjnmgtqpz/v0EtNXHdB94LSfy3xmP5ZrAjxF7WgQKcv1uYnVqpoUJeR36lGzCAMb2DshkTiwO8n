#include <metal_stdlib>
using namespace metal;

#define LPC_ORDER 8                     // Prediction depth
#define SAMPLE_WINDOW 256              // 5.3 ms @ 48kHz
#define PREDICTION_LENGTH 480          // 10 ms output

using namespace metal;

kernel void lpc_model(
    device const float* recentSamples     [[ buffer(0) ]],
    device float* predictedSamples        [[ buffer(1) ]],
    uint tid                              [[ thread_position_in_grid ]]
) {
    if (tid > 0) return; // LPC must run single-threaded due to matrix steps

    float r[LPC_ORDER + 1] = {0.0};
    float a[LPC_ORDER + 1] = {0.0};
    float e = 0.0;

    // Step 1: Autocorrelation
    for (int lag = 0; lag <= LPC_ORDER; ++lag) {
        for (int i = lag; i < SAMPLE_WINDOW; ++i) {
            r[lag] += recentSamples[i] * recentSamples[i - lag];
        }
    }

    // Step 2: Levinson-Durbin recursion
    a[0] = 1.0;
    e = r[0];

    for (int i = 1; i <= LPC_ORDER; ++i) {
        float k = r[i];
        for (int j = 1; j < i; ++j) {
            k -= a[j] * r[i - j];
        }
        k /= e;

        float newA[LPC_ORDER + 1];
        for (int j = 0; j <= i; ++j) {
            newA[j] = a[j];
        }
        for (int j = 1; j < i; ++j) {
            newA[j] -= k * a[i - j];
        }
        newA[i] = k;

        for (int j = 0; j <= i; ++j) {
            a[j] = newA[j];
        }

        e *= (1.0 - k * k);
    }

    // Step 3: Predict forward samples
    for (int i = 0; i < PREDICTION_LENGTH; ++i) {
        float pred = 0.0;
        for (int j = 1; j <= LPC_ORDER; ++j) {
            int idx = SAMPLE_WINDOW - j + i;
            if (idx < SAMPLE_WINDOW)
                pred -= a[j] * recentSamples[idx];
            else
                pred -= a[j] * predictedSamples[idx - SAMPLE_WINDOW];
        }
        predictedSamples[i] = pred;
    }
}
