#version 450

#define LPC_ORDER 8                     // Prediction depth
#define SAMPLE_WINDOW 256              // 5.3 ms @ 48kHz
#define PREDICTION_LENGTH 480          // 10 ms output

layout(std430, binding = 0) readonly buffer RecentSamples {
    float recentSamples[SAMPLE_WINDOW];
};

layout(std430, binding = 1) writeonly buffer PredictedSamples {
    float predictedSamples[PREDICTION_LENGTH];
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid > 0) return; // LPC must run single-threaded due to matrix steps

    float r[LPC_ORDER + 1];
    float a[LPC_ORDER + 1];
    float e = 0.0;

    // Initialize arrays
    for (int i = 0; i <= LPC_ORDER; ++i) {
        r[i] = 0.0;
        a[i] = 0.0;
    }

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
