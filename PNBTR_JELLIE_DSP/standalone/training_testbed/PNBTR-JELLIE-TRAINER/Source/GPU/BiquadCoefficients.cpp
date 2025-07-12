#include <cmath>

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Forward declarations to avoid header issues
struct BiquadParams {
    float b0, b1, b2;
    float a1, a2;
    unsigned int frameOffset;
    unsigned int numSamples;
    
    BiquadParams() : b0(0), b1(0), b2(0), a1(0), a2(0), frameOffset(0), numSamples(0) {}
};

class BiquadCoefficients {
public:
    static BiquadParams makeLowPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeHighPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeBandPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeNotchBiquad(float Fs, float Fc, float Q);
    static BiquadParams makePeakingEQ(float Fs, float Fc, float Q, float gainDB);
    static BiquadParams makeLowShelf(float Fs, float Fc, float Q, float gainDB);
    static BiquadParams makeHighShelf(float Fs, float Fc, float Q, float gainDB);
};

// Helper function for dB to linear conversion
float dbToLinear(float dB) {
    return powf(10.0f, dB / 20.0f);
}

// Implementation of biquad coefficient generators

BiquadParams BiquadCoefficients::makeLowPassBiquad(float Fs, float Fc, float Q) {
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    
    float b0 = (1 - cosw) / 2;
    float b1 = 1 - cosw;
    float b2 = (1 - cosw) / 2;
    float a0 = 1 + alpha;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha;
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makeHighPassBiquad(float Fs, float Fc, float Q) {
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    
    float b0 = (1 + cosw) / 2;
    float b1 = -(1 + cosw);
    float b2 = (1 + cosw) / 2;
    float a0 = 1 + alpha;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha;
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makeBandPassBiquad(float Fs, float Fc, float Q) {
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    
    float b0 = alpha;
    float b1 = 0;
    float b2 = -alpha;
    float a0 = 1 + alpha;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha;
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makeNotchBiquad(float Fs, float Fc, float Q) {
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    
    float b0 = 1;
    float b1 = -2 * cosw;
    float b2 = 1;
    float a0 = 1 + alpha;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha;
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makePeakingEQ(float Fs, float Fc, float Q, float gainDB) {
    float A = dbToLinear(gainDB);
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    
    float b0 = 1 + alpha * A;
    float b1 = -2 * cosw;
    float b2 = 1 - alpha * A;
    float a0 = 1 + alpha / A;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha / A;
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makeLowShelf(float Fs, float Fc, float Q, float gainDB) {
    float A = dbToLinear(gainDB);
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    float beta = sqrtf(A) / Q;
    
    float b0 = A * ((A + 1) - (A - 1) * cosw + beta * sinf(omega));
    float b1 = 2 * A * ((A - 1) - (A + 1) * cosw);
    float b2 = A * ((A + 1) - (A - 1) * cosw - beta * sinf(omega));
    float a0 = (A + 1) + (A - 1) * cosw + beta * sinf(omega);
    float a1 = -2 * ((A - 1) + (A + 1) * cosw);
    float a2 = (A + 1) + (A - 1) * cosw - beta * sinf(omega);
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
}

BiquadParams BiquadCoefficients::makeHighShelf(float Fs, float Fc, float Q, float gainDB) {
    float A = dbToLinear(gainDB);
    float omega = 2.0f * PI * Fc / Fs;
    float alpha = sinf(omega) / (2.0f * Q);
    float cosw = cosf(omega);
    float beta = sqrtf(A) / Q;
    
    float b0 = A * ((A + 1) + (A - 1) * cosw + beta * sinf(omega));
    float b1 = -2 * A * ((A - 1) + (A + 1) * cosw);
    float b2 = A * ((A + 1) + (A - 1) * cosw - beta * sinf(omega));
    float a0 = (A + 1) - (A - 1) * cosw + beta * sinf(omega);
    float a1 = 2 * ((A - 1) - (A + 1) * cosw);
    float a2 = (A + 1) - (A - 1) * cosw - beta * sinf(omega);
    
    BiquadParams params;
    params.b0 = b0 / a0;
    params.b1 = b1 / a0;
    params.b2 = b2 / a0;
    params.a1 = a1 / a0;
    params.a2 = a2 / a0;
    params.frameOffset = 0;
    params.numSamples = 0;
    
    return params;
} 