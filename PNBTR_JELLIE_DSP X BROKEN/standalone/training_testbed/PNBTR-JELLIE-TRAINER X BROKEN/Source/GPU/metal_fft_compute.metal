#include <metal_stdlib>
using namespace metal;

constant uint FFT_SIZE = 1024; // adjust for resolution

struct FFTUniforms {
    uint frameOffset;
    uint windowSize;
    float sampleRate;
    uint armed;
};

// Cooley-Tukey FFT implementation for Metal
kernel void fftComputeKernel(
    constant FFTUniforms& uniforms [[buffer(0)]],
    device const float*   audioBuffer  [[buffer(1)]],  // mono PCM from ring buffer
    device float*         spectrumBins [[buffer(2)]],  // FFT output (magnitude)
    device float*         realPart     [[buffer(3)]],  // working buffer for real part
    device float*         imagPart     [[buffer(4)]],  // working buffer for imaginary part
    uint threadID                      [[thread_position_in_grid]])
{
    if (uniforms.armed == 0) return;
    
    const uint half = FFT_SIZE / 2;
    if (threadID >= half) return;

    // Initialize working buffers
    if (threadID < uniforms.windowSize) {
        realPart[threadID] = audioBuffer[uniforms.frameOffset + threadID];
        imagPart[threadID] = 0.0;
    } else {
        realPart[threadID] = 0.0;
        imagPart[threadID] = 0.0;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Bit-reversal permutation
    uint j = 0;
    for (uint i = 0; i < FFT_SIZE; i++) {
        if (i < j) {
            // Swap realPart[i] and realPart[j]
            float tempReal = realPart[i];
            float tempImag = imagPart[i];
            realPart[i] = realPart[j];
            imagPart[i] = imagPart[j];
            realPart[j] = tempReal;
            imagPart[j] = tempImag;
        }
        
        uint k = FFT_SIZE >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Cooley-Tukey FFT
    for (uint length = 2; length <= FFT_SIZE; length <<= 1) {
        float angle = -2.0 * M_PI_F / length;
        float wlenReal = cos(angle);
        float wlenImag = sin(angle);
        
        for (uint i = threadID; i < FFT_SIZE; i += length) {
            float wReal = 1.0;
            float wImag = 0.0;
            
            for (uint j = 0; j < length / 2; j++) {
                uint u = i + j;
                uint v = i + j + length / 2;
                
                if (u < FFT_SIZE && v < FFT_SIZE) {
                    float uReal = realPart[u];
                    float uImag = imagPart[u];
                    float vReal = realPart[v];
                    float vImag = imagPart[v];
                    
                    float tempReal = wReal * vReal - wImag * vImag;
                    float tempImag = wReal * vImag + wImag * vReal;
                    
                    realPart[u] = uReal + tempReal;
                    imagPart[u] = uImag + tempImag;
                    realPart[v] = uReal - tempReal;
                    imagPart[v] = uImag - tempImag;
                }
                
                float nextWReal = wReal * wlenReal - wImag * wlenImag;
                float nextWImag = wReal * wlenImag + wImag * wlenReal;
                wReal = nextWReal;
                wImag = nextWImag;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }
    
    // Calculate magnitude spectrum
    if (threadID < half) {
        float re = realPart[threadID];
        float im = imagPart[threadID];
        spectrumBins[threadID] = sqrt(re * re + im * im);
    }
}

// Window function application (Hann window)
kernel void applyWindowKernel(
    constant FFTUniforms& uniforms [[buffer(0)]],
    device float*         audioBuffer [[buffer(1)]],
    uint threadID                     [[thread_position_in_grid]])
{
    if (threadID >= uniforms.windowSize) return;
    
    float n = float(threadID);
    float N = float(uniforms.windowSize);
    float window = 0.5 * (1.0 - cos(2.0 * M_PI_F * n / (N - 1.0)));
    
    audioBuffer[uniforms.frameOffset + threadID] *= window;
} 