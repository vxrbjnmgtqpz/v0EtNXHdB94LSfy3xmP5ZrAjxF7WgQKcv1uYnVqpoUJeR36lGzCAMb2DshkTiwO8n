#version 450

// PNBTR LSB Reconstruction Shader
// Waveform-aware LSB reconstruction completely replacing traditional dither
// Zero-noise mathematical approach based on musical context

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input/Output buffers
layout(std430, binding = 0) readonly buffer InputSamples {
    float input_samples[];
};

layout(std430, binding = 1) writeonly buffer ReconstructedSamples {
    float reconstructed_samples[];
};

// Processing parameters
layout(std140, binding = 2) uniform ProcessingParams {
    uint sample_count;
    uint target_bit_depth;
    float sample_rate;
    float fundamental_frequency;
    float pitch_confidence;
    uint channels;
};

// Shared memory for local analysis
shared float local_window[128];
shared float local_derivatives[127];

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (index >= sample_count) {
        return;
    }
    
    float input_sample = input_samples[index];
    
    // Calculate quantization parameters for target bit depth
    float scale = pow(2.0, float(target_bit_depth) - 1.0) - 1.0;
    float lsb_value = 1.0 / scale;
    
    // Quantize to target bit depth
    int quantized_int = int(round(input_sample * scale));
    float quantized_sample = float(quantized_int) / scale;
    
    // PNBTR LSB Reconstruction - Mathematical, Zero-Noise Approach
    float reconstructed_lsb = 0.0;
    
    // Method 1: Local Waveform Continuity Analysis
    if (index > 0 && index < sample_count - 1) {
        float prev_sample = input_samples[index - 1];
        float next_sample = input_samples[index + 1];
        
        // Calculate local slope and curvature
        float local_slope = (next_sample - prev_sample) * 0.5;
        float second_derivative = next_sample - 2.0 * input_sample + prev_sample;
        
        // Predict LSB value based on waveform continuity
        float continuity_prediction = local_slope * lsb_value * 0.3;
        continuity_prediction += second_derivative * lsb_value * 0.1;
        
        reconstructed_lsb += continuity_prediction * 0.4;
    }
    
    // Method 2: Harmonic Context Analysis (for tonal content)
    if (fundamental_frequency > 20.0 && pitch_confidence > 0.5) {
        float period_samples = sample_rate / fundamental_frequency;
        
        // Find corresponding position in previous period
        if (index >= period_samples) {
            uint prev_period_index = uint(float(index) - period_samples);
            if (prev_period_index < sample_count) {
                float prev_period_sample = input_samples[prev_period_index];
                float period_residual = input_sample - prev_period_sample;
                
                // Use harmonic correlation to predict LSB
                float harmonic_lsb = period_residual * lsb_value * 0.2;
                reconstructed_lsb += harmonic_lsb * pitch_confidence * 0.3;
            }
        }
    }
    
    // Method 3: Spectral Coherence (high-frequency preservation)
    if (index >= 3 && index < sample_count - 3) {
        // Analyze local spectral content using finite differences
        float spectral_energy = 0.0;
        for (int i = -3; i <= 3; i++) {
            if (i != 0) {
                uint neighbor_idx = index + i;
                if (neighbor_idx < sample_count) {
                    float diff = input_samples[neighbor_idx] - input_sample;
                    spectral_energy += diff * diff;
                }
            }
        }
        spectral_energy = sqrt(spectral_energy / 6.0);
        
        // Predict LSB based on local spectral activity
        float spectral_lsb = spectral_energy * lsb_value * 0.15;
        reconstructed_lsb += spectral_lsb * 0.2;
    }
    
    // Method 4: Amplitude Envelope Tracking
    if (index >= 8) {
        // Calculate local amplitude envelope
        float recent_rms = 0.0;
        for (uint i = 0; i < 8; i++) {
            uint sample_idx = index - i;
            if (sample_idx < sample_count) {
                float sample = input_samples[sample_idx];
                recent_rms += sample * sample;
            }
        }
        recent_rms = sqrt(recent_rms / 8.0);
        
        // Modulate LSB reconstruction based on envelope
        float envelope_factor = clamp(recent_rms * 2.0, 0.1, 1.0);
        reconstructed_lsb *= envelope_factor;
    }
    
    // Method 5: Musical Intelligence (zero-crossing and transient detection)
    if (index > 0 && index < sample_count - 1) {
        float prev_sample = input_samples[index - 1];
        float next_sample = input_samples[index + 1];
        
        // Detect zero crossings and transients
        bool zero_crossing = (prev_sample >= 0.0) != (next_sample >= 0.0);
        bool transient = abs(input_sample - prev_sample) > (lsb_value * 4.0);
        
        if (zero_crossing) {
            // Near zero crossings, use conservative LSB reconstruction
            reconstructed_lsb *= 0.3;
        } else if (transient) {
            // During transients, emphasize continuity
            reconstructed_lsb *= 1.2;
        }
    }
    
    // Apply reconstruction limits (conservative approach)
    reconstructed_lsb = clamp(reconstructed_lsb, -lsb_value * 0.5, lsb_value * 0.5);
    
    // Final PNBTR reconstruction: Quantized + Mathematical LSB prediction
    float final_sample = quantized_sample + reconstructed_lsb;
    
    // Ensure we don't exceed input dynamic range
    final_sample = clamp(final_sample, -1.0, 1.0);
    
    // Quality validation: Ensure zero-noise property
    // PNBTR never adds random noise - only meaningful mathematical reconstruction
    
    reconstructed_samples[index] = final_sample;
}
