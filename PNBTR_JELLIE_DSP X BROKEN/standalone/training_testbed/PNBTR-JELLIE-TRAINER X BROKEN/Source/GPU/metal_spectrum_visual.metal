#include <metal_stdlib>
using namespace metal;

struct VisualUniforms {
    float4 lowColor;      // Bass frequencies (red/orange)
    float4 midColor;      // Mid frequencies (green/yellow)
    float4 highColor;     // High frequencies (blue/cyan)
    uint   binCount;      // Number of FFT bins
    float  maxMagnitude;  // Dynamic scaling factor
    float  logScale;      // Logarithmic frequency scaling
    uint   armed;         // Track armed state
    float  pulse;         // Pulse animation factor
};

// Main spectrum visualization kernel
kernel void spectrumVisualKernel(
    constant VisualUniforms& uniforms [[buffer(0)]],
    device const float*      bins      [[buffer(1)]],  // FFT magnitude bins
    texture2d<float, access::write> outTexture [[texture(0)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    if (uniforms.armed == 0) {
        // Draw dim background when not armed
        outTexture.write(float4(0.1, 0.1, 0.1, 1.0), gid);
        return;
    }
    
    uint textureWidth = outTexture.get_width();
    uint textureHeight = outTexture.get_height();
    
    float xNorm = float(gid.x) / float(textureWidth);
    float yNorm = float(gid.y) / float(textureHeight);
    
    // Map X coordinate to frequency bin (with logarithmic scaling)
    float logX = pow(xNorm, uniforms.logScale);
    uint bin = min(uint(logX * uniforms.binCount), uniforms.binCount - 1);
    
    // Get magnitude and normalize
    float magnitude = bins[bin] / uniforms.maxMagnitude;
    magnitude = clamp(magnitude, 0.0, 1.0);
    
    // Apply pulse animation
    magnitude *= (1.0 + uniforms.pulse * 0.3);
    
    // DJ-style frequency-based color mapping
    float4 baseColor;
    if (xNorm < 0.33) {
        // Low frequencies: Red to Orange
        float t = xNorm * 3.0;
        baseColor = mix(uniforms.lowColor, 
                       float4(1.0, 0.5, 0.0, 1.0), t);
    } else if (xNorm < 0.66) {
        // Mid frequencies: Orange to Green
        float t = (xNorm - 0.33) * 3.0;
        baseColor = mix(float4(1.0, 0.5, 0.0, 1.0), 
                       uniforms.midColor, t);
    } else {
        // High frequencies: Green to Blue/Cyan
        float t = (xNorm - 0.66) * 3.0;
        baseColor = mix(uniforms.midColor, 
                       uniforms.highColor, t);
    }
    
    // Height-based rendering (spectrum bars)
    float barHeight = magnitude;
    float pixelHeight = 1.0 - yNorm;
    
    if (pixelHeight <= barHeight) {
        // Inside the spectrum bar
        float intensity = pixelHeight / barHeight;
        float4 color = baseColor * intensity;
        
        // Add brightness based on magnitude
        color.rgb += magnitude * 0.2;
        
        // Saturation boost for higher frequencies
        if (xNorm > 0.5) {
            color.rgb = mix(color.rgb, normalize(color.rgb), 0.3);
        }
        
        outTexture.write(color, gid);
    } else {
        // Background with subtle glow
        float4 bgColor = float4(0.05, 0.05, 0.1, 1.0);
        
        // Add glow effect near spectrum bars
        float glowDistance = abs(pixelHeight - barHeight);
        if (glowDistance < 0.1 && magnitude > 0.1) {
            float glowIntensity = (0.1 - glowDistance) * 10.0 * magnitude;
            bgColor += baseColor * glowIntensity * 0.3;
        }
        
        outTexture.write(bgColor, gid);
    }
}

// Peak hold visualization kernel
kernel void peakHoldKernel(
    constant VisualUniforms& uniforms [[buffer(0)]],
    device const float*      bins      [[buffer(1)]],  // Current FFT bins
    device float*            peakBins  [[buffer(2)]],  // Peak hold values
    device float*            peakDecay [[buffer(3)]],  // Peak decay timers
    uint binID                         [[thread_position_in_grid]])
{
    if (binID >= uniforms.binCount) return;
    
    float currentMagnitude = bins[binID];
    float peakMagnitude = peakBins[binID];
    float decayTimer = peakDecay[binID];
    
    // Update peak hold
    if (currentMagnitude > peakMagnitude) {
        peakBins[binID] = currentMagnitude;
        peakDecay[binID] = 0.0;
    } else {
        // Decay peak hold
        decayTimer += 0.016; // Assuming 60 FPS
        if (decayTimer > 1.0) { // 1 second hold
            peakBins[binID] = max(peakMagnitude * 0.95, currentMagnitude);
        }
        peakDecay[binID] = decayTimer;
    }
}

// Smoothing kernel for temporal coherence
kernel void smoothingKernel(
    device const float* inputBins  [[buffer(0)]],
    device float*       outputBins [[buffer(1)]],
    device float*       smoothingBuffer [[buffer(2)]],
    constant float&     smoothingFactor [[buffer(3)]],
    uint binID                          [[thread_position_in_grid]])
{
    if (binID >= 512) return; // Half of FFT_SIZE
    
    float currentValue = inputBins[binID];
    float smoothedValue = smoothingBuffer[binID];
    
    // Exponential smoothing
    smoothedValue = smoothingFactor * currentValue + (1.0 - smoothingFactor) * smoothedValue;
    
    smoothingBuffer[binID] = smoothedValue;
    outputBins[binID] = smoothedValue;
}

// Waveform overlay kernel (for combining with spectrum)
kernel void waveformOverlayKernel(
    constant VisualUniforms& uniforms [[buffer(0)]],
    device const float*      audioBuffer [[buffer(1)]],  // Time-domain audio
    texture2d<float, access::read_write> texture [[texture(0)]],
    uint2 gid                                     [[thread_position_in_grid]])
{
    if (uniforms.armed == 0) return;
    
    uint textureWidth = texture.get_width();
    uint textureHeight = texture.get_height();
    
    float xNorm = float(gid.x) / float(textureWidth);
    float yNorm = float(gid.y) / float(textureHeight);
    
    // Sample audio at this X position
    uint sampleIndex = uint(xNorm * 1024); // Match FFT_SIZE
    float sample = audioBuffer[sampleIndex];
    
    // Convert to Y coordinate (centered)
    float waveY = 0.5 + sample * 0.4; // ±40% of height
    
    // Draw waveform line
    float lineDistance = abs(yNorm - waveY);
    if (lineDistance < 0.01) {
        float4 existingColor = texture.read(gid);
        float4 waveColor = float4(1.0, 1.0, 1.0, 0.7);
        float4 blendedColor = mix(existingColor, waveColor, 0.5);
        texture.write(blendedColor, gid);
    }
} 