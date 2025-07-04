#include <metal_stdlib>
using namespace metal;

// Pixel temporal tracking shader for PNBTR-JVID
// Tracks per-pixel temporal evolution to predict future values

#define TEMPORAL_WINDOW 8       // Number of frames to analyze
#define PREDICTION_HORIZON 16   // Number of frames to predict

struct PixelHistory {
    float4 values[TEMPORAL_WINDOW];     // Recent pixel values
    float temporal_gradient;            // Rate of change
    float stability_score;              // How stable this pixel is
};

kernel void pixel_temporal_track(
    texture2d_array<float, access::read> frame_history   [[ texture(0) ]],  // Recent frames
    texture2d<float, access::write> predicted_frame      [[ texture(1) ]],  // Output prediction
    device PixelHistory* pixel_cache                     [[ buffer(0) ]],   // Temporal cache
    constant uint& frame_width                           [[ buffer(1) ]],
    constant uint& frame_height                          [[ buffer(2) ]],
    constant uint& history_depth                         [[ buffer(3) ]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= frame_width || gid.y >= frame_height) return;
    
    uint pixel_index = gid.y * frame_width + gid.x;
    
    // Gather temporal history for this pixel
    float4 history[TEMPORAL_WINDOW];
    for (uint i = 0; i < min(history_depth, TEMPORAL_WINDOW); ++i) {
        history[i] = frame_history.read(gid, i);
    }
    
    // Calculate temporal gradient (rate of change)
    float3 gradient = float3(0.0);
    float weight_sum = 0.0;
    
    for (uint i = 1; i < min(history_depth, TEMPORAL_WINDOW); ++i) {
        float weight = float(i) / float(TEMPORAL_WINDOW); // More recent frames have higher weight
        float3 diff = history[i].rgb - history[i-1].rgb;
        gradient += diff * weight;
        weight_sum += weight;
    }
    
    if (weight_sum > 0.0) {
        gradient /= weight_sum;
    }
    
    // Calculate stability score (how predictable this pixel is)
    float variance = 0.0;
    float3 mean_color = float3(0.0);
    
    for (uint i = 0; i < min(history_depth, TEMPORAL_WINDOW); ++i) {
        mean_color += history[i].rgb;
    }
    mean_color /= float(min(history_depth, TEMPORAL_WINDOW));
    
    for (uint i = 0; i < min(history_depth, TEMPORAL_WINDOW); ++i) {
        float3 diff = history[i].rgb - mean_color;
        variance += dot(diff, diff);
    }
    variance /= float(min(history_depth, TEMPORAL_WINDOW));
    
    float stability = 1.0 / (1.0 + variance); // Higher stability for lower variance
    
    // Store in pixel cache
    pixel_cache[pixel_index].temporal_gradient = length(gradient);
    pixel_cache[pixel_index].stability_score = stability;
    
    for (uint i = 0; i < min(history_depth, TEMPORAL_WINDOW); ++i) {
        pixel_cache[pixel_index].values[i] = history[i];
    }
    
    // Predict next pixel value using temporal model
    float4 latest_frame = history[0]; // Most recent frame
    float4 predicted_pixel;
    
    if (stability > 0.7) {
        // High stability: use linear extrapolation
        predicted_pixel.rgb = latest_frame.rgb + gradient;
        predicted_pixel.a = latest_frame.a;
    } else if (stability > 0.3) {
        // Medium stability: blend linear prediction with latest value
        float4 linear_pred;
        linear_pred.rgb = latest_frame.rgb + gradient * 0.5;
        linear_pred.a = latest_frame.a;
        
        predicted_pixel = mix(latest_frame, linear_pred, stability);
    } else {
        // Low stability: just copy latest frame (safest prediction)
        predicted_pixel = latest_frame;
    }
    
    // Clamp to valid color range
    predicted_pixel = clamp(predicted_pixel, 0.0, 1.0);
    
    predicted_frame.write(predicted_pixel, gid);
}
