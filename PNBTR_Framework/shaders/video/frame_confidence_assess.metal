#include <metal_stdlib>
using namespace metal;

// Frame confidence assessment shader for PNBTR-JVID
// Evaluates prediction quality and flags low-confidence regions

#define CONFIDENCE_BLOCK_SIZE 8
#define MIN_CONFIDENCE_THRESHOLD 0.3

struct ConfidenceMetrics {
    float motion_consistency;      // How consistent motion vectors are
    float temporal_stability;      // How stable pixels are over time
    float spatial_coherence;       // How well neighboring pixels match
    float overall_confidence;      // Combined confidence score
};

kernel void frame_confidence_assess(
    texture2d<float, access::read> predicted_frame    [[ texture(0) ]],
    texture2d<float, access::read> reference_frame    [[ texture(1) ]],  // If available
    texture2d<float, access::write> confidence_map    [[ texture(2) ]],
    device ConfidenceMetrics* block_metrics          [[ buffer(0) ]],
    constant uint& frame_width                       [[ buffer(1) ]],
    constant uint& frame_height                      [[ buffer(2) ]],
    constant bool& has_reference                     [[ buffer(3) ]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= frame_width || gid.y >= frame_height) return;
    
    // Calculate block coordinates
    uint2 block_coord = gid / CONFIDENCE_BLOCK_SIZE;
    uint block_index = block_coord.y * (frame_width / CONFIDENCE_BLOCK_SIZE) + block_coord.x;
    
    float4 predicted_pixel = predicted_frame.read(gid);
    
    // Calculate spatial coherence by comparing with neighbors
    float spatial_coherence = 0.0;
    int neighbor_count = 0;
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            int2 neighbor_pos = int2(gid) + int2(dx, dy);
            if (neighbor_pos.x >= 0 && neighbor_pos.y >= 0 && 
                neighbor_pos.x < int(frame_width) && neighbor_pos.y < int(frame_height)) {
                
                float4 neighbor_pixel = predicted_frame.read(uint2(neighbor_pos));
                float3 diff = abs(predicted_pixel.rgb - neighbor_pixel.rgb);
                float similarity = 1.0 - (diff.r + diff.g + diff.b) / 3.0;
                spatial_coherence += similarity;
                neighbor_count++;
            }
        }
    }
    
    if (neighbor_count > 0) {
        spatial_coherence /= float(neighbor_count);
    }
    
    // Calculate temporal stability (requires frame history - simplified here)
    float temporal_stability = 0.8; // Placeholder - would need frame history
    
    // If we have a reference frame, calculate accuracy-based confidence
    float accuracy_confidence = 1.0;
    if (has_reference) {
        float4 reference_pixel = reference_frame.read(gid);
        float3 error = abs(predicted_pixel.rgb - reference_pixel.rgb);
        float mse = dot(error, error) / 3.0;
        accuracy_confidence = exp(-mse * 10.0); // Exponential falloff for errors
    }
    
    // Motion consistency (would need motion vector data - simplified)
    float motion_consistency = 0.7; // Placeholder
    
    // Combine confidence metrics
    float overall_confidence = (spatial_coherence * 0.3 + 
                               temporal_stability * 0.3 + 
                               accuracy_confidence * 0.3 + 
                               motion_consistency * 0.1);
    
    // Store block-level metrics
    if (gid.x % CONFIDENCE_BLOCK_SIZE == 0 && gid.y % CONFIDENCE_BLOCK_SIZE == 0) {
        block_metrics[block_index].spatial_coherence = spatial_coherence;
        block_metrics[block_index].temporal_stability = temporal_stability;
        block_metrics[block_index].motion_consistency = motion_consistency;
        block_metrics[block_index].overall_confidence = overall_confidence;
    }
    
    // Create confidence visualization
    float4 confidence_color;
    if (overall_confidence > 0.8) {
        // High confidence: green tint
        confidence_color = float4(0.0, 1.0, 0.0, overall_confidence);
    } else if (overall_confidence > 0.5) {
        // Medium confidence: yellow tint
        confidence_color = float4(1.0, 1.0, 0.0, overall_confidence);
    } else {
        // Low confidence: red tint
        confidence_color = float4(1.0, 0.0, 0.0, overall_confidence);
    }
    
    confidence_map.write(confidence_color, gid);
}
