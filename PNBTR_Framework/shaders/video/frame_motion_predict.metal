#include <metal_stdlib>
using namespace metal;

// Frame motion prediction shader for PNBTR-JVID
// Analyzes optical flow between recent frames to predict motion vectors

#define MAX_MOTION_VECTORS 1024
#define BLOCK_SIZE 16

struct MotionVector {
    float2 velocity;        // Motion vector in pixels/frame
    float confidence;       // Prediction confidence (0-1)
    uint2 position;         // Block position in frame
};

struct FrameMetadata {
    uint width;
    uint height;
    uint frame_number;
    uint64_t timestamp_us;
};

kernel void frame_motion_predict(
    texture2d<float, access::read> previous_frame     [[ texture(0) ]],
    texture2d<float, access::read> current_frame      [[ texture(1) ]],
    texture2d<float, access::write> predicted_frame   [[ texture(2) ]],
    device MotionVector* motion_vectors               [[ buffer(0) ]],
    constant FrameMetadata& metadata                  [[ buffer(1) ]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Get frame dimensions
    uint2 frame_size = uint2(metadata.width, metadata.height);
    if (gid.x >= frame_size.x || gid.y >= frame_size.y) return;
    
    // Calculate block coordinates
    uint2 block_coord = gid / BLOCK_SIZE;
    uint block_index = block_coord.y * (frame_size.x / BLOCK_SIZE) + block_coord.x;
    
    if (block_index >= MAX_MOTION_VECTORS) return;
    
    // Sample current and previous pixels
    float4 curr_pixel = current_frame.read(gid);
    float4 prev_pixel = previous_frame.read(gid);
    
    // Simple motion estimation using block matching
    float2 best_motion = float2(0.0);
    float min_sad = MAXFLOAT;
    
    // Search window for motion vectors (±8 pixels)
    for (int dy = -8; dy <= 8; dy += 2) {
        for (int dx = -8; dx <= 8; dx += 2) {
            int2 search_pos = int2(gid) + int2(dx, dy);
            
            // Bounds check
            if (search_pos.x < 0 || search_pos.y < 0 || 
                search_pos.x >= int(frame_size.x) || search_pos.y >= int(frame_size.y)) {
                continue;
            }
            
            float4 search_pixel = previous_frame.read(uint2(search_pos));
            
            // Calculate Sum of Absolute Differences (SAD)
            float3 diff = abs(curr_pixel.rgb - search_pixel.rgb);
            float sad = diff.r + diff.g + diff.b;
            
            if (sad < min_sad) {
                min_sad = sad;
                best_motion = float2(dx, dy);
            }
        }
    }
    
    // Store motion vector for this block
    if (gid.x % BLOCK_SIZE == 0 && gid.y % BLOCK_SIZE == 0) {
        motion_vectors[block_index].velocity = best_motion;
        motion_vectors[block_index].confidence = 1.0 - (min_sad / 3.0); // Normalize confidence
        motion_vectors[block_index].position = block_coord;
    }
    
    // Predict next frame pixel using motion vector
    float2 predicted_pos = float2(gid) + best_motion;
    
    // Bilinear interpolation for sub-pixel accuracy
    if (predicted_pos.x >= 0 && predicted_pos.y >= 0 && 
        predicted_pos.x < frame_size.x - 1 && predicted_pos.y < frame_size.y - 1) {
        
        uint2 p0 = uint2(floor(predicted_pos));
        uint2 p1 = p0 + uint2(1, 0);
        uint2 p2 = p0 + uint2(0, 1);
        uint2 p3 = p0 + uint2(1, 1);
        
        float2 frac = predicted_pos - float2(p0);
        
        float4 c0 = current_frame.read(p0);
        float4 c1 = current_frame.read(p1);
        float4 c2 = current_frame.read(p2);
        float4 c3 = current_frame.read(p3);
        
        // Bilinear interpolation
        float4 interpolated = mix(mix(c0, c1, frac.x), mix(c2, c3, frac.x), frac.y);
        
        predicted_frame.write(interpolated, gid);
    } else {
        // Fallback: copy current pixel if motion prediction goes out of bounds
        predicted_frame.write(curr_pixel, gid);
    }
}
