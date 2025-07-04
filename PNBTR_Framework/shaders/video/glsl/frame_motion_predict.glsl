#version 450

// Frame motion prediction compute shader for PNBTR-JVID (GLSL)
// Cross-platform equivalent of Metal shader

#define MAX_MOTION_VECTORS 1024
#define BLOCK_SIZE 16

struct MotionVector {
    vec2 velocity;          // Motion vector in pixels/frame
    float confidence;       // Prediction confidence (0-1)
    uvec2 position;         // Block position in frame
};

struct FrameMetadata {
    uint width;
    uint height;
    uint frame_number;
    uint64_t timestamp_us;
};

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0, rgba8) uniform readonly image2D previous_frame;
layout(binding = 1, rgba8) uniform readonly image2D current_frame;
layout(binding = 2, rgba8) uniform writeonly image2D predicted_frame;

layout(std430, binding = 3) writeonly buffer MotionVectorBuffer {
    MotionVector motion_vectors[MAX_MOTION_VECTORS];
};

layout(std430, binding = 4) uniform MetadataBuffer {
    FrameMetadata metadata;
};

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    
    // Get frame dimensions
    uvec2 frame_size = uvec2(metadata.width, metadata.height);
    if (gid.x >= frame_size.x || gid.y >= frame_size.y) return;
    
    // Calculate block coordinates
    uvec2 block_coord = gid / BLOCK_SIZE;
    uint block_index = block_coord.y * (frame_size.x / BLOCK_SIZE) + block_coord.x;
    
    if (block_index >= MAX_MOTION_VECTORS) return;
    
    // Sample current and previous pixels
    vec4 curr_pixel = imageLoad(current_frame, ivec2(gid));
    vec4 prev_pixel = imageLoad(previous_frame, ivec2(gid));
    
    // Simple motion estimation using block matching
    vec2 best_motion = vec2(0.0);
    float min_sad = 3.402823e+38; // GLSL equivalent of MAXFLOAT
    
    // Search window for motion vectors (±8 pixels)
    for (int dy = -8; dy <= 8; dy += 2) {
        for (int dx = -8; dx <= 8; dx += 2) {
            ivec2 search_pos = ivec2(gid) + ivec2(dx, dy);
            
            // Bounds check
            if (search_pos.x < 0 || search_pos.y < 0 || 
                search_pos.x >= int(frame_size.x) || search_pos.y >= int(frame_size.y)) {
                continue;
            }
            
            vec4 search_pixel = imageLoad(previous_frame, search_pos);
            
            // Calculate Sum of Absolute Differences (SAD)
            vec3 diff = abs(curr_pixel.rgb - search_pixel.rgb);
            float sad = diff.r + diff.g + diff.b;
            
            if (sad < min_sad) {
                min_sad = sad;
                best_motion = vec2(dx, dy);
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
    vec2 predicted_pos = vec2(gid) + best_motion;
    
    // Bilinear interpolation for sub-pixel accuracy
    if (predicted_pos.x >= 0 && predicted_pos.y >= 0 && 
        predicted_pos.x < frame_size.x - 1 && predicted_pos.y < frame_size.y - 1) {
        
        ivec2 p0 = ivec2(floor(predicted_pos));
        ivec2 p1 = p0 + ivec2(1, 0);
        ivec2 p2 = p0 + ivec2(0, 1);
        ivec2 p3 = p0 + ivec2(1, 1);
        
        vec2 frac = predicted_pos - vec2(p0);
        
        vec4 c0 = imageLoad(current_frame, p0);
        vec4 c1 = imageLoad(current_frame, p1);
        vec4 c2 = imageLoad(current_frame, p2);
        vec4 c3 = imageLoad(current_frame, p3);
        
        // Bilinear interpolation
        vec4 interpolated = mix(mix(c0, c1, frac.x), mix(c2, c3, frac.x), frac.y);
        
        imageStore(predicted_frame, ivec2(gid), interpolated);
    } else {
        // Fallback: copy current pixel if motion prediction goes out of bounds
        imageStore(predicted_frame, ivec2(gid), curr_pixel);
    }
}
