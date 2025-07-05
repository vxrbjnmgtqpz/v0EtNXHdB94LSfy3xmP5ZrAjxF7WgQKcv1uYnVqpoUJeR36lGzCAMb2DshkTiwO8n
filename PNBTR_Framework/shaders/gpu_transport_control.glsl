//
// gpu_transport_control.glsl  
// GPU-Native Transport Controller Compute Shader (Vulkan/OpenGL)
//
// Cross-platform equivalent of gpu_transport_control.metal
// Handles PLAY/STOP/PAUSE/RECORD state management directly on GPU
//

#version 450
#extension GL_ARB_gpu_shader_int64 : enable

// Workgroup size - single thread for transport state
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Transport state definitions (must match GPUTransportController.h)
const uint GPU_TRANSPORT_STOPPED = 0u;
const uint GPU_TRANSPORT_PLAYING = 1u;
const uint GPU_TRANSPORT_PAUSED = 2u;
const uint GPU_TRANSPORT_RECORDING = 3u;

// Transport control buffer structure
layout(std140, binding = 0) buffer GPUTransportControlBuffer {
    uint current_state;
    uint64_t play_start_frame;
    uint64_t pause_frame;
    uint64_t record_start_frame;
    uint64_t current_frame;
    float bpm;
    uint samples_per_beat;
    uint network_sync_offset;
    uint frame_counter;
    float position_seconds;
    uint is_network_synced;
    uint padding[5]; // Align to 64 bytes
} transport_buffer;

// Master timebase buffer
layout(std140, binding = 1) buffer MasterTimebaseBuffer {
    uint64_t master_timebase_ns;
} timebase_buffer;

// Timeline events buffer
layout(std140, binding = 2) buffer TimelineEventsBuffer {
    uint event_count;
    uint padding[3];
    
    // Event structure
    struct {
        uint64_t timestamp_ns;
        uint event_type;
        uint event_data;
        uint new_state;
        uint source_peer_id;
    } events[256]; // Max 256 pending events
} timeline_buffer;

//==============================================================================
// TRANSPORT STATE CONTROL FUNCTIONS
//==============================================================================

void transport_control_play(uint64_t trigger_frame)
{
    if (transport_buffer.current_state == GPU_TRANSPORT_PAUSED) {
        // Resume from pause - calculate new start frame accounting for pause duration
        uint64_t pause_duration = trigger_frame - transport_buffer.pause_frame;
        transport_buffer.play_start_frame = transport_buffer.play_start_frame + pause_duration;
    } else {
        // Start fresh
        transport_buffer.play_start_frame = trigger_frame;
    }
    
    transport_buffer.current_state = GPU_TRANSPORT_PLAYING;
    transport_buffer.pause_frame = 0UL;
}

void transport_control_stop(uint64_t trigger_frame)
{
    transport_buffer.current_state = GPU_TRANSPORT_STOPPED;
    transport_buffer.play_start_frame = 0UL;
    transport_buffer.pause_frame = 0UL;
    transport_buffer.record_start_frame = 0UL;
    transport_buffer.position_seconds = 0.0;
}

void transport_control_pause(uint64_t trigger_frame)
{
    if (transport_buffer.current_state == GPU_TRANSPORT_PLAYING || 
        transport_buffer.current_state == GPU_TRANSPORT_RECORDING) {
        transport_buffer.current_state = GPU_TRANSPORT_PAUSED;
        transport_buffer.pause_frame = trigger_frame;
    }
}

void transport_control_record(uint64_t trigger_frame)
{
    transport_buffer.current_state = GPU_TRANSPORT_RECORDING;
    transport_buffer.record_start_frame = trigger_frame;
    transport_buffer.play_start_frame = trigger_frame; // Recording also plays
}

//==============================================================================
// POSITION AND TIMING CALCULATIONS
//==============================================================================

void update_transport_position()
{
    if (transport_buffer.current_state == GPU_TRANSPORT_PLAYING || 
        transport_buffer.current_state == GPU_TRANSPORT_RECORDING) {
        
        // Calculate elapsed time since play start
        uint64_t elapsed_ns = transport_buffer.current_frame - transport_buffer.play_start_frame;
        transport_buffer.position_seconds = float(elapsed_ns) / 1e9;
        
    } else if (transport_buffer.current_state == GPU_TRANSPORT_PAUSED) {
        
        // Maintain position at pause point
        uint64_t elapsed_ns = transport_buffer.pause_frame - transport_buffer.play_start_frame;
        transport_buffer.position_seconds = float(elapsed_ns) / 1e9;
        
    } else {
        // Stopped
        transport_buffer.position_seconds = 0.0;
    }
}

//==============================================================================
// MAIN TRANSPORT CONTROL COMPUTE SHADER
//==============================================================================

void main()
{
    // Update current frame from master GPU timebase
    transport_buffer.current_frame = timebase_buffer.master_timebase_ns;
    
    // Process any pending timeline events
    for (uint i = 0; i < timeline_buffer.event_count; ++i) {
        uint64_t event_timestamp = timeline_buffer.events[i].timestamp_ns;
        
        // Check if event should trigger now
        if (event_timestamp <= timebase_buffer.master_timebase_ns) {
            uint new_state = timeline_buffer.events[i].new_state;
            
            switch (new_state) {
                case GPU_TRANSPORT_PLAYING:
                    transport_control_play(event_timestamp);
                    break;
                case GPU_TRANSPORT_STOPPED:
                    transport_control_stop(event_timestamp);
                    break;
                case GPU_TRANSPORT_PAUSED:
                    transport_control_pause(event_timestamp);
                    break;
                case GPU_TRANSPORT_RECORDING:
                    transport_control_record(event_timestamp);
                    break;
            }
            
            // Mark event as processed by moving last event to this position
            timeline_buffer.events[i] = timeline_buffer.events[timeline_buffer.event_count - 1];
            timeline_buffer.event_count--;
            i--; // Recheck this position
        }
    }
    
    // Update position and timing calculations
    update_transport_position();
    
    // Increment frame counter for continuous operations
    transport_buffer.frame_counter++;
    
    // Memory barrier to ensure writes are visible
    memoryBarrierBuffer();
}

//==============================================================================
// NETWORK SYNCHRONIZATION COMPUTE SHADER
//==============================================================================

// Network sync buffer
layout(std140, binding = 3) buffer NetworkSyncBuffer {
    uint64_t network_sync_frame;
    uint sync_command;
    uint padding[2];
} network_buffer;

// Separate compute shader for network sync
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void network_sync_main()
{
    // Apply network synchronization offset for tight peer sync
    uint64_t adjusted_frame = network_buffer.network_sync_frame + uint64_t(transport_buffer.network_sync_offset);
    
    // Execute synchronized transport command
    switch (network_buffer.sync_command) {
        case 1: // PLAY
            transport_control_play(adjusted_frame);
            break;
        case 2: // STOP  
            transport_control_stop(adjusted_frame);
            break;
        case 3: // PAUSE
            transport_control_pause(adjusted_frame);
            break;
        case 4: // RECORD
            transport_control_record(adjusted_frame);
            break;
    }
    
    transport_buffer.is_network_synced = 1;
    update_transport_position();
    
    // Clear sync command
    network_buffer.sync_command = 0;
    
    memoryBarrierBuffer();
}

//==============================================================================
// BPM AND TEMPO UPDATE COMPUTE SHADER
//==============================================================================

// Tempo update buffer
layout(std140, binding = 4) buffer TempoUpdateBuffer {
    float new_bpm;
    uint64_t sample_rate;
    uint padding[2];
} tempo_buffer;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void tempo_update_main()
{
    transport_buffer.bpm = tempo_buffer.new_bpm;
    
    // Calculate samples per beat for precise timing
    float samples_per_second = float(tempo_buffer.sample_rate);
    float beats_per_second = tempo_buffer.new_bpm / 60.0;
    transport_buffer.samples_per_beat = uint(samples_per_second / beats_per_second);
    
    memoryBarrierBuffer();
}
