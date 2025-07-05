//
// gpu_transport_control.metal
// GPU-Native Transport Controller Compute Shader
//
// Handles PLAY/STOP/PAUSE/RECORD state management directly on GPU
// with precise frame-accurate timing and network synchronization
//

#include <metal_stdlib>
using namespace metal;

// Transport state definitions (must match GPUTransportController.h)
enum class GPUTransportState : uint32_t {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
    Recording = 3
};

// Transport control buffer structure
struct GPUTransportControlBuffer {
    GPUTransportState current_state;
    uint64_t play_start_frame;
    uint64_t pause_frame;
    uint64_t record_start_frame;
    uint64_t current_frame;
    float bpm;
    uint32_t samples_per_beat;
    uint32_t network_sync_offset;
    uint32_t frame_counter;
    float position_seconds;
    uint32_t is_network_synced;
    uint32_t padding[5]; // Align to 64 bytes
};

// Timeline event for synchronization
struct GPUTimelineEvent {
    uint64_t timestamp_ns;
    uint32_t event_type;
    uint32_t event_data;
    GPUTransportState new_state;
    uint32_t source_peer_id;
};

//==============================================================================
// MAIN TRANSPORT CONTROL COMPUTE KERNEL
//==============================================================================

kernel void gpu_transport_control_update(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device uint64_t& master_timebase_ns [[ buffer(1) ]],
    device GPUTimelineEvent* timeline_events [[ buffer(2) ]],
    device uint32_t& event_count [[ buffer(3) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
)
{
    // Only use thread 0 - transport is singular state
    if (thread_position_in_grid != 0) return;
    
    // Update current frame from master GPU timebase
    transport_buffer.current_frame = master_timebase_ns;
    
    // Process any pending timeline events
    for (uint32_t i = 0; i < event_count; ++i) {
        GPUTimelineEvent event = timeline_events[i];
        
        // Check if event should trigger now
        if (event.timestamp_ns <= master_timebase_ns) {
            switch (event.new_state) {
                case GPUTransportState::Playing:
                    transport_control_play(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Stopped:
                    transport_control_stop(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Paused:
                    transport_control_pause(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Recording:
                    transport_control_record(transport_buffer, event.timestamp_ns);
                    break;
            }
            
            // Mark event as processed by moving it to end
            timeline_events[i] = timeline_events[event_count - 1];
            event_count--;
            i--; // Recheck this position
        }
    }
    
    // Update position and timing calculations
    update_transport_position(transport_buffer);
    
    // Increment frame counter for continuous operations
    transport_buffer.frame_counter++;
}

//==============================================================================
// TRANSPORT STATE CONTROL FUNCTIONS
//==============================================================================

inline void transport_control_play(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame)
{
    if (buffer.current_state == GPUTransportState::Paused) {
        // Resume from pause - calculate new start frame accounting for pause duration
        uint64_t pause_duration = trigger_frame - buffer.pause_frame;
        buffer.play_start_frame = buffer.play_start_frame + pause_duration;
    } else {
        // Start fresh
        buffer.play_start_frame = trigger_frame;
    }
    
    buffer.current_state = GPUTransportState::Playing;
    buffer.pause_frame = 0;
}

inline void transport_control_stop(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame)
{
    buffer.current_state = GPUTransportState::Stopped;
    buffer.play_start_frame = 0;
    buffer.pause_frame = 0;
    buffer.record_start_frame = 0;
    buffer.position_seconds = 0.0f;
}

inline void transport_control_pause(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame)
{
    if (buffer.current_state == GPUTransportState::Playing || 
        buffer.current_state == GPUTransportState::Recording) {
        buffer.current_state = GPUTransportState::Paused;
        buffer.pause_frame = trigger_frame;
    }
}

inline void transport_control_record(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame)
{
    buffer.current_state = GPUTransportState::Recording;
    buffer.record_start_frame = trigger_frame;
    buffer.play_start_frame = trigger_frame; // Recording also plays
}

//==============================================================================
// POSITION AND TIMING CALCULATIONS
//==============================================================================

inline void update_transport_position(device GPUTransportControlBuffer& buffer)
{
    if (buffer.current_state == GPUTransportState::Playing || 
        buffer.current_state == GPUTransportState::Recording) {
        
        // Calculate elapsed time since play start
        uint64_t elapsed_ns = buffer.current_frame - buffer.play_start_frame;
        buffer.position_seconds = float(elapsed_ns) / 1e9f;
        
    } else if (buffer.current_state == GPUTransportState::Paused) {
        
        // Maintain position at pause point
        uint64_t elapsed_ns = buffer.pause_frame - buffer.play_start_frame;
        buffer.position_seconds = float(elapsed_ns) / 1e9f;
        
    } else {
        // Stopped
        buffer.position_seconds = 0.0f;
    }
}

//==============================================================================
// NETWORK SYNCHRONIZATION SUPPORT
//==============================================================================

kernel void gpu_transport_network_sync(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device uint64_t& network_sync_frame [[ buffer(1) ]],
    device uint32_t& sync_command [[ buffer(2) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
)
{
    if (thread_position_in_grid != 0) return;
    
    // Apply network synchronization offset for tight peer sync
    uint64_t adjusted_frame = network_sync_frame + transport_buffer.network_sync_offset;
    
    // Execute synchronized transport command
    switch (sync_command) {
        case 1: // PLAY
            transport_control_play(transport_buffer, adjusted_frame);
            break;
        case 2: // STOP  
            transport_control_stop(transport_buffer, adjusted_frame);
            break;
        case 3: // PAUSE
            transport_control_pause(transport_buffer, adjusted_frame);
            break;
        case 4: // RECORD
            transport_control_record(transport_buffer, adjusted_frame);
            break;
    }
    
    transport_buffer.is_network_synced = 1;
    update_transport_position(transport_buffer);
}

//==============================================================================
// BPM AND TEMPO CALCULATIONS  
//==============================================================================

kernel void gpu_transport_tempo_update(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device float& new_bpm [[ buffer(1) ]],
    device uint64_t& sample_rate [[ buffer(2) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
)
{
    if (thread_position_in_grid != 0) return;
    
    transport_buffer.bpm = new_bpm;
    
    // Calculate samples per beat for precise timing
    float samples_per_second = float(sample_rate);
    float beats_per_second = new_bpm / 60.0f;
    transport_buffer.samples_per_beat = uint32_t(samples_per_second / beats_per_second);
}
