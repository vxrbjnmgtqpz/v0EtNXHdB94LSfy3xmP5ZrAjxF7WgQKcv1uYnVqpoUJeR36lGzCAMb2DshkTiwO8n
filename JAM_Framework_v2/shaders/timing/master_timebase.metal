#include <metal_stdlib>
using namespace metal;

struct GPUTimebaseState {
    atomic_uint current_frame;
    atomic_uint sample_rate;
    atomic_uint bpm;
    atomic_uint beat_position;
    atomic_uint transport_state; // 0=stop, 1=play, 2=pause, 3=record
    atomic_uint quantum_frames;
    atomic_uint timestamp_ns_high; // High 32 bits of nanosecond timestamp
    atomic_uint timestamp_ns_low;  // Low 32 bits of nanosecond timestamp
    atomic_uint tempo_scale; // Fixed point: 1000 = 1.0x
    atomic_uint metronome_enable;
    atomic_uint loop_start;
    atomic_uint loop_end;
    atomic_uint loop_enable;
};

struct NetworkSyncEvent {
    uint peer_id;
    uint event_type; // 0=sync_request, 1=sync_response, 2=transport_change
    uint timestamp;
    uint data[4]; // Event-specific data
};

struct MIDIEvent {
    uint timestamp;
    uint channel;
    uint note_velocity;
    uint controller_value;
    uint event_type; // 0=note_on, 1=note_off, 2=cc, 3=pitchbend
};

struct AudioSyncEvent {
    uint buffer_index;
    uint frame_offset;
    uint sample_count;
    uint sync_type; // 0=buffer_ready, 1=xrun, 2=format_change
};

// Master timebase compute kernel - runs continuously at sample rate precision
kernel void master_timebase_tick(
    device GPUTimebaseState& timebase [[buffer(0)]],
    device NetworkSyncEvent* network_queue [[buffer(1)]],
    device MIDIEvent* midi_queue [[buffer(2)]],
    device AudioSyncEvent* audio_queue [[buffer(3)]],
    device atomic_uint& network_queue_size [[buffer(4)]],
    device atomic_uint& midi_queue_size [[buffer(5)]],
    device atomic_uint& audio_queue_size [[buffer(6)]],
    constant uint& delta_frames [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return; // Only one thread updates the master timebase
    
    uint current_state = atomic_load_explicit(&timebase.transport_state, memory_order_relaxed);
    
    if (current_state == 1 || current_state == 3) { // Playing or Recording
        // Advance the master frame counter
        uint old_frame = atomic_fetch_add_explicit(&timebase.current_frame, delta_frames, memory_order_relaxed);
        uint new_frame = old_frame + delta_frames;
        
        // Update timestamp with nanosecond precision using 64-bit split
        uint sample_rate = atomic_load_explicit(&timebase.sample_rate, memory_order_relaxed);
        ulong new_timestamp = (ulong(new_frame) * 1000000000UL) / sample_rate;
        atomic_store_explicit(&timebase.timestamp_ns_high, uint(new_timestamp >> 32), memory_order_relaxed);
        atomic_store_explicit(&timebase.timestamp_ns_low, uint(new_timestamp & 0xFFFFFFFF), memory_order_relaxed);
        
        // Calculate beat position from BPM
        uint bpm = atomic_load_explicit(&timebase.bpm, memory_order_relaxed);
        uint frames_per_beat = (sample_rate * 60) / bpm;
        uint beat_pos = new_frame / frames_per_beat;
        atomic_store_explicit(&timebase.beat_position, beat_pos, memory_order_relaxed);
        
        // Handle loop boundaries
        uint loop_enable = atomic_load_explicit(&timebase.loop_enable, memory_order_relaxed);
        if (loop_enable) {
            uint loop_start = atomic_load_explicit(&timebase.loop_start, memory_order_relaxed);
            uint loop_end = atomic_load_explicit(&timebase.loop_end, memory_order_relaxed);
            
            if (new_frame >= loop_end) {
                atomic_store_explicit(&timebase.current_frame, loop_start, memory_order_relaxed);
            }
        }
    }
}

// Network sync dispatch kernel  
kernel void network_sync_dispatch(
    device GPUTimebaseState& timebase [[buffer(0)]],
    device NetworkSyncEvent* network_queue [[buffer(1)]],
    device atomic_uint& network_queue_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint queue_size = atomic_load_explicit(&network_queue_size, memory_order_relaxed);
    if (gid >= queue_size) return;
    
    NetworkSyncEvent event = network_queue[gid];
    // Reconstruct 64-bit timestamp from two 32-bit atomics
    uint timestamp_high = atomic_load_explicit(&timebase.timestamp_ns_high, memory_order_relaxed);
    uint timestamp_low = atomic_load_explicit(&timebase.timestamp_ns_low, memory_order_relaxed);
    uint current_timestamp = uint((ulong(timestamp_high) << 32 | timestamp_low) / 1000000);
    
    // Process network sync events based on type
    switch (event.event_type) {
        case 0: // sync_request
            // Prepare sync response with current timebase state
            break;
        case 1: // sync_response
            // Adjust timebase based on network peer sync
            break;
        case 2: // transport_change
            // Apply transport changes from network peers
            if (event.data[0] != atomic_load_explicit(&timebase.transport_state, memory_order_relaxed)) {
                atomic_store_explicit(&timebase.transport_state, event.data[0], memory_order_relaxed);
            }
            break;
    }
}

// MIDI event dispatch kernel
kernel void midi_event_dispatch(
    device GPUTimebaseState& timebase [[buffer(0)]],
    device MIDIEvent* midi_queue [[buffer(1)]],
    device atomic_uint& midi_queue_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint queue_size = atomic_load_explicit(&midi_queue_size, memory_order_relaxed);
    if (gid >= queue_size) return;
    
    MIDIEvent event = midi_queue[gid];
    uint current_frame = atomic_load_explicit(&timebase.current_frame, memory_order_relaxed);
    
    // Schedule MIDI events at precise frame timestamps
    if (event.timestamp <= current_frame) {
        // Event is ready for dispatch - mark as processed
        midi_queue[gid].timestamp = 0xFFFFFFFF; // Mark as processed
    }
}

// Audio sync kernel for buffer alignment
kernel void audio_sync_process(
    device GPUTimebaseState& timebase [[buffer(0)]],
    device AudioSyncEvent* audio_queue [[buffer(1)]],
    device atomic_uint& audio_queue_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint queue_size = atomic_load_explicit(&audio_queue_size, memory_order_relaxed);
    if (gid >= queue_size) return;
    
    AudioSyncEvent event = audio_queue[gid];
    // Current frame for audio sync timing
    
    // Synchronize audio buffers to GPU timebase
    switch (event.sync_type) {
        case 0: // buffer_ready
            // Audio buffer is aligned with GPU timebase
            break;
        case 1: // xrun
            // Handle audio underrun/overrun with frame adjustment
            break;
        case 2: // format_change
            // Adjust timebase for sample rate changes
            if (event.frame_offset != atomic_load_explicit(&timebase.sample_rate, memory_order_relaxed)) {
                atomic_store_explicit(&timebase.sample_rate, event.frame_offset, memory_order_relaxed);
            }
            break;
    }
}

// Transport control kernel
kernel void transport_control(
    device GPUTimebaseState& timebase [[buffer(0)]],
    constant uint& command [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return; // Only one thread handles transport
    
    switch (command) {
        case 0: // Stop
            atomic_store_explicit(&timebase.transport_state, 0, memory_order_relaxed);
            break;
        case 1: // Play
            atomic_store_explicit(&timebase.transport_state, 1, memory_order_relaxed);
            break;
        case 2: // Pause
            atomic_store_explicit(&timebase.transport_state, 2, memory_order_relaxed);
            break;
        case 3: // Record
            atomic_store_explicit(&timebase.transport_state, 3, memory_order_relaxed);
            break;
        case 4: // Seek
            atomic_store_explicit(&timebase.current_frame, position, memory_order_relaxed);
            break;
        case 5: // Set BPM
            atomic_store_explicit(&timebase.bpm, position, memory_order_relaxed);
            break;
    }
}
