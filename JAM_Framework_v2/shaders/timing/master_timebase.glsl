#version 450

// GPU-Native Master Timebase Compute Shader (Vulkan/GLSL)

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict buffer GPUTimebaseState {
    uint current_frame;
    uint sample_rate;
    uint bpm;
    uint beat_position;
    uint transport_state; // 0=stop, 1=play, 2=pause, 3=record
    uint quantum_frames;
    uint64_t timestamp_ns;
    uint tempo_scale; // Fixed point: 1000 = 1.0x
    uint metronome_enable;
    uint loop_start;
    uint loop_end;
    uint loop_enable;
} timebase;

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

layout(std430, binding = 1) restrict buffer NetworkQueue {
    NetworkSyncEvent events[];
} network_queue;

layout(std430, binding = 2) restrict buffer MIDIQueue {
    MIDIEvent events[];
} midi_queue;

layout(std430, binding = 3) restrict buffer AudioQueue {
    AudioSyncEvent events[];
} audio_queue;

layout(std430, binding = 4) restrict buffer QueueSizes {
    uint network_size;
    uint midi_size;
    uint audio_size;
    uint padding;
} queue_sizes;

layout(push_constant) uniform PushConstants {
    uint delta_frames;
    uint command;
    uint position;
    uint mode; // 0=tick, 1=network_sync, 2=midi_dispatch, 3=audio_sync, 4=transport
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    if (pc.mode == 0) { // Master timebase tick
        if (gid != 0) return; // Only one thread updates master timebase
        
        if (timebase.transport_state == 1 || timebase.transport_state == 3) { // Playing or Recording
            // Advance the master frame counter atomically
            uint old_frame = atomicAdd(timebase.current_frame, pc.delta_frames);
            uint new_frame = old_frame + pc.delta_frames;
            
            // Update timestamp with nanosecond precision
            uint64_t new_timestamp = (uint64_t(new_frame) * 1000000000UL) / timebase.sample_rate;
            timebase.timestamp_ns = new_timestamp;
            
            // Calculate beat position from BPM
            uint frames_per_beat = (timebase.sample_rate * 60) / timebase.bpm;
            timebase.beat_position = new_frame / frames_per_beat;
            
            // Handle loop boundaries
            if (timebase.loop_enable != 0) {
                if (new_frame >= timebase.loop_end) {
                    timebase.current_frame = timebase.loop_start;
                }
            }
        }
    }
    else if (pc.mode == 1) { // Network sync dispatch
        if (gid >= queue_sizes.network_size) return;
        
        NetworkSyncEvent event = network_queue.events[gid];
        uint current_timestamp = uint(timebase.timestamp_ns / 1000000);
        
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
                if (event.data[0] != timebase.transport_state) {
                    timebase.transport_state = event.data[0];
                }
                break;
        }
    }
    else if (pc.mode == 2) { // MIDI event dispatch
        if (gid >= queue_sizes.midi_size) return;
        
        MIDIEvent event = midi_queue.events[gid];
        
        // Schedule MIDI events at precise frame timestamps
        if (event.timestamp <= timebase.current_frame) {
            // Event is ready for dispatch - mark as processed
            midi_queue.events[gid].timestamp = 0xFFFFFFFF; // Mark as processed
        }
    }
    else if (pc.mode == 3) { // Audio sync process
        if (gid >= queue_sizes.audio_size) return;
        
        AudioSyncEvent event = audio_queue.events[gid];
        
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
                if (event.frame_offset != timebase.sample_rate) {
                    timebase.sample_rate = event.frame_offset;
                }
                break;
        }
    }
    else if (pc.mode == 4) { // Transport control
        if (gid != 0) return; // Only one thread handles transport
        
        switch (pc.command) {
            case 0: // Stop
                timebase.transport_state = 0;
                break;
            case 1: // Play
                timebase.transport_state = 1;
                break;
            case 2: // Pause
                timebase.transport_state = 2;
                break;
            case 3: // Record
                timebase.transport_state = 3;
                break;
            case 4: // Seek
                timebase.current_frame = pc.position;
                break;
            case 5: // Set BPM
                timebase.bpm = pc.position;
                break;
        }
    }
}
