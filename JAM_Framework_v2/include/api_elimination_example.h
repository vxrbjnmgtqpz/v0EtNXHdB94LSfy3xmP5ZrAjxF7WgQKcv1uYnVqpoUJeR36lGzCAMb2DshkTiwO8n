/**
 * JAM Framework v2: API Elimination Example
 * 
 * Practical demonstration of how the JAMMessageRouter completely
 * eliminates traditional framework APIs in favor of JSON messages.
 */

#pragma once

#include "message_router.h"
#include <nlohmann/json.hpp>
#include <iostream>

namespace jam {

/**
 * MIDI Message Processor - Replaces JMID Framework API
 * 
 * Instead of: jmid->getMidiMessage(), jmid->setQuantization(), etc.
 * Uses: JSON messages routed through JAMMessageRouter
 */
class MIDIMessageProcessor {
public:
    void initialize(std::shared_ptr<JAMMessageRouter> router) {
        router_ = router;
        
        // Subscribe to MIDI events (replaces API registration)
        router_->subscribe("jmid_event", [this](const nlohmann::json& msg) {
            processMIDIEvent(msg);
        });
        
        // Subscribe to transport changes (replaces transport API callbacks)
        router_->subscribe("transport_command", [this](const nlohmann::json& msg) {
            handleTransportChange(msg);
        });
        
        // Subscribe to parameter updates (replaces parameter API)
        router_->subscribe("parameter_update", [this](const nlohmann::json& msg) {
            if (msg.contains("target") && msg["target"] == "jmid") {
                updateParameter(msg);
            }
        });
        
        std::cout << "MIDIMessageProcessor: Initialized with message-based interface (no APIs!)" << std::endl;
    }
    
    // Send MIDI data (replaces API calls like jmid->sendMidiMessage())
    void sendMIDINote(int channel, int note, int velocity) {
        nlohmann::json message = {
            {"type", "jmid_event"},
            {"timestamp_gpu", getCurrentGPUTime()},
            {"event_type", "note_on"},
            {"channel", channel},
            {"note", note},
            {"velocity", velocity},
            {"source", "midi_processor"}
        };
        
        router_->sendMessage(message);
        std::cout << "MIDIMessageProcessor: Sent MIDI note via JSON message (not API call!)" << std::endl;
    }
    
    // Set quantization (replaces API like jmid->setQuantization())
    void setQuantization(const std::string& quantization) {
        nlohmann::json message = {
            {"type", "parameter_update"},
            {"target", "jmid"},
            {"parameter", "quantization"},
            {"value", quantization},
            {"timestamp_gpu", getCurrentGPUTime()}
        };
        
        router_->sendMessage(message);
        std::cout << "MIDIMessageProcessor: Set quantization via JSON message (not API call!)" << std::endl;
    }
    
private:
    std::shared_ptr<JAMMessageRouter> router_;
    
    void processMIDIEvent(const nlohmann::json& msg) {
        // Process MIDI event from JSON (replaces traditional API data structures)
        if (msg["event_type"] == "note_on") {
            int channel = msg["channel"];
            int note = msg["note"];
            int velocity = msg["velocity"];
            
            std::cout << "MIDIMessageProcessor: Processed MIDI note_on from JSON: "
                      << "ch=" << channel << " note=" << note << " vel=" << velocity << std::endl;
            
            // Send processed result as JSON message (not API callback!)
            nlohmann::json result = {
                {"type", "jmid_processed"},
                {"original_timestamp", msg["timestamp_gpu"]},
                {"processing_time_ns", 15000},
                {"processed_note", {
                    {"channel", channel},
                    {"note", note},
                    {"velocity", velocity},
                    {"quantized", true}
                }}
            };
            
            router_->sendMessage(result);
        }
    }
    
    void handleTransportChange(const nlohmann::json& msg) {
        // Handle transport changes via JSON (replaces transport API callbacks)
        if (msg["action"] == "play") {
            std::cout << "MIDIMessageProcessor: Transport started (from JSON message, not API callback!)" << std::endl;
            
            // Send transport acknowledgment
            nlohmann::json response = {
                {"type", "transport_ack"},
                {"source", "midi_processor"},
                {"action", "play"},
                {"status", "ready"}
            };
            router_->sendMessage(response);
        }
    }
    
    void updateParameter(const nlohmann::json& msg) {
        std::string param = msg["parameter"];
        std::cout << "MIDIMessageProcessor: Updated parameter '" << param 
                  << "' via JSON message (not API call!)" << std::endl;
    }
    
    uint64_t getCurrentGPUTime() const {
        // Mock GPU timestamp
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
};

/**
 * Audio Message Processor - Replaces JDAT Framework API
 */
class AudioMessageProcessor {
public:
    void initialize(std::shared_ptr<JAMMessageRouter> router) {
        router_ = router;
        
        // Subscribe to audio buffer events (replaces JDAT API)
        router_->subscribe("jdat_buffer", [this](const nlohmann::json& msg) {
            processAudioBuffer(msg);
        });
        
        // Subscribe to audio parameter changes
        router_->subscribe("parameter_update", [this](const nlohmann::json& msg) {
            if (msg.contains("target") && msg["target"] == "jdat") {
                updateAudioParameter(msg);
            }
        });
        
        std::cout << "AudioMessageProcessor: Initialized with message-based interface (no JDAT APIs!)" << std::endl;
    }
    
    // Send audio buffer (replaces API like jdat->sendAudioBuffer())
    void sendAudioBuffer(const std::vector<float>& samples, int sample_rate) {
        nlohmann::json message = {
            {"type", "jdat_buffer"},
            {"timestamp_gpu", getCurrentGPUTime()},
            {"samples", samples},
            {"sample_rate", sample_rate},
            {"channels", 2},
            {"buffer_size", samples.size()},
            {"source", "audio_processor"}
        };
        
        router_->sendMessage(message);
        std::cout << "AudioMessageProcessor: Sent audio buffer via JSON message (not JDAT API!)" << std::endl;
    }
    
    // Set volume (replaces API like jdat->setVolume())
    void setVolume(float volume) {
        nlohmann::json message = {
            {"type", "parameter_update"},
            {"target", "jdat"},
            {"parameter", "volume"},
            {"value", volume},
            {"ramp_time_ms", 50}
        };
        
        router_->sendMessage(message);
        std::cout << "AudioMessageProcessor: Set volume via JSON message (not JDAT API!)" << std::endl;
    }
    
private:
    std::shared_ptr<JAMMessageRouter> router_;
    
    void processAudioBuffer(const nlohmann::json& msg) {
        auto samples = msg["samples"].get<std::vector<float>>();
        int sample_rate = msg["sample_rate"];
        
        std::cout << "AudioMessageProcessor: Processed audio buffer from JSON: "
                  << samples.size() << " samples at " << sample_rate << "Hz" << std::endl;
        
        // Send processed result
        nlohmann::json result = {
            {"type", "jdat_processed"},
            {"original_timestamp", msg["timestamp_gpu"]},
            {"processed_samples", samples.size()},
            {"peak_level", 0.85},
            {"rms_level", 0.42}
        };
        
        router_->sendMessage(result);
    }
    
    void updateAudioParameter(const nlohmann::json& msg) {
        std::string param = msg["parameter"];
        std::cout << "AudioMessageProcessor: Updated audio parameter '" << param 
                  << "' via JSON message (not JDAT API!)" << std::endl;
    }
    
    uint64_t getCurrentGPUTime() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
};

/**
 * Transport Controller - Replaces Transport API
 */
class TransportController {
public:
    void initialize(std::shared_ptr<JAMMessageRouter> router) {
        router_ = router;
        
        // Subscribe to transport commands
        router_->subscribe("transport_command", [this](const nlohmann::json& msg) {
            executeTransportCommand(msg);
        });
        
        std::cout << "TransportController: Initialized with message-based interface (no Transport APIs!)" << std::endl;
    }
    
    // Play transport (replaces API like transport->play())
    void play() {
        nlohmann::json message = {
            {"type", "transport_command"},
            {"action", "play"},
            {"timestamp_gpu", getCurrentGPUTime()},
            {"position_samples", current_position_},
            {"bpm", current_bpm_}
        };
        
        router_->sendMessage(message);
        std::cout << "TransportController: Sent play command via JSON message (not Transport API!)" << std::endl;
    }
    
    // Set position (replaces API like transport->setPosition())
    void setPosition(uint64_t position_samples) {
        current_position_ = position_samples;
        
        nlohmann::json message = {
            {"type", "transport_command"},
            {"action", "set_position"},
            {"position_samples", position_samples},
            {"timestamp_gpu", getCurrentGPUTime()}
        };
        
        router_->sendMessage(message);
        std::cout << "TransportController: Set position via JSON message (not Transport API!)" << std::endl;
    }
    
private:
    std::shared_ptr<JAMMessageRouter> router_;
    uint64_t current_position_ = 0;
    double current_bpm_ = 120.0;
    
    void executeTransportCommand(const nlohmann::json& msg) {
        std::string action = msg["action"];
        
        if (action == "play") {
            std::cout << "TransportController: Executing play command from JSON message" << std::endl;
        } else if (action == "set_position") {
            uint64_t position = msg["position_samples"];
            current_position_ = position;
            std::cout << "TransportController: Set position to " << position << " samples" << std::endl;
        }
        
        // Send transport state update
        nlohmann::json state = {
            {"type", "transport_state"},
            {"state", action == "play" ? "playing" : "stopped"},
            {"position_samples", current_position_},
            {"bpm", current_bpm_},
            {"timestamp_gpu", getCurrentGPUTime()}
        };
        
        router_->sendMessage(state);
    }
    
    uint64_t getCurrentGPUTime() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
};

/**
 * API Elimination Demo - Shows the complete transformation
 */
class APIEliminationDemo {
public:
    void runDemo() {
        std::cout << "\n=== JAMNet API Elimination Revolution Demo ===" << std::endl;
        std::cout << "Demonstrating how JSON messages completely replace traditional APIs\n" << std::endl;
        
        // Create the universal message router
        auto router = std::make_shared<JAMMessageRouter>();
        router->initialize();
        router->setLoggingEnabled(true);
        
        // Set up output handler (normally would connect to UDP transport)
        router->setOutputHandler([](const std::string& message) {
            std::cout << "TRANSPORT: " << message << std::endl;
        });
        
        // Initialize processors (no APIs - only message handlers!)
        MIDIMessageProcessor midi_processor;
        AudioMessageProcessor audio_processor;
        TransportController transport;
        
        midi_processor.initialize(router);
        audio_processor.initialize(router);
        transport.initialize(router);
        
        std::cout << "\n--- Traditional API vs JSON Message Comparison ---" << std::endl;
        
        // Traditional API approach (ELIMINATED):
        std::cout << "\n❌ ELIMINATED: Traditional API calls like:" << std::endl;
        std::cout << "   jmid->sendMidiMessage(channel, note, velocity)" << std::endl;
        std::cout << "   jdat->sendAudioBuffer(samples, sample_rate)" << std::endl;
        std::cout << "   transport->play()" << std::endl;
        std::cout << "   transport->setPosition(samples)" << std::endl;
        
        // New JSON message approach:
        std::cout << "\n✅ REPLACED WITH: JSON message-based communication:" << std::endl;
        
        std::cout << "\n1. MIDI Communication (replaces JMID APIs):" << std::endl;
        midi_processor.sendMIDINote(1, 60, 100);
        midi_processor.setQuantization("quarter_note");
        
        std::cout << "\n2. Audio Communication (replaces JDAT APIs):" << std::endl;
        std::vector<float> audio_samples = {0.1f, 0.2f, -0.1f, 0.3f};
        audio_processor.sendAudioBuffer(audio_samples, 48000);
        audio_processor.setVolume(0.8f);
        
        std::cout << "\n3. Transport Communication (replaces Transport APIs):" << std::endl;
        transport.setPosition(44100);
        transport.play();
        
        std::cout << "\n--- Performance Statistics ---" << std::endl;
        auto stats = router->getStats();
        std::cout << "Total messages processed: " << stats.total_messages_processed << std::endl;
        std::cout << "Average processing time: " << stats.avg_processing_time_ns << "ns" << std::endl;
        std::cout << "Routing errors: " << stats.routing_errors << std::endl;
        
        std::cout << "\nActive message types (replacing APIs):" << std::endl;
        for (const auto& type : router->getActiveMessageTypes()) {
            std::cout << "  - " << type << std::endl;
        }
        
        std::cout << "\n=== Revolution Complete: APIs Eliminated! ===" << std::endl;
        std::cout << "✅ No more framework APIs" << std::endl;
        std::cout << "✅ Universal JSON message routing" << std::endl;
        std::cout << "✅ Self-contained, stateless communication" << std::endl;
        std::cout << "✅ Platform-agnostic protocol" << std::endl;
        std::cout << "✅ Perfect debugging and replay capability" << std::endl;
        
        router->shutdown();
    }
};

} // namespace jam
