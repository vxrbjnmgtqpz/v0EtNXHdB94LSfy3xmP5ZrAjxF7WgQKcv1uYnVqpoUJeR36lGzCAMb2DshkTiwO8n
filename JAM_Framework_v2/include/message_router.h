#pragma once

/**
 * JAM Framework v2: Universal Message Router
 * 
 * THE STREAM IS THE INTERFACE - Eliminates traditional APIs
 * 
 * This router implements the revolutionary paradigm where JSON messages
 * completely replace framework APIs. Every interaction (MIDI, audio, video,
 * transport, sync) becomes a self-contained JSON message in a universal stream.
 */

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>

namespace jam {

/**
 * Universal Message Handler - replaces all traditional API callbacks
 */
using MessageHandler = std::function<void(const nlohmann::json& message)>;

/**
 * Message Router Statistics
 */
struct RouterStats {
    uint64_t total_messages_processed = 0;
    uint64_t messages_per_second = 0;
    uint64_t avg_processing_time_ns = 0;
    std::unordered_map<std::string, uint64_t> message_type_counts;
    uint64_t routing_errors = 0;
};

/**
 * JAMMessageRouter - The Universal Interface Revolution
 * 
 * CORE PRINCIPLE: The stream IS the interface
 * 
 * Instead of framework APIs like:
 *   jmid->getMidiMessage()
 *   jdat->getAudioBuffer()
 *   transport->setPosition()
 * 
 * Everything becomes JSON messages:
 *   {"type":"jmid_event","midi_data":[144,60,100]}
 *   {"type":"jdat_buffer","samples":[0.1,0.2]}
 *   {"type":"transport_command","action":"play"}
 */
class JAMMessageRouter {
public:
    JAMMessageRouter();
    ~JAMMessageRouter();
    
    /**
     * Initialize the universal message router
     */
    bool initialize();
    
    /**
     * Shutdown and cleanup all handlers
     */
    void shutdown();
    
    /**
     * Subscribe to message types - REPLACES API registration
     * 
     * Example:
     *   router->subscribe("jmid_event", [](const json& msg) {
     *       processMIDI(msg["midi_data"]);
     *   });
     * 
     * @param message_type Type of message to handle
     * @param handler Function to process messages of this type
     */
    void subscribe(const std::string& message_type, MessageHandler handler);
    
    /**
     * Unsubscribe from message type
     */
    void unsubscribe(const std::string& message_type);
    
    /**
     * Process incoming JSON message - REPLACES API calls
     * 
     * Routes message to all registered handlers for the message type.
     * Adds automatic timestamping and performance monitoring.
     * 
     * @param jsonl_message Single line of JSON (JSONL format)
     */
    void processMessage(const std::string& jsonl_message);
    
    /**
     * Send message to the stream - REPLACES API calls
     * 
     * Automatically adds timestamp, device ID, and routing metadata.
     * 
     * @param message JSON message to send
     */
    void sendMessage(const nlohmann::json& message);
    
    /**
     * Send message to specific targets
     * 
     * @param message JSON message to send
     * @param target_types List of message types to route to
     */
    void sendMessageToTargets(const nlohmann::json& message, 
                             const std::vector<std::string>& target_types);
    
    /**
     * Set output handler for sending messages to transport layer
     */
    void setOutputHandler(std::function<void(const std::string&)> handler);
    
    /**
     * Get routing performance statistics
     */
    RouterStats getStats() const;
    
    /**
     * Enable/disable message logging for debugging
     */
    void setLoggingEnabled(bool enabled);
    
    /**
     * Get list of active message types
     */
    std::vector<std::string> getActiveMessageTypes() const;
    
    /**
     * Validate message schema (optional type safety)
     */
    void enableSchemaValidation(bool enabled);
    
private:
    // Message type -> list of handlers
    std::unordered_map<std::string, std::vector<MessageHandler>> handlers_;
    
    // Output handler for sending messages
    std::function<void(const std::string&)> output_handler_;
    
    // Performance monitoring
    mutable std::mutex stats_mutex_;
    RouterStats stats_;
    
    // Configuration
    bool logging_enabled_ = false;
    bool schema_validation_enabled_ = false;
    
    // Thread safety
    mutable std::mutex router_mutex_;
    
    // Device identification
    std::string device_id_;
    
    // Helper methods
    uint64_t getCurrentTimestamp() const;
    void updateStats(const std::string& message_type, uint64_t processing_time_ns);
    bool validateMessageSchema(const nlohmann::json& message);
    void logMessage(const std::string& direction, const nlohmann::json& message);
};

} // namespace jam
