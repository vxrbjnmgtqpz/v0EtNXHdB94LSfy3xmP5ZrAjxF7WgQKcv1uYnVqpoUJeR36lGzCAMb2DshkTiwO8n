/**
 * JAM Framework v2: Universal Message Router Implementation
 * 
 * THE STREAM IS THE INTERFACE - Revolutionary API Elimination
 */

#include "message_router.h"
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>

namespace jam {

JAMMessageRouter::JAMMessageRouter() {
    // Generate unique device ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::stringstream ss;
    ss << "jam_device_";
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << dis(gen);
    }
    device_id_ = ss.str();
    
    std::cout << "JAMMessageRouter: Initialized with device_id=" << device_id_ << std::endl;
}

JAMMessageRouter::~JAMMessageRouter() {
    shutdown();
}

bool JAMMessageRouter::initialize() {
    std::lock_guard<std::mutex> lock(router_mutex_);
    
    // Initialize core message types that replace traditional APIs
    std::cout << "JAMMessageRouter: Initializing universal message routing..." << std::endl;
    std::cout << "  - Replacing JMID APIs with 'jmid_*' messages" << std::endl;
    std::cout << "  - Replacing JDAT APIs with 'jdat_*' messages" << std::endl;
    std::cout << "  - Replacing JVID APIs with 'jvid_*' messages" << std::endl;
    std::cout << "  - Replacing Transport APIs with 'transport_*' messages" << std::endl;
    std::cout << "  - Replacing Sync APIs with 'sync_*' messages" << std::endl;
    
    return true;
}

void JAMMessageRouter::shutdown() {
    std::lock_guard<std::mutex> lock(router_mutex_);
    handlers_.clear();
    output_handler_ = nullptr;
    std::cout << "JAMMessageRouter: Shutdown complete" << std::endl;
}

void JAMMessageRouter::subscribe(const std::string& message_type, MessageHandler handler) {
    std::lock_guard<std::mutex> lock(router_mutex_);
    handlers_[message_type].push_back(handler);
    
    if (logging_enabled_) {
        std::cout << "JAMMessageRouter: Subscribed to message type '" << message_type 
                  << "' (replaces traditional API registration)" << std::endl;
    }
}

void JAMMessageRouter::unsubscribe(const std::string& message_type) {
    std::lock_guard<std::mutex> lock(router_mutex_);
    handlers_.erase(message_type);
    
    if (logging_enabled_) {
        std::cout << "JAMMessageRouter: Unsubscribed from message type '" << message_type << "'" << std::endl;
    }
}

void JAMMessageRouter::processMessage(const std::string& jsonl_message) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Parse JSON message
        auto message = nlohmann::json::parse(jsonl_message);
        
        // Validate required fields
        if (!message.contains("type")) {
            std::cerr << "JAMMessageRouter: Message missing 'type' field: " << jsonl_message << std::endl;
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.routing_errors++;
            }
            return;
        }
        
        auto message_type = message["type"].get<std::string>();
        
        // Add routing metadata if not present
        if (!message.contains("timestamp_router")) {
            message["timestamp_router"] = getCurrentTimestamp();
        }
        if (!message.contains("router_device_id")) {
            message["router_device_id"] = device_id_;
        }
        
        // Optional schema validation
        if (schema_validation_enabled_ && !validateMessageSchema(message)) {
            std::cerr << "JAMMessageRouter: Schema validation failed for type '" << message_type << "'" << std::endl;
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.routing_errors++;
            }
            return;
        }
        
        // Route to handlers (REPLACES traditional API calls)
        {
            std::lock_guard<std::mutex> lock(router_mutex_);
            auto it = handlers_.find(message_type);
            if (it != handlers_.end()) {
                for (const auto& handler : it->second) {
                    try {
                        handler(message);
                    } catch (const std::exception& e) {
                        std::cerr << "JAMMessageRouter: Handler error for type '" << message_type 
                                  << "': " << e.what() << std::endl;
                        {
                            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                            stats_.routing_errors++;
                        }
                    }
                }
                
                if (logging_enabled_) {
                    logMessage("PROCESSED", message);
                }
            } else {
                if (logging_enabled_) {
                    std::cout << "JAMMessageRouter: No handlers for message type '" << message_type << "'" << std::endl;
                }
            }
        }
        
        // Update performance statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        updateStats(message_type, processing_time.count());
        
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JAMMessageRouter: JSON parse error: " << e.what() << std::endl;
        std::cerr << "  Message: " << jsonl_message << std::endl;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.routing_errors++;
        }
    } catch (const std::exception& e) {
        std::cerr << "JAMMessageRouter: Unexpected error: " << e.what() << std::endl;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.routing_errors++;
        }
    }
}

void JAMMessageRouter::sendMessage(const nlohmann::json& message) {
    try {
        // Create enhanced message with routing metadata
        auto enhanced_message = message;
        
        // Add automatic timestamping and device identification
        enhanced_message["timestamp_router"] = getCurrentTimestamp();
        enhanced_message["sender_device_id"] = device_id_;
        
        // Add version information
        if (!enhanced_message.contains("version")) {
            enhanced_message["version"] = "jamnet/1.0";
        }
        
        // Validate message has required fields
        if (!enhanced_message.contains("type")) {
            std::cerr << "JAMMessageRouter: Cannot send message without 'type' field" << std::endl;
            return;
        }
        
        // Convert to JSONL format and send
        std::string jsonl_message = enhanced_message.dump();
        
        if (output_handler_) {
            output_handler_(jsonl_message);
            
            if (logging_enabled_) {
                logMessage("SENT", enhanced_message);
            }
        } else {
            std::cerr << "JAMMessageRouter: No output handler configured for sending messages" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "JAMMessageRouter: Error sending message: " << e.what() << std::endl;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.routing_errors++;
        }
    }
}

void JAMMessageRouter::sendMessageToTargets(const nlohmann::json& message, 
                                           const std::vector<std::string>& target_types) {
    auto enhanced_message = message;
    enhanced_message["target_types"] = target_types;
    sendMessage(enhanced_message);
}

void JAMMessageRouter::setOutputHandler(std::function<void(const std::string&)> handler) {
    output_handler_ = handler;
    std::cout << "JAMMessageRouter: Output handler configured for message transport" << std::endl;
}

RouterStats JAMMessageRouter::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void JAMMessageRouter::setLoggingEnabled(bool enabled) {
    logging_enabled_ = enabled;
    std::cout << "JAMMessageRouter: Logging " << (enabled ? "enabled" : "disabled") << std::endl;
}

std::vector<std::string> JAMMessageRouter::getActiveMessageTypes() const {
    std::lock_guard<std::mutex> lock(router_mutex_);
    std::vector<std::string> types;
    for (const auto& pair : handlers_) {
        types.push_back(pair.first);
    }
    return types;
}

void JAMMessageRouter::enableSchemaValidation(bool enabled) {
    schema_validation_enabled_ = enabled;
    std::cout << "JAMMessageRouter: Schema validation " << (enabled ? "enabled" : "disabled") << std::endl;
}

// Private helper methods

uint64_t JAMMessageRouter::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void JAMMessageRouter::updateStats(const std::string& message_type, uint64_t processing_time_ns) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_messages_processed++;
    stats_.message_type_counts[message_type]++;
    
    // Update average processing time (simple moving average)
    if (stats_.avg_processing_time_ns == 0) {
        stats_.avg_processing_time_ns = processing_time_ns;
    } else {
        stats_.avg_processing_time_ns = (stats_.avg_processing_time_ns * 0.9) + (processing_time_ns * 0.1);
    }
    
    // Calculate messages per second (approximate)
    static auto last_calculation = std::chrono::steady_clock::now();
    static uint64_t last_message_count = 0;
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_calculation);
    
    if (duration.count() >= 1) {
        stats_.messages_per_second = stats_.total_messages_processed - last_message_count;
        last_message_count = stats_.total_messages_processed;
        last_calculation = now;
    }
}

bool JAMMessageRouter::validateMessageSchema(const nlohmann::json& message) {
    // Basic schema validation
    if (!message.contains("type") || !message["type"].is_string()) {
        return false;
    }
    
    // Type-specific validation could be added here
    std::string type = message["type"];
    
    if (type.starts_with("jmid_")) {
        // MIDI message validation
        return message.contains("timestamp_gpu");
    } else if (type.starts_with("jdat_")) {
        // Audio message validation
        return message.contains("timestamp_gpu") && message.contains("sample_rate");
    } else if (type.starts_with("jvid_")) {
        // Video message validation
        return message.contains("timestamp_gpu") && message.contains("frame_data");
    } else if (type.starts_with("transport_")) {
        // Transport message validation
        return message.contains("action") || message.contains("state");
    } else if (type.starts_with("sync_")) {
        // Sync message validation
        return message.contains("timestamp_gpu");
    }
    
    return true; // Unknown types pass validation by default
}

void JAMMessageRouter::logMessage(const std::string& direction, const nlohmann::json& message) {
    if (!logging_enabled_) return;
    
    std::cout << "JAMMessageRouter [" << direction << "]: " 
              << message["type"].get<std::string>();
    
    if (message.contains("timestamp_gpu")) {
        std::cout << " (gpu_ts=" << message["timestamp_gpu"] << ")";
    }
    
    if (message.contains("sender_device_id")) {
        std::cout << " from=" << message["sender_device_id"].get<std::string>();
    }
    
    std::cout << std::endl;
}

} // namespace jam
