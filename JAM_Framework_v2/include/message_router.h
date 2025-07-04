#pragma once

/**
 * JAM Framework v2: Message Router Header
 * 
 * Routes messages between different subsystems
 */

#include <functional>
#include <string>

namespace jam {

class MessageRouter {
public:
    MessageRouter();
    ~MessageRouter();
    
    bool initialize();
    void shutdown();
    
    // Message routing callbacks
    using MessageCallback = std::function<void(const std::string& message)>;
    void set_message_callback(MessageCallback callback);
    
    // Route message to appropriate handler
    void route_message(const std::string& message);
    
private:
    MessageCallback message_callback_;
};

} // namespace jam
