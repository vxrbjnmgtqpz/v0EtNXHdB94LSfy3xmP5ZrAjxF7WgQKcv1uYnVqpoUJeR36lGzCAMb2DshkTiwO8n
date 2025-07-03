// SchemaValidator implementation - Phase 1.2: Advanced real-time validation
#include "JMIDParser.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <unordered_map>
// Note: Full JSON schema validation will be implemented when the library is properly integrated
// #include <nlohmann/json-schema.hpp>

namespace JMID {

/**
 * Private implementation for SchemaValidator
 * Implements fast schema validation with caching and error recovery
 */
class SchemaValidator::Impl {
public:
    Impl() : 
        validationEnabled_(true),
        cacheEnabled_(true),
        errorRecoveryEnabled_(true) {
        
        // Pre-compile common validation patterns
        initializeValidationPatterns();
    }
    
    bool loadSchema(const std::string& schemaPath) {
        std::ifstream file(schemaPath);
        if (!file.is_open()) {
            lastError_ = "Failed to open schema file: " + schemaPath;
            return false;
        }
        
        std::string schemaContent(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        
        return loadSchemaFromString(schemaContent);
    }
    
    bool loadSchemaFromString(const std::string& schemaJson) {
        try {
            // For now, use a simplified validation approach
            // In a full implementation, this would use nlohmann::json_schema::json_validator
            cachedSchemaJson_ = nlohmann::json::parse(schemaJson);
            schemaLoaded_ = true;
            
            // Pre-analyze schema for optimization hints
            analyzeSchemaForOptimization(cachedSchemaJson_);
            
            return true;
            
        } catch (const std::exception& e) {
            lastError_ = "Failed to load schema: " + std::string(e.what());
            return false;
        }
    }
    
    ValidationResult validate(const std::string& json) const {
        if (!schemaLoaded_) {
            return ValidationResult(false, "No schema loaded", "");
        }
        
        // Check cache first for performance
        if (cacheEnabled_) {
            auto cached = validationCache_.find(json);
            if (cached != validationCache_.end()) {
                return cached->second;
            }
        }
        
        try {
            // Parse JSON for validation
            nlohmann::json jsonDoc = nlohmann::json::parse(json);
            
            // Perform fast pre-validation checks
            auto preValidation = performPreValidation(jsonDoc);
            if (!preValidation.isValid) {
                return cacheAndReturn(json, preValidation);
            }
            
            // Full schema validation (simplified for Phase 1.2)
            ValidationResult result = performFullValidation(jsonDoc);
            
            return cacheAndReturn(json, result);
            
        } catch (const std::exception& e) {
            ValidationResult result(false, "JSON parsing error: " + std::string(e.what()), "");
            return cacheAndReturn(json, result);
        }
    }
    
    ValidationResult validateMessage(const MIDIMessage& message) const {
        // Convert message to JSON and validate
        std::string json = message.toJSON();
        return validate(json);
    }
    
    void setValidationEnabled(bool enabled) {
        validationEnabled_ = enabled;
    }
    
    void setCacheEnabled(bool enabled) {
        cacheEnabled_ = enabled;
        if (!enabled) {
            validationCache_.clear();
        }
    }
    
    void setErrorRecoveryEnabled(bool enabled) {
        errorRecoveryEnabled_ = enabled;
    }
    
    void clearCache() {
        validationCache_.clear();
    }
    
    size_t getCacheSize() const {
        return validationCache_.size();
    }
    
    std::string getLastError() const {
        return lastError_;
    }

private:
    nlohmann::json cachedSchemaJson_;
    mutable std::unordered_map<std::string, ValidationResult> validationCache_;
    
    bool schemaLoaded_ = false;
    bool validationEnabled_;
    bool cacheEnabled_;
    bool errorRecoveryEnabled_;
    mutable std::string lastError_;
    
    // Pre-compiled validation patterns for fast checks
    std::regex midiTypePattern_;
    std::regex timestampPattern_;
    std::unordered_map<std::string, std::vector<std::string>> requiredFields_;
    
    void initializeValidationPatterns() {
        // Pre-compile common MIDI message validation patterns
        midiTypePattern_ = std::regex(R"(^(note_on|note_off|control_change|system_exclusive|program_change|pitch_bend|aftertouch)$)");
        timestampPattern_ = std::regex(R"(^\d+$)");
        
        // Define required fields for each message type
        requiredFields_["note_on"] = {"type", "timestamp", "channel", "note", "velocity"};
        requiredFields_["note_off"] = {"type", "timestamp", "channel", "note", "velocity"};
        requiredFields_["control_change"] = {"type", "timestamp", "channel", "controller", "value"};
        requiredFields_["system_exclusive"] = {"type", "timestamp", "data"};
        requiredFields_["program_change"] = {"type", "timestamp", "channel", "program"};
        requiredFields_["pitch_bend"] = {"type", "timestamp", "channel", "value"};
        requiredFields_["aftertouch"] = {"type", "timestamp", "channel", "pressure"};
    }
    
    void analyzeSchemaForOptimization(const nlohmann::json& schema) {
        // Analyze schema structure for validation shortcuts
        if (schema.contains("properties")) {
            auto properties = schema["properties"];
            
            // Count validation complexity
            size_t complexityScore = properties.size();
            
            // Adjust cache strategy based on complexity
            if (complexityScore > 10) {
                validationCache_.reserve(1000);
            } else {
                validationCache_.reserve(100);
            }
        }
    }
    
    ValidationResult performPreValidation(const nlohmann::json& json) const {
        // Fast validation checks before full schema validation
        
        // Check required top-level fields
        if (!json.contains("type")) {
            return ValidationResult(false, "Missing required field 'type'", "/type");
        }
        
        if (!json.contains("timestamp")) {
            return ValidationResult(false, "Missing required field 'timestamp'", "/timestamp");
        }
        
        // Validate message type
        std::string type = json["type"];
        if (!std::regex_match(type, midiTypePattern_)) {
            return ValidationResult(false, "Invalid message type: " + type, "/type");
        }
        
        // Check type-specific required fields
        auto requiredIter = requiredFields_.find(type);
        if (requiredIter != requiredFields_.end()) {
            for (const auto& field : requiredIter->second) {
                if (!json.contains(field)) {
                    return ValidationResult(false, 
                        "Missing required field '" + field + "' for type '" + type + "'", 
                        "/" + field);
                }
            }
        }
        
        // Validate MIDI value ranges for performance
        if (type == "note_on" || type == "note_off") {
            if (json.contains("channel")) {
                int channel = json["channel"];
                if (channel < 0 || channel > 15) {
                    return ValidationResult(false, "Channel out of range (0-15): " + std::to_string(channel), "/channel");
                }
            }
            
            if (json.contains("note")) {
                int note = json["note"];
                if (note < 0 || note > 127) {
                    return ValidationResult(false, "Note out of range (0-127): " + std::to_string(note), "/note");
                }
            }
            
            if (json.contains("velocity")) {
                int velocity = json["velocity"];
                if (velocity < 0 || velocity > 127) {
                    return ValidationResult(false, "Velocity out of range (0-127): " + std::to_string(velocity), "/velocity");
                }
            }
        }
        
        return ValidationResult(true); // Pre-validation passed
    }
    
    ValidationResult performFullValidation(const nlohmann::json& json) const {
        // Simplified full validation for Phase 1.2
        // In production, this would use a proper JSON schema validator
        
        try {
            // Basic structural validation
            std::string type = json["type"];
            
            // Validate based on message type
            if (type == "note_on" || type == "note_off") {
                if (!json.contains("channel") || !json.contains("note") || !json.contains("velocity")) {
                    return ValidationResult(false, "Missing required fields for " + type, "");
                }
            } else if (type == "control_change") {
                if (!json.contains("channel") || !json.contains("controller") || !json.contains("value")) {
                    return ValidationResult(false, "Missing required fields for control_change", "");
                }
            } else if (type == "system_exclusive") {
                if (!json.contains("data") || !json["data"].is_array()) {
                    return ValidationResult(false, "Missing or invalid data field for system_exclusive", "");
                }
            }
            
            return ValidationResult(true);
            
        } catch (const std::exception& e) {
            return ValidationResult(false, "Validation error: " + std::string(e.what()), "");
        }
    }
    
    ValidationResult cacheAndReturn(const std::string& json, const ValidationResult& result) const {
        if (cacheEnabled_ && validationCache_.size() < 10000) { // Limit cache size
            validationCache_[json] = result;
        }
        return result;
    }
};

// SchemaValidator public interface implementation
SchemaValidator::SchemaValidator() : impl_(std::make_unique<Impl>()) {}

SchemaValidator::~SchemaValidator() = default;

bool SchemaValidator::loadSchema(const std::string& schemaPath) {
    return impl_->loadSchema(schemaPath);
}

bool SchemaValidator::loadSchemaFromString(const std::string& schemaJson) {
    return impl_->loadSchemaFromString(schemaJson);
}

ValidationResult SchemaValidator::validate(const std::string& json) const {
    return impl_->validate(json);
}

ValidationResult SchemaValidator::validateMessage(const MIDIMessage& message) const {
    return impl_->validateMessage(message);
}

} // namespace JMID
