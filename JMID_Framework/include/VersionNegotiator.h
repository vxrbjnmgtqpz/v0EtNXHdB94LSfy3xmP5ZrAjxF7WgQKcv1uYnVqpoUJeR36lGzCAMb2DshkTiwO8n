#pragma once

#include <string>
#include <optional>
#include <tuple>
#include <vector>

namespace JMID {

/**
 * Utility class for handling JSON protocol version negotiation
 * and backward compatibility checking.
 */
class VersionNegotiator {
public:
    struct Version {
        uint16_t major;
        uint16_t minor;
        uint16_t patch;
        
        Version(uint16_t maj = 1, uint16_t min = 0, uint16_t pat = 0) 
            : major(maj), minor(min), patch(pat) {}
        
        std::string toString() const {
            return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
        }
        
        bool operator==(const Version& other) const {
            return major == other.major && minor == other.minor && patch == other.patch;
        }
        
        bool operator<(const Version& other) const {
            if (major != other.major) return major < other.major;
            if (minor != other.minor) return minor < other.minor;
            return patch < other.patch;
        }
        
        bool operator<=(const Version& other) const {
            return *this < other || *this == other;
        }
    };
    
    /**
     * Parse version string in format "major.minor.patch"
     */
    static std::optional<Version> parseVersion(const std::string& versionStr);
    
    /**
     * Check if two versions are compatible (same major, receiver minor >= sender minor)
     */
    static bool areCompatible(const Version& senderVersion, const Version& receiverVersion);
    
    /**
     * Get the current framework version
     */
    static Version getCurrentVersion();
    
    /**
     * Get list of supported version ranges for backward compatibility
     */
    static std::vector<std::pair<Version, Version>> getSupportedVersionRanges();
    
    /**
     * Negotiate the best common version between sender and receiver capabilities
     */
    static std::optional<Version> negotiateVersion(
        const std::vector<Version>& senderSupported,
        const std::vector<Version>& receiverSupported
    );
    
    /**
     * Check if a JSON message string contains a compatible protocol version
     */
    static bool isMessageCompatible(const std::string& jsonMessage);
    
    /**
     * Extract protocol version from a JSON message string
     */
    static std::optional<Version> extractVersionFromMessage(const std::string& jsonMessage);
};

} // namespace JMID
