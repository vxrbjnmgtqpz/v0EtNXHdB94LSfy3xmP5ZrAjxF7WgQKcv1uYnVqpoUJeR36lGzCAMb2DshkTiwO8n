#include "VersionNegotiator.h"
#include "JSONMIDIMessage.h"
#include <sstream>
#include <regex>
#include <algorithm>

namespace JSONMIDI {

std::optional<VersionNegotiator::Version> VersionNegotiator::parseVersion(const std::string& versionStr) {
    std::regex versionRegex(R"(^(\d+)\.(\d+)\.(\d+)$)");
    std::smatch matches;
    
    if (std::regex_match(versionStr, matches, versionRegex)) {
        try {
            uint16_t major = static_cast<uint16_t>(std::stoi(matches[1].str()));
            uint16_t minor = static_cast<uint16_t>(std::stoi(matches[2].str()));
            uint16_t patch = static_cast<uint16_t>(std::stoi(matches[3].str()));
            return Version(major, minor, patch);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    
    return std::nullopt;
}

bool VersionNegotiator::areCompatible(const Version& senderVersion, const Version& receiverVersion) {
    // Major versions must match exactly
    if (senderVersion.major != receiverVersion.major) {
        return false;
    }
    
    // Receiver must support sender's minor version or higher (backward compatibility)
    return receiverVersion.minor >= senderVersion.minor;
}

VersionNegotiator::Version VersionNegotiator::getCurrentVersion() {
    return Version(JSONProtocolVersion::MAJOR, JSONProtocolVersion::MINOR, JSONProtocolVersion::PATCH);
}

std::vector<std::pair<VersionNegotiator::Version, VersionNegotiator::Version>> 
VersionNegotiator::getSupportedVersionRanges() {
    // For now, we only support current version
    // In the future, add backward compatibility ranges here
    Version current = getCurrentVersion();
    return {{current, current}};
}

std::optional<VersionNegotiator::Version> VersionNegotiator::negotiateVersion(
    const std::vector<Version>& senderSupported,
    const std::vector<Version>& receiverSupported
) {
    // Find the highest compatible version both sides support
    std::optional<Version> bestVersion;
    
    for (const auto& senderVer : senderSupported) {
        for (const auto& receiverVer : receiverSupported) {
            if (areCompatible(senderVer, receiverVer)) {
                // Choose the higher version for better features
                Version useVersion = (senderVer <= receiverVer) ? receiverVer : senderVer;
                
                if (!bestVersion || bestVersion < useVersion) {
                    bestVersion = useVersion;
                }
            }
        }
    }
    
    return bestVersion;
}

bool VersionNegotiator::isMessageCompatible(const std::string& jsonMessage) {
    auto version = extractVersionFromMessage(jsonMessage);
    if (!version) {
        // If no version specified, assume compatibility for backward compatibility
        return true;
    }
    
    return areCompatible(*version, getCurrentVersion());
}

std::optional<VersionNegotiator::Version> VersionNegotiator::extractVersionFromMessage(const std::string& jsonMessage) {
    // Simple regex to extract protocolVersion field
    std::regex versionRegex("\"protocolVersion\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch matches;
    
    if (std::regex_search(jsonMessage, matches, versionRegex)) {
        return parseVersion(matches[1].str());
    }
    
    return std::nullopt;
}

} // namespace JSONMIDI
