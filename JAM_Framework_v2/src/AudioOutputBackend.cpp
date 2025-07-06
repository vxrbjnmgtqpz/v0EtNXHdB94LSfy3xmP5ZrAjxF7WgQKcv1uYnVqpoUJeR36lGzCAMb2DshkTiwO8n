#include "AudioOutputBackend.h"

#ifdef JAM_ENABLE_JACK
#include "JackAudioBackend.h"
#endif

#ifdef __APPLE__
// #include "CoreAudioBackend.h" // To be implemented
#endif

#include <iostream>

namespace JAMNet {

std::unique_ptr<AudioOutputBackend> AudioOutputBackend::create(BackendType preferredType) {
    // Try preferred type first
    switch (preferredType) {
        case BackendType::JACK:
#ifdef JAM_ENABLE_JACK
            if (JackAudioBackend::isJackRunning()) {
                auto backend = std::make_unique<JackAudioBackend>();
                std::cout << "AudioOutputBackend: Created JACK backend" << std::endl;
                return std::move(backend);
            } else {
                std::cout << "AudioOutputBackend: JACK not running, trying alternatives" << std::endl;
            }
#endif
            break;
            
        case BackendType::CORE_AUDIO:
#ifdef __APPLE__
            // Core Audio backend implementation would go here
            std::cout << "AudioOutputBackend: Core Audio backend not yet implemented" << std::endl;
#endif
            break;
            
        default:
            break;
    }
    
    // Fallback to any available backend
    std::vector<BackendType> available = getAvailableBackends();
    for (BackendType type : available) {
        if (type == preferredType) {
            continue; // Already tried
        }
        
        switch (type) {
            case BackendType::JACK:
#ifdef JAM_ENABLE_JACK
                if (JackAudioBackend::isJackRunning()) {
                    auto backend = std::make_unique<JackAudioBackend>();
                    std::cout << "AudioOutputBackend: Fallback to JACK backend" << std::endl;
                    return std::move(backend);
                }
#endif
                break;
                
            case BackendType::CORE_AUDIO:
#ifdef __APPLE__
                // Core Audio fallback would go here
#endif
                break;
                
            default:
                break;
        }
    }
    
    std::cerr << "AudioOutputBackend: No suitable backend available" << std::endl;
    return nullptr;
}

std::vector<AudioOutputBackend::BackendType> AudioOutputBackend::getAvailableBackends() {
    std::vector<BackendType> available;
    
#ifdef JAM_ENABLE_JACK
    if (JackAudioBackend::isJackRunning()) {
        available.push_back(BackendType::JACK);
    }
#endif

#ifdef __APPLE__
    // Core Audio is always available on macOS
    available.push_back(BackendType::CORE_AUDIO);
#endif

#ifdef __linux__
    // ALSA could be added here as a fallback
    available.push_back(BackendType::ALSA);
#endif

    return available;
}

} // namespace JAMNet
