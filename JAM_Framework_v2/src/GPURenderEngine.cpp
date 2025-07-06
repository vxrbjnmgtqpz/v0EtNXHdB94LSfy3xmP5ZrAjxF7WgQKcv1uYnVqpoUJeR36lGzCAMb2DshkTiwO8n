#include "GPURenderEngine.h"
#include <iostream>

namespace JAMNet {

// Forward declaration of platform-specific factory functions
#ifdef __APPLE__
extern std::unique_ptr<GPURenderEngine> createMetalRenderEngine();
#endif

#ifdef __linux__
extern std::unique_ptr<GPURenderEngine> createVulkanRenderEngine();
#endif

std::unique_ptr<GPURenderEngine> GPURenderEngine::create() {
#ifdef __APPLE__
    // On macOS, prefer Metal
    auto engine = createMetalRenderEngine();
    if (engine) {
        std::cout << "GPURenderEngine: Created Metal render engine" << std::endl;
    }
    return engine;
#elif defined(__linux__)
    // On Linux, use Vulkan (to be implemented)
    auto engine = createVulkanRenderEngine();
    if (engine) {
        std::cout << "GPURenderEngine: Created Vulkan render engine" << std::endl;
    } else {
        std::cerr << "GPURenderEngine: Vulkan engine not yet implemented" << std::endl;
    }
    return engine;
#else
    std::cerr << "GPURenderEngine: No GPU backend available for this platform" << std::endl;
    return nullptr;
#endif
}

} // namespace JAMNet
