#include "GPURenderEngine.h"

#ifdef __linux__

#include <iostream>

namespace JAMNet {

// Stub factory function for Vulkan render engine (to be implemented)
std::unique_ptr<GPURenderEngine> createVulkanRenderEngine() {
    std::cerr << "VulkanRenderEngine: Not yet implemented" << std::endl;
    return nullptr;
}

} // namespace JAMNet

#endif // __linux__
