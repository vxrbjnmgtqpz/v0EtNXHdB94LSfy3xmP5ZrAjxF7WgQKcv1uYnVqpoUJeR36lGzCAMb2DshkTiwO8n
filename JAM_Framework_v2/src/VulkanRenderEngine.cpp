#include "VulkanRenderEngine.h"

#ifdef __linux__

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <string>

namespace JAMNet {

VulkanRenderEngine::VulkanRenderEngine() 
    : vulkan_{}, buffers_{}, pipelines_{}, initialized_(false),
      lastGPUTimestamp_(0), calibrationOffset_(0.0f) {
    syncBlock_ = {0, 0, 0.0f, false};
}

VulkanRenderEngine::~VulkanRenderEngine() {
    shutdown();
}

bool VulkanRenderEngine::initialize(const RenderConfig& config) {
    if (initialized_) {
        return true;
    }
    
    config_ = config;
    
    std::cout << "VulkanRenderEngine: Initializing..." << std::endl;
    
    if (!setupVulkan()) {
        std::cerr << "VulkanRenderEngine: Failed to setup Vulkan" << std::endl;
        return false;
    }
    
    if (!queryTimestampSupport()) {
        std::cerr << "VulkanRenderEngine: Warning - limited timestamp support" << std::endl;
    }
    
    if (!createBuffers()) {
        std::cerr << "VulkanRenderEngine: Failed to create buffers" << std::endl;
        return false;
    }
    
    if (!createComputePipelines()) {
        std::cerr << "VulkanRenderEngine: Failed to create compute pipelines" << std::endl;
        return false;
    }
    
    initialized_ = true;
    syncBlock_.valid = true;
    
    std::cout << "VulkanRenderEngine: Initialized successfully" << std::endl;
    std::cout << "  Device: " << vulkan_.physicalDevice << std::endl;
    std::cout << "  Sample Rate: " << config_.sampleRate << "Hz" << std::endl;
    std::cout << "  Buffer Size: " << config_.bufferSize << " frames" << std::endl;
    std::cout << "  Channels: " << config_.channels << std::endl;
    std::cout << "  Timestamp Support: " << (vulkan_.timestampSupported ? "Yes" : "Limited") << std::endl;
    
    return true;
}

void VulkanRenderEngine::shutdown() {
    if (!initialized_) return;
    
    if (vulkan_.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vulkan_.device);
        
        // Cleanup pipelines
        if (pipelines_.audioPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vulkan_.device, pipelines_.audioPipeline, nullptr);
        }
        if (pipelines_.pnbtrPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vulkan_.device, pipelines_.pnbtrPipeline, nullptr);
        }
        if (pipelines_.audioLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vulkan_.device, pipelines_.audioLayout, nullptr);
        }
        if (pipelines_.pnbtrLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vulkan_.device, pipelines_.pnbtrLayout, nullptr);
        }
        if (pipelines_.descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vulkan_.device, pipelines_.descriptorPool, nullptr);
        }
        if (pipelines_.descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vulkan_.device, pipelines_.descriptorSetLayout, nullptr);
        }
        if (pipelines_.audioShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkan_.device, pipelines_.audioShader, nullptr);
        }
        if (pipelines_.pnbtrShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkan_.device, pipelines_.pnbtrShader, nullptr);
        }
        
        // Cleanup buffers
        if (buffers_.inputMemory != VK_NULL_HANDLE) {
            if (buffers_.inputMapped) {
                vkUnmapMemory(vulkan_.device, buffers_.inputMemory);
            }
            vkFreeMemory(vulkan_.device, buffers_.inputMemory, nullptr);
        }
        if (buffers_.inputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkan_.device, buffers_.inputBuffer, nullptr);
        }
        
        if (buffers_.outputMemory != VK_NULL_HANDLE) {
            if (buffers_.outputMapped) {
                vkUnmapMemory(vulkan_.device, buffers_.outputMemory);
            }
            vkFreeMemory(vulkan_.device, buffers_.outputMemory, nullptr);
        }
        if (buffers_.outputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkan_.device, buffers_.outputBuffer, nullptr);
        }
        
        if (buffers_.timestampMemory != VK_NULL_HANDLE) {
            if (buffers_.timestampMapped) {
                vkUnmapMemory(vulkan_.device, buffers_.timestampMemory);
            }
            vkFreeMemory(vulkan_.device, buffers_.timestampMemory, nullptr);
        }
        if (buffers_.timestampBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkan_.device, buffers_.timestampBuffer, nullptr);
        }
        
        // Cleanup command pool and queries
        if (vulkan_.timestampQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(vulkan_.device, vulkan_.timestampQueryPool, nullptr);
        }
        if (vulkan_.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(vulkan_.device, vulkan_.commandPool, nullptr);
        }
        
        vkDestroyDevice(vulkan_.device, nullptr);
    }
    
    if (vulkan_.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vulkan_.instance, nullptr);
    }
    
    initialized_ = false;
    syncBlock_.valid = false;
    
    std::cout << "VulkanRenderEngine: Shutdown complete" << std::endl;
}

bool VulkanRenderEngine::renderToBuffer(const float* inputSamples, uint32_t numInputSamples) {
    if (!initialized_) return false;
    
    // Validate input parameters
    if (numInputSamples > config_.bufferSize) {
        std::cerr << "VulkanRenderEngine: Input samples exceed buffer size: " 
                  << numInputSamples << " > " << config_.bufferSize << std::endl;
        return false;
    }
    
    // Record start time for performance monitoring
    auto renderStart = std::chrono::high_resolution_clock::now();
    
    // Copy input samples to GPU buffer
    if (inputSamples && numInputSamples > 0) {
        size_t inputSize = numInputSamples * sizeof(float);
        if (inputSize <= buffers_.bufferSize) {
            memcpy(buffers_.inputMapped, inputSamples, inputSize);
        }
    }
    
    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(vulkan_.commandBuffer, &beginInfo) != VK_SUCCESS) {
        return false;
    }
    
    // Start timestamp query
    if (vulkan_.timestampSupported) {
        vkCmdResetQueryPool(vulkan_.commandBuffer, vulkan_.timestampQueryPool, 0, 2);
        vkCmdWriteTimestamp(vulkan_.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
                           vulkan_.timestampQueryPool, 0);
    }
    
    // Dispatch audio compute shader
    vkCmdBindPipeline(vulkan_.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_.audioPipeline);
    vkCmdBindDescriptorSets(vulkan_.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelines_.audioLayout, 0, 1, &pipelines_.descriptorSet, 0, nullptr);
    
    // Dispatch compute work groups based on buffer size
    uint32_t workGroups = (config_.bufferSize + 63) / 64; // 64 threads per work group
    vkCmdDispatch(vulkan_.commandBuffer, workGroups, 1, 1);
    
    // Memory barrier to ensure compute completion
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    
    vkCmdPipelineBarrier(vulkan_.commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
    
    // End timestamp query
    if (vulkan_.timestampSupported) {
        vkCmdWriteTimestamp(vulkan_.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                           vulkan_.timestampQueryPool, 1);
    }
    
    // End and submit command buffer
    if (vkEndCommandBuffer(vulkan_.commandBuffer) != VK_SUCCESS) {
        return false;
    }
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vulkan_.commandBuffer;
    
    if (vkQueueSubmit(vulkan_.computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        return false;
    }
    
    // Wait for completion
    vkQueueWaitIdle(vulkan_.computeQueue);
    
    // Record completion time
    auto renderEnd = std::chrono::high_resolution_clock::now();
    auto renderDuration = std::chrono::duration_cast<std::chrono::microseconds>(renderEnd - renderStart);
    
    // Check if rendering took too long (should be < 1ms for real-time)
    if (renderDuration.count() > 1000) {
        std::cerr << "VulkanRenderEngine: Warning - GPU render took " << renderDuration.count() 
                  << "µs (> 1ms threshold)" << std::endl;
    }
    
    // Update timestamp
    if (vulkan_.timestampSupported) {
        lastGPUTimestamp_ = getGPUTimestamp();
    } else {
        // Fallback to system time
        auto now = std::chrono::high_resolution_clock::now();
        lastGPUTimestamp_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    
    return true;
}

bool VulkanRenderEngine::isGPUAvailable() const {
    return initialized_ && vulkan_.device != VK_NULL_HANDLE;
}

uint64_t VulkanRenderEngine::getCurrentTimestamp() const {
    if (vulkan_.timestampSupported) {
        return getGPUTimestamp();
    } else {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
}

void VulkanRenderEngine::updateSyncCalibration(uint64_t hostTimeNs, float offsetMs) {
    syncBlock_.gpu_time_ns = lastGPUTimestamp_;
    syncBlock_.host_time_ns = hostTimeNs;
    calibrationOffset_ = offsetMs;
    syncBlock_.calibration_offset_ms = offsetMs;
}

void* VulkanRenderEngine::getSharedBuffer() const {
    return buffers_.outputMapped;
}

size_t VulkanRenderEngine::getSharedBufferSize() const {
    return buffers_.bufferSize;
}

bool VulkanRenderEngine::flushBufferToGPU() {
    // For coherent memory, no explicit flush needed
    // For non-coherent memory, would need vkFlushMappedMemoryRanges
    return true;
}

// Implementation continues with private methods...
bool VulkanRenderEngine::setupVulkan() {
    // Create Vulkan instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "JAMNet VulkanRenderEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "JAMNet";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    if (vkCreateInstance(&createInfo, nullptr, &vulkan_.instance) != VK_SUCCESS) {
        return false;
    }
    
    // Find physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(vulkan_.instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(vulkan_.instance, &deviceCount, devices.data());
    
    // Pick first device with compute support
    vulkan_.physicalDevice = VK_NULL_HANDLE;
    for (const auto& device : devices) {
        if (findComputeQueue()) {
            vulkan_.physicalDevice = device;
            break;
        }
    }
    
    if (vulkan_.physicalDevice == VK_NULL_HANDLE) {
        return false;
    }
    
    // Create logical device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = vulkan_.computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    
    if (vkCreateDevice(vulkan_.physicalDevice, &deviceCreateInfo, nullptr, &vulkan_.device) != VK_SUCCESS) {
        return false;
    }
    
    // Get compute queue
    vkGetDeviceQueue(vulkan_.device, vulkan_.computeQueueFamily, 0, &vulkan_.computeQueue);
    
    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = vulkan_.computeQueueFamily;
    
    if (vkCreateCommandPool(vulkan_.device, &poolInfo, nullptr, &vulkan_.commandPool) != VK_SUCCESS) {
        return false;
    }
    
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkan_.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    if (vkAllocateCommandBuffers(vulkan_.device, &allocInfo, &vulkan_.commandBuffer) != VK_SUCCESS) {
        return false;
    }
    
    return true;
}

bool VulkanRenderEngine::findComputeQueue() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vulkan_.physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vulkan_.physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vulkan_.computeQueueFamily = i;
            return true;
        }
    }
    
    return false;
}

bool VulkanRenderEngine::createBuffers() {
    // Calculate buffer size
    buffers_.bufferSize = config_.bufferSize * config_.channels * sizeof(float);
    
    // Create input buffer
    if (!createBuffer(buffers_.bufferSize, 
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     buffers_.inputBuffer, buffers_.inputMemory, &buffers_.inputMapped)) {
        return false;
    }
    
    // Create output buffer (this is the shared buffer for JACK)
    if (!createBuffer(buffers_.bufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     buffers_.outputBuffer, buffers_.outputMemory, &buffers_.outputMapped)) {
        return false;
    }
    
    // Create timestamp buffer
    if (!createBuffer(sizeof(uint64_t) * 2,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     buffers_.timestampBuffer, buffers_.timestampMemory, &buffers_.timestampMapped)) {
        return false;
    }
    
    return true;
}

bool VulkanRenderEngine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags properties,
                                     VkBuffer& buffer, VkDeviceMemory& memory, void** mapped) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(vulkan_.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan_.device, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(vulkan_.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        return false;
    }
    
    vkBindBufferMemory(vulkan_.device, buffer, memory, 0);
    
    if (mapped) {
        if (vkMapMemory(vulkan_.device, memory, 0, size, 0, mapped) != VK_SUCCESS) {
            return false;
        }
    }
    
    return true;
}

uint32_t VulkanRenderEngine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(vulkan_.physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    return 0; // Should not reach here in a proper implementation
}

bool VulkanRenderEngine::queryTimestampSupport() {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(vulkan_.physicalDevice, &deviceProperties);
    
    vulkan_.timestampSupported = deviceProperties.limits.timestampComputeAndGraphics;
    vulkan_.timestampPeriod = deviceProperties.limits.timestampPeriod;
    
    if (vulkan_.timestampSupported) {
        // Create timestamp query pool
        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2; // Start and end timestamps
        
        if (vkCreateQueryPool(vulkan_.device, &queryPoolInfo, nullptr, &vulkan_.timestampQueryPool) != VK_SUCCESS) {
            vulkan_.timestampSupported = false;
        }
    }
    
    return vulkan_.timestampSupported;
}

uint64_t VulkanRenderEngine::getGPUTimestamp() const {
    if (!vulkan_.timestampSupported) {
        return 0;
    }
    
    uint64_t timestamps[2];
    VkResult result = vkGetQueryPoolResults(vulkan_.device, vulkan_.timestampQueryPool, 0, 2,
                                           sizeof(timestamps), timestamps, sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    
    if (result == VK_SUCCESS) {
        // Convert GPU timestamp to nanoseconds
        return static_cast<uint64_t>(timestamps[1] * vulkan_.timestampPeriod);
    }
    
    return 0;
}

bool VulkanRenderEngine::createComputePipelines() {
    // For now, create minimal pipeline structure
    // In a complete implementation, this would load and compile actual SPIR-V shaders
    // For demonstration, we'll create a placeholder pipeline
    
    // Load and create audio processing shader
    if (!loadShader("shaders/vulkan/spirv/audio_processing.spv", pipelines_.audioShader)) {
        std::cerr << "VulkanRenderEngine: Failed to load audio processing shader" << std::endl;
        return false;
    }
    
    // Create audio processing compute pipeline
    VkComputePipelineCreateInfo audioComputePipelineInfo{};
    audioComputePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    audioComputePipelineInfo.layout = pipelines_.audioLayout;
    audioComputePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    audioComputePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    audioComputePipelineInfo.stage.module = pipelines_.audioShader;
    audioComputePipelineInfo.stage.pName = "main";
    
    if (vkCreateComputePipelines(vulkan_.device, VK_NULL_HANDLE, 1, &audioComputePipelineInfo, 
                                nullptr, &pipelines_.audioPipeline) != VK_SUCCESS) {
        std::cerr << "VulkanRenderEngine: Failed to create audio compute pipeline" << std::endl;
        return false;
    }
    
    // Load and create PNBTR prediction shader
    if (!loadShader("shaders/vulkan/spirv/pnbtr_predict.spv", pipelines_.pnbtrShader)) {
        std::cerr << "VulkanRenderEngine: Failed to load PNBTR prediction shader" << std::endl;
        return false;
    }
    
    // Create PNBTR layout (same as audio for now)
    pipelines_.pnbtrLayout = pipelines_.audioLayout;
    
    // Create PNBTR compute pipeline
    VkComputePipelineCreateInfo pnbtrComputePipelineInfo{};
    pnbtrComputePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pnbtrComputePipelineInfo.layout = pipelines_.pnbtrLayout;
    pnbtrComputePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pnbtrComputePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pnbtrComputePipelineInfo.stage.module = pipelines_.pnbtrShader;
    pnbtrComputePipelineInfo.stage.pName = "main";
    
    if (vkCreateComputePipelines(vulkan_.device, VK_NULL_HANDLE, 1, &pnbtrComputePipelineInfo,
                                nullptr, &pipelines_.pnbtrPipeline) != VK_SUCCESS) {
        std::cerr << "VulkanRenderEngine: Failed to create PNBTR compute pipeline" << std::endl;
        return false;
    }
    
    std::cout << "VulkanRenderEngine: Compute pipelines created successfully" << std::endl;
    return true;
}

// Shader loading helper function
bool VulkanRenderEngine::loadShader(const std::string& filename, VkShaderModule& shaderModule) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "VulkanRenderEngine: Failed to open shader file: " << filename << std::endl;
        return false;
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
    
    if (vkCreateShaderModule(vulkan_.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "VulkanRenderEngine: Failed to create shader module from: " << filename << std::endl;
        return false;
    }
    
    std::cout << "VulkanRenderEngine: Loaded shader: " << filename << std::endl;
    return true;
    return true;
}

// Factory function for creating Vulkan render engine
std::unique_ptr<GPURenderEngine> createVulkanRenderEngine() {
    return std::make_unique<VulkanRenderEngine>();
}

} // namespace JAMNet

#endif // __linux__
