#pragma once

#include "GPURenderEngine.h"

#ifdef __linux__

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

namespace JAMNet {

/**
 * @brief Vulkan-based GPU render engine for Linux
 * 
 * Implements GPU-native audio rendering using Vulkan compute shaders with
 * precise timing integration and JACK compatibility.
 */
class VulkanRenderEngine : public GPURenderEngine {
public:
    VulkanRenderEngine();
    virtual ~VulkanRenderEngine();

    // GPURenderEngine interface
    bool initialize(const RenderConfig& config) override;
    void shutdown() override;
    bool renderToBuffer(const float* inputSamples, uint32_t numInputSamples) override;
    bool isGPUAvailable() const override;
    uint64_t getCurrentTimestamp() const override;
    void updateSyncCalibration(uint64_t hostTimeNs, float offsetMs) override;
    
    // Vulkan-specific methods
    bool setupVulkan();
    bool createComputePipelines();
    bool createBuffers();
    bool setupTimestampQueries();
    
    // Buffer access for shared memory
    void* getSharedBuffer() const override;
    size_t getSharedBufferSize() const override;
    bool flushBufferToGPU() override;
    
    // PNBTR prediction methods
    bool renderPredictedAudio(uint32_t numSamples);

private:
    struct VulkanContext {
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue computeQueue;
        uint32_t computeQueueFamily;
        
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
        
        // Timestamp query support
        VkQueryPool timestampQueryPool;
        bool timestampSupported;
        float timestampPeriod;
    } vulkan_;

    struct AudioBuffers {
        VkBuffer inputBuffer;
        VkDeviceMemory inputMemory;
        void* inputMapped;
        
        VkBuffer outputBuffer;
        VkDeviceMemory outputMemory;
        void* outputMapped;
        
        VkBuffer timestampBuffer;
        VkDeviceMemory timestampMemory;
        void* timestampMapped;
        
        size_t bufferSize;
    } buffers_;

    struct ComputePipelines {
        VkShaderModule audioShader;
        VkShaderModule pnbtrShader;
        
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        
        VkPipelineLayout audioLayout;
        VkPipeline audioPipeline;
        
        VkPipelineLayout pnbtrLayout;
        VkPipeline pnbtrPipeline;
    } pipelines_;

    // Configuration and state
    RenderConfig config_;
    SyncCalibrationBlock syncBlock_;
    bool initialized_;
    
    // Timing
    uint64_t lastGPUTimestamp_;
    float calibrationOffset_;
    
    // Helper methods
    bool findComputeQueue();
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties,
                     VkBuffer& buffer, VkDeviceMemory& memory, void** mapped);
    bool loadShader(const std::string& filename, VkShaderModule& shaderModule);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool queryTimestampSupport();
    uint64_t getGPUTimestamp();
    uint64_t convertGPUToHostTime(uint64_t gpuTime);
};

} // namespace JAMNet

#endif // __linux__
