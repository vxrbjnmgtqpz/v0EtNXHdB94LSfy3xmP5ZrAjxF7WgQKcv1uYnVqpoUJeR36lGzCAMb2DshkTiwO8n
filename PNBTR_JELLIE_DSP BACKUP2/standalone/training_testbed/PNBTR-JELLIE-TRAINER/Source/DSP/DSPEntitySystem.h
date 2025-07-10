/*
  ==============================================================================

    DSPEntitySystem.h
    Created: ECS-Style DSP Module System

    Implements Unity/Unreal-style Entity-Component-System for audio:
    - Entities = Audio processing nodes (like GameObjects)
    - Components = DSP modules (like Transform, Renderer, etc.)
    - Systems = Processing logic operating on entities
    
    Features:
    - Hot-swappable DSP modules without audio interruption
    - Component-based architecture
    - Signal routing via DAG (Directed Acyclic Graph)
    - Voice virtualization for efficient processing

  ==============================================================================
*/

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <typeindex>
#include <functional>

//==============================================================================
// Core ECS Types (Unity/Unreal style)

using EntityID = uint32_t;
using ComponentTypeID = std::type_index;

static constexpr EntityID INVALID_ENTITY = 0;
static constexpr size_t MAX_ENTITIES = 4096;
static constexpr size_t MAX_AUDIO_CHANNELS = 64;

//==============================================================================
/**
 * Audio Block - Standard audio processing unit
 * Like Unity's RenderTexture or Unreal's render targets
 */
struct AudioBlock {
    float* channels[MAX_AUDIO_CHANNELS];
    size_t numChannels = 2;
    size_t numFrames = 512;
    double sampleRate = 48000.0;
    uint64_t timestamp_us = 0;
    
    // Metadata for processing
    float gain = 1.0f;
    bool isSilent = false;
    uint32_t processingFlags = 0;
    
    AudioBlock() {
        for (size_t i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
            channels[i] = nullptr;
        }
    }
    
    // Utility methods
    void clearToSilence();
    void applyGain(float gainAmount);
    void copyFrom(const AudioBlock& source);
    void mixWith(const AudioBlock& source, float mixLevel = 1.0f);
};

//==============================================================================
/**
 * Base DSP Component (like Unity's Component base class)
 * All DSP modules inherit from this
 */
class DSPComponent {
public:
    DSPComponent(EntityID entity) : entityID(entity) {}
    virtual ~DSPComponent() = default;
    
    //==============================================================================
    // Component lifecycle (Unity-style)
    virtual void initialize(double sampleRate, size_t maxBufferSize) {}
    virtual void processAudio(AudioBlock& input, AudioBlock& output) = 0;
    virtual void cleanup() {}
    
    //==============================================================================
    // Real-time parameter updates (atomic for audio thread safety)
    virtual void setParameter(const std::string& name, float value) {}
    virtual float getParameter(const std::string& name) const { return 0.0f; }
    
    //==============================================================================
    // Component properties
    EntityID getEntityID() const { return entityID; }
    bool isEnabled() const { return enabled.load(); }
    void setEnabled(bool enable) { enabled.store(enable); }
    
    // Debugging and monitoring
    virtual std::string getName() const = 0;
    virtual size_t getLatencyFrames() const { return 0; }
    virtual float getCPUUsage() const { return 0.0f; }

protected:
    EntityID entityID;
    std::atomic<bool> enabled{true};
    
    // Performance monitoring
    mutable std::atomic<uint64_t> processCallCount{0};
    mutable std::atomic<float> averageProcessTime_us{0.0f};
};

//==============================================================================
/**
 * Signal Connection (DAG edge)
 * Defines audio routing between entities
 */
struct SignalConnection {
    EntityID sourceEntity;
    EntityID targetEntity;
    uint32_t sourcePort = 0;
    uint32_t targetPort = 0;
    float gain = 1.0f;
    bool enabled = true;
    
    // Connection metadata
    std::string connectionName;
    uint32_t priority = 0;  // For connection ordering
};

//==============================================================================
/**
 * DSP Entity (like Unity's GameObject)
 * Container for DSP components with signal routing
 */
class DSPEntity {
public:
    DSPEntity(EntityID id) : entityID(id) {}
    ~DSPEntity() = default;
    
    //==============================================================================
    // Component management (Unity AddComponent/GetComponent pattern)
    template<typename T, typename... Args>
    T* addComponent(Args&&... args) {
        static_assert(std::is_base_of<DSPComponent, T>::value, "T must inherit from DSPComponent");
        
        auto component = std::make_unique<T>(entityID, std::forward<Args>(args)...);
        T* componentPtr = component.get();
        
        ComponentTypeID typeID = std::type_index(typeid(T));
        components[typeID] = std::move(component);
        
        return componentPtr;
    }
    
    template<typename T>
    T* getComponent() {
        ComponentTypeID typeID = std::type_index(typeid(T));
        auto it = components.find(typeID);
        if (it != components.end()) {
            return static_cast<T*>(it->second.get());
        }
        return nullptr;
    }
    
    template<typename T>
    void removeComponent() {
        ComponentTypeID typeID = std::type_index(typeid(T));
        components.erase(typeID);
    }
    
    //==============================================================================
    // Entity properties
    EntityID getID() const { return entityID; }
    const std::string& getName() const { return entityName; }
    void setName(const std::string& name) { entityName = name; }
    
    bool isEnabled() const { return enabled.load(); }
    void setEnabled(bool enable) { enabled.store(enable); }
    
    //==============================================================================
    // Audio processing
    void processAllComponents(AudioBlock& input, AudioBlock& output);
    
    //==============================================================================
    // Signal routing
    void addInputConnection(const SignalConnection& connection);
    void addOutputConnection(const SignalConnection& connection);
    void removeConnection(EntityID otherEntity, uint32_t port = 0);
    
    const std::vector<SignalConnection>& getInputConnections() const { return inputConnections; }
    const std::vector<SignalConnection>& getOutputConnections() const { return outputConnections; }

private:
    EntityID entityID;
    std::string entityName;
    std::atomic<bool> enabled{true};
    
    // Component storage
    std::unordered_map<ComponentTypeID, std::unique_ptr<DSPComponent>> components;
    
    // Signal routing (DAG connections)
    std::vector<SignalConnection> inputConnections;
    std::vector<SignalConnection> outputConnections;
    mutable std::mutex connectionsMutex;  // Protects connection modifications
};

//==============================================================================
/**
 * DSP Entity System (like Unity's ComponentSystem)
 * Manages all DSP entities and processes the audio graph
 */
class DSPEntitySystem {
public:
    DSPEntitySystem();
    ~DSPEntitySystem();
    
    //==============================================================================
    // System lifecycle
    bool initialize(double sampleRate, size_t maxBufferSize);
    void shutdown();
    
    //==============================================================================
    // Entity management (Unity-style)
    EntityID createEntity(const std::string& name = "");
    void destroyEntity(EntityID entityID);
    DSPEntity* getEntity(EntityID entityID);
    DSPEntity* findEntityByName(const std::string& name);
    
    //==============================================================================
    // Signal graph management (DAG)
    bool connectEntities(EntityID source, EntityID target, 
                        uint32_t sourcePort = 0, uint32_t targetPort = 0,
                        float gain = 1.0f);
    void disconnectEntities(EntityID source, EntityID target);
    
    //==============================================================================
    // Real-time audio processing
    void processAudioGraph(AudioBlock& inputBlock, AudioBlock& outputBlock);
    
    //==============================================================================
    // Hot-swapping support (real-time safe)
    void requestComponentSwap(EntityID entityID, std::function<void(DSPEntity*)> swapFunction);
    void processSwapRequests(); // Called on audio thread
    
    //==============================================================================
    // Voice virtualization
    void setMaxActiveVoices(size_t maxVoices) { maxActiveVoices = maxVoices; }
    size_t getActiveVoiceCount() const { return activeVoiceCount.load(); }
    
    //==============================================================================
    // Performance monitoring
    struct SystemStats {
        size_t totalEntities = 0;
        size_t activeEntities = 0;
        size_t totalConnections = 0;
        float averageProcessTime_ms = 0.0f;
        float peakProcessTime_ms = 0.0f;
        size_t activeVoices = 0;
        size_t virtualizedVoices = 0;
    };
    
    SystemStats getStats() const { return stats; }

private:
    //==============================================================================
    // Entity storage
    std::unordered_map<EntityID, std::unique_ptr<DSPEntity>> entities;
    EntityID nextEntityID = 1;
    mutable std::mutex entitiesMutex;
    
    //==============================================================================
    // Audio graph processing
    void buildProcessingOrder();  // Topological sort of DAG
    std::vector<EntityID> processingOrder;
    bool graphNeedsRebuild = true;
    
    //==============================================================================
    // Hot-swapping infrastructure
    struct SwapRequest {
        EntityID entityID;
        std::function<void(DSPEntity*)> swapFunction;
    };
    
    std::vector<SwapRequest> pendingSwaps;
    std::mutex swapMutex;
    
    //==============================================================================
    // Voice virtualization
    std::vector<EntityID> activeVoices;
    std::atomic<size_t> maxActiveVoices{32};
    std::atomic<size_t> activeVoiceCount{0};
    
    void updateVoiceVirtualization();
    
    //==============================================================================
    // System configuration
    double currentSampleRate = 48000.0;
    size_t currentMaxBufferSize = 2048;
    std::atomic<bool> initialized{false};
    
    //==============================================================================
    // Performance monitoring
    mutable SystemStats stats;
    
    // Non-copyable
    DSPEntitySystem(const DSPEntitySystem&) = delete;
    DSPEntitySystem& operator=(const DSPEntitySystem&) = delete;
}; 