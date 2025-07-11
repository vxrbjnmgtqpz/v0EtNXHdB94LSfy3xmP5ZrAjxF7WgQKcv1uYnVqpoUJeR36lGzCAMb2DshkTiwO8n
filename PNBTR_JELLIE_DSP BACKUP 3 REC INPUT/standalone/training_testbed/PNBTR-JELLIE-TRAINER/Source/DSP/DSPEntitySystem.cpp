/*
  ==============================================================================

    DSPEntitySystem.cpp
    Created: ECS-Style DSP Module System Implementation

    Implements the complete Entity-Component-System for audio processing
    with Unity/Unreal-style hot-swappable DSP modules.

  ==============================================================================
*/

#include "DSPEntitySystem.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <queue>
#include <set>

//==============================================================================
// AudioBlock utility implementations
void AudioBlock::clearToSilence() {
    for (size_t ch = 0; ch < numChannels; ++ch) {
        if (channels[ch]) {
            std::fill(channels[ch], channels[ch] + numFrames, 0.0f);
        }
    }
    isSilent = true;
}

void AudioBlock::applyGain(float gainAmount) {
    if (gainAmount == 1.0f) return;
    
    for (size_t ch = 0; ch < numChannels; ++ch) {
        if (channels[ch]) {
            for (size_t frame = 0; frame < numFrames; ++frame) {
                channels[ch][frame] *= gainAmount;
            }
        }
    }
    
    if (gainAmount == 0.0f) {
        isSilent = true;
    } else if (isSilent && gainAmount != 0.0f) {
        isSilent = false;
    }
}

void AudioBlock::copyFrom(const AudioBlock& source) {
    size_t channelsToCopy = std::min(numChannels, source.numChannels);
    size_t framesToCopy = std::min(numFrames, source.numFrames);
    
    for (size_t ch = 0; ch < channelsToCopy; ++ch) {
        if (channels[ch] && source.channels[ch]) {
            std::copy(source.channels[ch], 
                     source.channels[ch] + framesToCopy, 
                     channels[ch]);
        }
    }
    
    isSilent = source.isSilent;
    timestamp_us = source.timestamp_us;
}

void AudioBlock::mixWith(const AudioBlock& source, float mixLevel) {
    size_t channelsToCopy = std::min(numChannels, source.numChannels);
    size_t framesToCopy = std::min(numFrames, source.numFrames);
    
    for (size_t ch = 0; ch < channelsToCopy; ++ch) {
        if (channels[ch] && source.channels[ch]) {
            for (size_t frame = 0; frame < framesToCopy; ++frame) {
                channels[ch][frame] += source.channels[ch][frame] * mixLevel;
            }
        }
    }
    
    if (!source.isSilent && mixLevel != 0.0f) {
        isSilent = false;
    }
}

//==============================================================================
// DSPEntity implementation
void DSPEntity::processAllComponents(AudioBlock& input, AudioBlock& output) {
    if (!enabled.load()) {
        output.copyFrom(input); // Pass through if disabled
        return;
    }
    
    // Process all components sequentially
    // In a more advanced system, this could be parallelized
    AudioBlock tempInput = input;
    AudioBlock tempOutput = output;
    
    for (auto& [typeID, component] : components) {
        if (component && component->isEnabled()) {
            component->processAudio(tempInput, tempOutput);
            tempInput = tempOutput; // Chain output to next input
        }
    }
}

void DSPEntity::addInputConnection(const SignalConnection& connection) {
    std::lock_guard<std::mutex> lock(connectionsMutex);
    inputConnections.push_back(connection);
}

void DSPEntity::addOutputConnection(const SignalConnection& connection) {
    std::lock_guard<std::mutex> lock(connectionsMutex);
    outputConnections.push_back(connection);
}

void DSPEntity::removeConnection(EntityID otherEntity, uint32_t port) {
    std::lock_guard<std::mutex> lock(connectionsMutex);
    
    // Remove from input connections
    inputConnections.erase(
        std::remove_if(inputConnections.begin(), inputConnections.end(),
            [otherEntity, port](const SignalConnection& conn) {
                return conn.sourceEntity == otherEntity && conn.sourcePort == port;
            }),
        inputConnections.end());
    
    // Remove from output connections
    outputConnections.erase(
        std::remove_if(outputConnections.begin(), outputConnections.end(),
            [otherEntity, port](const SignalConnection& conn) {
                return conn.targetEntity == otherEntity && conn.targetPort == port;
            }),
        outputConnections.end());
}

//==============================================================================
// DSPEntitySystem implementation
DSPEntitySystem::DSPEntitySystem() {
    std::cout << "[DSPEntitySystem] Created ECS audio processing system" << std::endl;
}

DSPEntitySystem::~DSPEntitySystem() {
    shutdown();
}

bool DSPEntitySystem::initialize(double sampleRate, size_t maxBufferSize) {
    if (initialized.load()) {
        std::cout << "[DSPEntitySystem] Already initialized" << std::endl;
        return true;
    }
    
    currentSampleRate = sampleRate;
    currentMaxBufferSize = maxBufferSize;
    
    // Initialize all existing entities
    std::lock_guard<std::mutex> lock(entitiesMutex);
    for (auto& [entityID, entity] : entities) {
        // Initialize all components in this entity
        // Note: This would need component iteration in a real implementation
    }
    
    initialized.store(true);
    std::cout << "[DSPEntitySystem] Initialized: " << sampleRate << "Hz, " 
              << maxBufferSize << " buffer size" << std::endl;
    
    return true;
}

void DSPEntitySystem::shutdown() {
    if (!initialized.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(entitiesMutex);
    
    // Cleanup all entities and components
    entities.clear();
    processingOrder.clear();
    activeVoices.clear();
    
    initialized.store(false);
    std::cout << "[DSPEntitySystem] Shutdown complete" << std::endl;
}

//==============================================================================
// Entity management (Unity-style)
EntityID DSPEntitySystem::createEntity(const std::string& name) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    
    EntityID newID = nextEntityID++;
    auto entity = std::make_unique<DSPEntity>(newID);
    
    if (!name.empty()) {
        entity->setName(name);
    } else {
        entity->setName("Entity_" + std::to_string(newID));
    }
    
    entities[newID] = std::move(entity);
    graphNeedsRebuild = true; // Mark for processing order rebuild
    
    std::cout << "[DSPEntitySystem] Created entity " << newID 
              << " (" << entities[newID]->getName() << ")" << std::endl;
    
    return newID;
}

void DSPEntitySystem::destroyEntity(EntityID entityID) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    
    auto it = entities.find(entityID);
    if (it != entities.end()) {
        std::cout << "[DSPEntitySystem] Destroying entity " << entityID 
                  << " (" << it->second->getName() << ")" << std::endl;
        
        // Remove all connections involving this entity
        for (auto& [otherID, otherEntity] : entities) {
            if (otherID != entityID) {
                otherEntity->removeConnection(entityID);
            }
        }
        
        entities.erase(it);
        graphNeedsRebuild = true;
    }
}

DSPEntity* DSPEntitySystem::getEntity(EntityID entityID) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    auto it = entities.find(entityID);
    return (it != entities.end()) ? it->second.get() : nullptr;
}

DSPEntity* DSPEntitySystem::findEntityByName(const std::string& name) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    for (auto& [entityID, entity] : entities) {
        if (entity->getName() == name) {
            return entity.get();
        }
    }
    return nullptr;
}

//==============================================================================
// Signal graph management (DAG)
bool DSPEntitySystem::connectEntities(EntityID source, EntityID target, 
                                      uint32_t sourcePort, uint32_t targetPort,
                                      float gain) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    
    auto sourceEntity = getEntity(source);
    auto targetEntity = getEntity(target);
    
    if (!sourceEntity || !targetEntity) {
        std::cerr << "[DSPEntitySystem] Cannot connect - invalid entities: " 
                  << source << " -> " << target << std::endl;
        return false;
    }
    
    // Check for cycles (simplified cycle detection)
    // In a production system, this would use a proper topological sort
    if (source == target) {
        std::cerr << "[DSPEntitySystem] Cannot connect entity to itself: " << source << std::endl;
        return false;
    }
    
    // Create connection
    SignalConnection connection;
    connection.sourceEntity = source;
    connection.targetEntity = target;
    connection.sourcePort = sourcePort;
    connection.targetPort = targetPort;
    connection.gain = gain;
    connection.enabled = true;
    connection.connectionName = sourceEntity->getName() + " -> " + targetEntity->getName();
    
    // Add to both entities
    sourceEntity->addOutputConnection(connection);
    targetEntity->addInputConnection(connection);
    
    graphNeedsRebuild = true;
    
    std::cout << "[DSPEntitySystem] Connected " << connection.connectionName 
              << " (gain: " << gain << ")" << std::endl;
    
    return true;
}

void DSPEntitySystem::disconnectEntities(EntityID source, EntityID target) {
    std::lock_guard<std::mutex> lock(entitiesMutex);
    
    auto sourceEntity = getEntity(source);
    auto targetEntity = getEntity(target);
    
    if (sourceEntity && targetEntity) {
        sourceEntity->removeConnection(target);
        targetEntity->removeConnection(source);
        graphNeedsRebuild = true;
        
        std::cout << "[DSPEntitySystem] Disconnected " << source << " -> " << target << std::endl;
    }
}

//==============================================================================
// Audio graph processing (topological sort + processing)
void DSPEntitySystem::buildProcessingOrder() {
    if (!graphNeedsRebuild) {
        return;
    }
    
    processingOrder.clear();
    std::set<EntityID> visited;
    std::set<EntityID> recursionStack;
    
    // Simple topological sort implementation
    std::function<bool(EntityID)> topologicalSortUtil = [&](EntityID entityID) -> bool {
        visited.insert(entityID);
        recursionStack.insert(entityID);
        
        auto entity = getEntity(entityID);
        if (!entity) return true;
        
        // Visit all entities this one connects to
        for (const auto& connection : entity->getOutputConnections()) {
            EntityID targetID = connection.targetEntity;
            
            if (recursionStack.find(targetID) != recursionStack.end()) {
                // Cycle detected
                std::cerr << "[DSPEntitySystem] Cycle detected in audio graph!" << std::endl;
                return false;
            }
            
            if (visited.find(targetID) == visited.end()) {
                if (!topologicalSortUtil(targetID)) {
                    return false;
                }
            }
        }
        
        recursionStack.erase(entityID);
        processingOrder.insert(processingOrder.begin(), entityID); // Prepend for correct order
        return true;
    };
    
    // Process all entities
    for (const auto& [entityID, entity] : entities) {
        if (visited.find(entityID) == visited.end()) {
            if (!topologicalSortUtil(entityID)) {
                // Graph has cycles, use original order
                processingOrder.clear();
                for (const auto& [id, ent] : entities) {
                    processingOrder.push_back(id);
                }
                break;
            }
        }
    }
    
    graphNeedsRebuild = false;
    std::cout << "[DSPEntitySystem] Built processing order for " << processingOrder.size() 
              << " entities" << std::endl;
}

void DSPEntitySystem::processAudioGraph(AudioBlock& inputBlock, AudioBlock& outputBlock) {
    if (!initialized.load()) {
        outputBlock.copyFrom(inputBlock);
        return;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Rebuild processing order if needed
    buildProcessingOrder();
    
    // Process hot-swap requests (real-time safe)
    processSwapRequests();
    
    // Update voice virtualization
    updateVoiceVirtualization();
    
    // Process entities in topological order
    // This is a simplified implementation - a production system would use
    // a more sophisticated graph execution with parallel processing
    
    AudioBlock currentBlock = inputBlock;
    
    for (EntityID entityID : processingOrder) {
        auto entity = getEntity(entityID);
        if (entity && entity->isEnabled()) {
            AudioBlock entityOutput = currentBlock;
            entity->processAllComponents(currentBlock, entityOutput);
            currentBlock = entityOutput;
        }
    }
    
    outputBlock.copyFrom(currentBlock);
    
    // Update performance statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    float duration_ms = duration_us / 1000.0f;
    
    stats.averageProcessTime_ms = (stats.averageProcessTime_ms * 0.95f) + (duration_ms * 0.05f);
    stats.peakProcessTime_ms = std::max(stats.peakProcessTime_ms, duration_ms);
    stats.totalEntities = entities.size();
    
    // Count active entities
    stats.activeEntities = std::count_if(entities.begin(), entities.end(),
        [](const auto& pair) { return pair.second->isEnabled(); });
}

//==============================================================================
// Hot-swapping support (real-time safe)
void DSPEntitySystem::requestComponentSwap(EntityID entityID, 
                                          std::function<void(DSPEntity*)> swapFunction) {
    std::lock_guard<std::mutex> lock(swapMutex);
    
    SwapRequest request;
    request.entityID = entityID;
    request.swapFunction = swapFunction;
    
    pendingSwaps.push_back(request);
    
    std::cout << "[DSPEntitySystem] Queued hot-swap request for entity " << entityID << std::endl;
}

void DSPEntitySystem::processSwapRequests() {
    std::lock_guard<std::mutex> lock(swapMutex);
    
    for (const auto& request : pendingSwaps) {
        auto entity = getEntity(request.entityID);
        if (entity) {
            try {
                request.swapFunction(entity);
                std::cout << "[DSPEntitySystem] Executed hot-swap for entity " 
                          << request.entityID << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[DSPEntitySystem] Hot-swap failed for entity " 
                          << request.entityID << ": " << e.what() << std::endl;
            }
        }
    }
    
    pendingSwaps.clear();
}

//==============================================================================
// Voice virtualization
void DSPEntitySystem::updateVoiceVirtualization() {
    size_t maxVoices = maxActiveVoices.load();
    
    // Simple voice virtualization: disable least important entities if over limit
    if (entities.size() > maxVoices) {
        // In a real implementation, this would prioritize voices based on:
        // - Volume level
        // - Distance from listener
        // - Importance/priority
        // - Age of voice
        
        std::vector<EntityID> entitiesToVirtualize;
        size_t activeCount = 0;
        
        for (const auto& [entityID, entity] : entities) {
            if (entity->isEnabled()) {
                activeCount++;
                if (activeCount > maxVoices) {
                    entitiesToVirtualize.push_back(entityID);
                }
            }
        }
        
        // Virtualize excess entities
        for (EntityID entityID : entitiesToVirtualize) {
            auto entity = getEntity(entityID);
            if (entity) {
                entity->setEnabled(false);
                std::cout << "[DSPEntitySystem] Virtualized entity " << entityID << std::endl;
            }
        }
        
        stats.virtualizedVoices = entitiesToVirtualize.size();
    }
    
    activeVoiceCount.store(std::count_if(entities.begin(), entities.end(),
        [](const auto& pair) { return pair.second->isEnabled(); }));
    
    stats.activeVoices = activeVoiceCount.load();
} 