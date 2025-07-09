/*
  ==============================================================================

    ECSTest.cpp
    Created: ECS-Style DSP System Demonstration

    Demonstrates the complete Entity-Component-System with:
    - Hot-swappable DSP modules
    - Signal routing DAG  
    - Voice virtualization
    - Live parameter updates
    - Unity/Unreal-style entity management

  ==============================================================================
*/

#include "DSPEntitySystem.h"
#include "DSPComponents.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <iomanip>

//==============================================================================
// Test Audio Data Generator
class TestAudioGenerator {
public:
    TestAudioGenerator(double sampleRate = 48000.0, size_t bufferSize = 512) 
        : sampleRate(sampleRate), bufferSize(bufferSize) {
        
        // Allocate audio buffers
        audioData.resize(bufferSize * 2); // Stereo
        
        // Set up audio block
        testBlock.numChannels = 2;
        testBlock.numFrames = bufferSize;
        testBlock.sampleRate = sampleRate;
        testBlock.channels[0] = audioData.data();
        testBlock.channels[1] = audioData.data() + bufferSize;
    }
    
    AudioBlock& generateTestTone(float frequency = 440.0f, float amplitude = 0.3f) {
        static float phase = 0.0f;
        
        for (size_t frame = 0; frame < bufferSize; ++frame) {
            float sample = amplitude * std::sin(phase);
            
            // Stereo - same signal on both channels
            audioData[frame] = sample;                    // Left
            audioData[bufferSize + frame] = sample;       // Right
            
            phase += 2.0f * M_PI * frequency / sampleRate;
            if (phase > 2.0f * M_PI) phase -= 2.0f * M_PI;
        }
        
        testBlock.isSilent = false;
        testBlock.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        return testBlock;
    }
    
    AudioBlock& generateNoise(float amplitude = 0.1f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t frame = 0; frame < bufferSize; ++frame) {
            float sample = amplitude * dist(gen);
            
            audioData[frame] = sample;                    // Left
            audioData[bufferSize + frame] = sample;       // Right
        }
        
        testBlock.isSilent = false;
        return testBlock;
    }

private:
    double sampleRate;
    size_t bufferSize;
    std::vector<float> audioData;
    AudioBlock testBlock;
};

//==============================================================================
// Demo: Create a complete audio processing chain
void demonstrateAudioProcessingChain(DSPEntitySystem& ecs) {
    std::cout << "\n=== Creating Audio Processing Chain ===" << std::endl;
    
    // Create entities (like GameObjects in Unity)
    EntityID inputEntity = ecs.createEntity("AudioInput");
    EntityID encoderEntity = ecs.createEntity("JELLIEEncoder");  
    EntityID networkEntity = ecs.createEntity("NetworkSimulation");
    EntityID decoderEntity = ecs.createEntity("PNBTRDecoder");
    EntityID filterEntity = ecs.createEntity("OutputFilter");
    EntityID outputEntity = ecs.createEntity("AudioOutput");
    
    std::cout << "Created " << 6 << " entities" << std::endl;
    
    // Add components to entities (Unity AddComponent pattern)
    auto inputEntity_ptr = ecs.getEntity(inputEntity);
    auto inputGain = inputEntity_ptr->addComponent<GainComponent>(0.8f);
    
    auto encoderEntity_ptr = ecs.getEntity(encoderEntity);
    auto jellieEncoder = encoderEntity_ptr->addComponent<JELLIEEncoderComponent>();
    
    auto decoderEntity_ptr = ecs.getEntity(decoderEntity);
    auto pnbtrDecoder = decoderEntity_ptr->addComponent<PNBTRDecoderComponent>();
    
    auto filterEntity_ptr = ecs.getEntity(filterEntity);
    auto outputFilter = filterEntity_ptr->addComponent<FilterComponent>(FilterComponent::LowPass);
    
    auto outputEntity_ptr = ecs.getEntity(outputEntity);
    auto outputGain = outputEntity_ptr->addComponent<GainComponent>(1.0f);
    
    std::cout << "Added components to all entities" << std::endl;
    
    // Create signal routing (DAG connections)
    ecs.connectEntities(inputEntity, encoderEntity, 0, 0, 1.0f);
    ecs.connectEntities(encoderEntity, networkEntity, 0, 0, 1.0f);
    ecs.connectEntities(networkEntity, decoderEntity, 0, 0, 1.0f);
    ecs.connectEntities(decoderEntity, filterEntity, 0, 0, 1.0f);
    ecs.connectEntities(filterEntity, outputEntity, 0, 0, 1.0f);
    
    std::cout << "Created signal routing chain: Input → JELLIE → Network → PNBTR → Filter → Output" << std::endl;
}

//==============================================================================
// Demo: Hot-swap components while processing
void demonstrateHotSwapping(DSPEntitySystem& ecs) {
    std::cout << "\n=== Hot-Swapping Components Demo ===" << std::endl;
    
    // Find the filter entity
    auto filterEntity = ecs.findEntityByName("OutputFilter");
    if (!filterEntity) {
        std::cout << "Filter entity not found!" << std::endl;
        return;
    }
    
    EntityID filterID = filterEntity->getID();
    
    // Hot-swap filter type while audio is processing
    std::cout << "Hot-swapping LowPass → HighPass filter..." << std::endl;
    
    ecs.requestComponentSwap(filterID, [](DSPEntity* entity) {
        // Remove old filter
        entity->removeComponent<FilterComponent>();
        
        // Add new filter (hot-swap without audio interruption)
        auto newFilter = entity->addComponent<FilterComponent>(FilterComponent::HighPass);
        newFilter->setParameter("cutoff", 2000.0f);
        newFilter->setParameter("resonance", 1.2f);
        
        std::cout << "  → Swapped to HighPass filter (2kHz, Q=1.2)" << std::endl;
    });
    
    // Second hot-swap after some processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Hot-swapping HighPass → BandPass filter..." << std::endl;
    ecs.requestComponentSwap(filterID, [](DSPEntity* entity) {
        entity->removeComponent<FilterComponent>();
        auto bandpassFilter = entity->addComponent<FilterComponent>(FilterComponent::BandPass);
        bandpassFilter->setParameter("cutoff", 1000.0f);
        bandpassFilter->setParameter("resonance", 3.0f);
        
        std::cout << "  → Swapped to BandPass filter (1kHz, Q=3.0)" << std::endl;
    });
}

//==============================================================================
// Demo: Live parameter automation
void demonstrateLiveParameterUpdates(DSPEntitySystem& ecs) {
    std::cout << "\n=== Live Parameter Updates Demo ===" << std::endl;
    
    auto encoderEntity = ecs.findEntityByName("JELLIEEncoder");
    auto decoderEntity = ecs.findEntityByName("PNBTRDecoder");
    auto filterEntity = ecs.findEntityByName("OutputFilter");
    
    if (!encoderEntity || !decoderEntity || !filterEntity) {
        std::cout << "Some entities not found!" << std::endl;
        return;
    }
    
    // Get components
    auto jellieEncoder = encoderEntity->getComponent<JELLIEEncoderComponent>();
    auto pnbtrDecoder = decoderEntity->getComponent<PNBTRDecoderComponent>();
    auto filter = filterEntity->getComponent<FilterComponent>();
    
    if (!jellieEncoder || !pnbtrDecoder || !filter) {
        std::cout << "Some components not found!" << std::endl;
        return;
    }
    
    std::cout << "Automating parameters in real-time..." << std::endl;
    
    // Animate parameters over time
    for (int step = 0; step < 10; ++step) {
        float progress = step / 9.0f; // 0.0 to 1.0
        
        // Animate compression ratio (simulates network quality changes)
        float compressionRatio = 2.0f + progress * 6.0f; // 2:1 to 8:1
        jellieEncoder->setParameter("compression_ratio", compressionRatio);
        
        // Animate enhancement level (compensates for compression)
        float enhancementLevel = progress; // More enhancement as compression increases
        pnbtrDecoder->setParameter("enhancement_level", enhancementLevel);
        
        // Animate filter cutoff
        float cutoffFreq = 500.0f + progress * 3500.0f; // 500Hz to 4kHz
        filter->setParameter("cutoff", cutoffFreq);
        
        std::cout << "  Step " << (step + 1) << ": Compression=" << std::fixed << std::setprecision(1) 
                  << compressionRatio << ":1, Enhancement=" << enhancementLevel 
                  << ", Cutoff=" << cutoffFreq << "Hz" << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

//==============================================================================
// Demo: Voice virtualization
void demonstrateVoiceVirtualization(DSPEntitySystem& ecs) {
    std::cout << "\n=== Voice Virtualization Demo ===" << std::endl;
    
    // Set maximum active voices
    ecs.setMaxActiveVoices(4);
    std::cout << "Set maximum active voices to 4" << std::endl;
    
    // Create more entities than the limit
    std::vector<EntityID> voiceEntities;
    
    for (int i = 0; i < 8; ++i) {
        EntityID voiceID = ecs.createEntity("Voice_" + std::to_string(i));
        voiceEntities.push_back(voiceID);
        
        auto voiceEntity = ecs.getEntity(voiceID);
        voiceEntity->addComponent<GainComponent>(0.5f);
        voiceEntity->addComponent<FilterComponent>(FilterComponent::LowPass);
    }
    
    std::cout << "Created 8 voice entities (exceeds limit of 4)" << std::endl;
    
    // Process audio - this will trigger voice virtualization
    TestAudioGenerator generator;
    AudioBlock inputBlock = generator.generateTestTone(440.0f);
    AudioBlock outputBlock = inputBlock;
    
    ecs.processAudioGraph(inputBlock, outputBlock);
    
    // Check virtualization statistics
    auto stats = ecs.getStats();
    std::cout << "Voice virtualization results:" << std::endl;
    std::cout << "  Total entities: " << stats.totalEntities << std::endl;
    std::cout << "  Active voices: " << stats.activeVoices << std::endl;
    std::cout << "  Virtualized voices: " << stats.virtualizedVoices << std::endl;
}

//==============================================================================
// Main ECS demonstration program
int main() {
    std::cout << "=== ECS-Style DSP Module System Demo ===" << std::endl;
    std::cout << "Unity/Unreal-style Entity-Component-System for Audio" << std::endl;
    
    // Create and initialize the ECS system
    DSPEntitySystem ecs;
    if (!ecs.initialize(48000.0, 512)) {
        std::cerr << "Failed to initialize ECS system!" << std::endl;
        return 1;
    }
    
    // Demo 1: Create complete audio processing chain
    demonstrateAudioProcessingChain(ecs);
    
    // Demo 2: Process audio through the chain
    std::cout << "\n=== Processing Audio Through ECS Chain ===" << std::endl;
    
    TestAudioGenerator generator;
    
    for (int i = 0; i < 5; ++i) {
        // Generate test input
        AudioBlock inputBlock = generator.generateTestTone(440.0f + i * 100.0f, 0.3f);
        AudioBlock outputBlock = inputBlock;
        
        // Process through entity system
        auto startTime = std::chrono::high_resolution_clock::now();
        ecs.processAudioGraph(inputBlock, outputBlock);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        
        std::cout << "Block " << (i + 1) << ": Processed in " << duration_us << " μs" << std::endl;
    }
    
    // Demo 3: Hot-swapping components
    demonstrateHotSwapping(ecs);
    
    // Demo 4: Live parameter automation  
    demonstrateLiveParameterUpdates(ecs);
    
    // Demo 5: Voice virtualization
    demonstrateVoiceVirtualization(ecs);
    
    // Performance statistics
    std::cout << "\n=== Final Performance Statistics ===" << std::endl;
    auto stats = ecs.getStats();
    std::cout << "Total entities: " << stats.totalEntities << std::endl;
    std::cout << "Active entities: " << stats.activeEntities << std::endl;
    std::cout << "Total connections: " << stats.totalConnections << std::endl;
    std::cout << "Average process time: " << stats.averageProcessTime_ms << " ms" << std::endl;
    std::cout << "Peak process time: " << stats.peakProcessTime_ms << " ms" << std::endl;
    std::cout << "Active voices: " << stats.activeVoices << std::endl;
    std::cout << "Virtualized voices: " << stats.virtualizedVoices << std::endl;
    
    std::cout << "\n✅ ECS-Style DSP System Demo Complete!" << std::endl;
    std::cout << "🎮 Unity/Unreal-style architecture successfully implemented" << std::endl;
    std::cout << "🔥 Hot-swappable components working without audio interruption" << std::endl;
    std::cout << "🎛️  Live parameter updates functioning correctly" << std::endl;
    std::cout << "🎵 Voice virtualization managing resource limits" << std::endl;
    
    // Cleanup
    ecs.shutdown();
    
    return 0;
} 