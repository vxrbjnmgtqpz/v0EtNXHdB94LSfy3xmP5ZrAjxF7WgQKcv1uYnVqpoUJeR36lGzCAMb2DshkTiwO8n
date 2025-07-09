/*
  ==============================================================================

    PacketLossSimulator.h
    Created: Packet loss simulation for network training

    Generates realistic packet loss patterns for PNBTR training:
    - Configurable loss percentage and jitter
    - Burst loss patterns (realistic network behavior)
    - Integration with Metal GPU buffers

  ==============================================================================
*/

#pragma once

#include <vector>
#include <random>
#include <atomic>

//==============================================================================
class PacketLossSimulator
{
public:
    PacketLossSimulator();
    ~PacketLossSimulator();

    //==============================================================================
    // Initialization
    void prepare(double sampleRate, int blockSize);

    //==============================================================================
    // Loss pattern generation
    void generateLossMap(float lossPercentage, float jitterMs);
    
    //==============================================================================
    // Access to loss map (for GPU processing)
    const std::vector<bool>& getLossMap() const { return lossMap; }
    // WARNING: std::vector<bool>::data() is not a real pointer; workaround for compatibility
    void getLossMapCopy(uint8_t* dest, int maxPackets) const {
        int n = std::min(static_cast<int>(lossMap.size()), maxPackets);
        for (int i = 0; i < n; ++i) dest[i] = lossMap[i] ? 1 : 0;
    }
    int getPacketCount() const { return static_cast<int>(lossMap.size()); }
    
    //==============================================================================
    // Configuration
    void setLossPercentage(float percentage);
    void setJitterAmount(float jitterMs);
    void setBurstLength(int burstLength);
    
    //==============================================================================
    // Statistics
    int getTotalPackets() const { return totalPackets; }
    int getLostPackets() const { return lostPackets; }
    float getActualLossPercentage() const;

private:
    //==============================================================================
    // Processing parameters
    double sampleRate = 48000.0;
    int blockSize = 512;
    int packetSize = 64; // Samples per packet
    
    //==============================================================================
    // Loss simulation parameters
    std::atomic<float> lossPercentage{2.0f};
    std::atomic<float> jitterAmount{1.0f};
    std::atomic<int> burstLength{3}; // Average burst length
    
    //==============================================================================
    // Loss map and statistics
    std::vector<bool> lossMap;
    int totalPackets = 0;
    int lostPackets = 0;
    
    //==============================================================================
    // Random number generation
    std::random_device randomDevice;
    std::mt19937 randomEngine;
    std::uniform_real_distribution<float> uniformDist;
    
    //==============================================================================
    // Internal methods
    void calculatePacketCount();
    void generateBurstLoss();
    void generateRandomLoss();
    void applyJitter();
    
    // No copy constructor or assignment
    PacketLossSimulator(const PacketLossSimulator&) = delete;
    PacketLossSimulator& operator=(const PacketLossSimulator&) = delete;
}; 