/*
  ==============================================================================

    TrainingMetrics.h
    Created: Training metrics management for GPU-native processing

    Manages metrics calculation and storage for PNBTR+JELLIE training:
    - SNR, THD, latency from GPU kernels
    - Packet loss statistics
    - Real-time performance monitoring

  ==============================================================================
*/

#pragma once

#include <atomic>
#include <memory>

//==============================================================================
class TrainingMetrics
{
public:
    TrainingMetrics();
    ~TrainingMetrics();

    //==============================================================================
    // Initialization
    void prepare(double sampleRate, int blockSize);
    void reset();

    //==============================================================================
    // GPU metrics update (called from processBlock)
    void updateFromGPU(float snr, float thd, float latency, 
                      int totalPackets, int lostPackets);

    //==============================================================================
    // Metrics access (thread-safe)
    float getSNR() const { return currentSNR.load(); }
    float getTHD() const { return currentTHD.load(); }
    float getLatency() const { return currentLatency.load(); }
    
    float getPacketLossPercentage() const { return packetLossPercentage.load(); }
    int getTotalPackets() const { return totalPackets.load(); }
    int getLostPackets() const { return lostPackets.load(); }
    
    //==============================================================================
    // Statistics
    float getAverageSNR() const { return averageSNR.load(); }
    float getAverageTHD() const { return averageTHD.load(); }
    float getAverageLatency() const { return averageLatency.load(); }
    
    int getUpdateCount() const { return updateCount.load(); }
    
private:
    //==============================================================================
    // Current metrics (atomic for thread safety)
    std::atomic<float> currentSNR{0.0f};
    std::atomic<float> currentTHD{0.0f};
    std::atomic<float> currentLatency{0.0f};
    
    std::atomic<float> packetLossPercentage{0.0f};
    std::atomic<int> totalPackets{0};
    std::atomic<int> lostPackets{0};
    
    //==============================================================================
    // Running averages
    std::atomic<float> averageSNR{0.0f};
    std::atomic<float> averageTHD{0.0f};
    std::atomic<float> averageLatency{0.0f};
    
    std::atomic<int> updateCount{0};
    
    //==============================================================================
    // Processing parameters
    double sampleRate = 48000.0;
    int blockSize = 512;
    
    //==============================================================================
    // Internal methods
    void updateRunningAverages();
    
    // No copy constructor or assignment
    TrainingMetrics(const TrainingMetrics&) = delete;
    TrainingMetrics& operator=(const TrainingMetrics&) = delete;
}; 