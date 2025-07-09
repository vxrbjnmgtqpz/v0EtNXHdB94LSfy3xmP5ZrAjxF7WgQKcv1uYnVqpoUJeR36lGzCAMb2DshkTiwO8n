#include "TrainingMetrics.h"
#include <algorithm>
#include <cmath>

TrainingMetrics::TrainingMetrics() = default;
TrainingMetrics::~TrainingMetrics() = default;

void TrainingMetrics::prepare(double sampleRate, int blockSize)
{
    this->sampleRate = sampleRate;
    this->blockSize = blockSize;
    reset();
}

void TrainingMetrics::reset()
{
    currentSNR.store(0.0f);
    currentTHD.store(0.0f);
    currentLatency.store(0.0f);
    packetLossPercentage.store(0.0f);
    totalPackets.store(0);
    lostPackets.store(0);
    averageSNR.store(0.0f);
    averageTHD.store(0.0f);
    averageLatency.store(0.0f);
    updateCount.store(0);
}

void TrainingMetrics::updateFromGPU(float snr, float thd, float latency, 
                                   int totalPacketsCount, int lostPacketsCount)
{
    // Update current values
    currentSNR.store(snr);
    currentTHD.store(thd);
    currentLatency.store(latency);
    totalPackets.store(totalPacketsCount);
    lostPackets.store(lostPacketsCount);
    
    // Calculate packet loss percentage
    if (totalPacketsCount > 0) {
        float lossPercent = (static_cast<float>(lostPacketsCount) / static_cast<float>(totalPacketsCount)) * 100.0f;
        packetLossPercentage.store(lossPercent);
    } else {
        packetLossPercentage.store(0.0f);
    }
    
    // Update running averages
    updateRunningAverages();
}

void TrainingMetrics::updateRunningAverages()
{
    int count = updateCount.fetch_add(1) + 1;
    
    // Running average calculation using exponential moving average for efficiency
    float alpha = 1.0f / std::min(count, 100); // Limit window to last 100 updates
    
    float currentAvgSNR = averageSNR.load();
    float currentAvgTHD = averageTHD.load();
    float currentAvgLatency = averageLatency.load();
    
    // Exponential moving average: avg = alpha * current + (1 - alpha) * avg
    averageSNR.store(alpha * currentSNR.load() + (1.0f - alpha) * currentAvgSNR);
    averageTHD.store(alpha * currentTHD.load() + (1.0f - alpha) * currentAvgTHD);
    averageLatency.store(alpha * currentLatency.load() + (1.0f - alpha) * currentAvgLatency);
}
