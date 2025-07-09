#include "TrainingMetrics.h"

TrainingMetrics::TrainingMetrics() = default;
TrainingMetrics::~TrainingMetrics() = default;
void TrainingMetrics::prepare(double, int) {}
void TrainingMetrics::reset() {}
void TrainingMetrics::updateFromGPU(float, float, float, int, int) {}
