#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Struct for the AudioInputGateShader
typedef struct {
    float threshold;
    bool jellieRecordArmed;
    bool pnbtrRecordArmed;
} GateParams;

// Struct for the DJSpectralAnalysisShader
typedef struct {
    float fftSize;
    float sampleRate;
} DJAnalysisParams;

// Struct for the RecordArmVisualShader
typedef struct {
    float redLevel;
    float greenLevel;
} RecordArmVisualParams;

// Struct for the JELLIEPreprocessShader
typedef struct {
    float compressionRatio;
    float gain;
} JELLIEPreprocessParams;

// Struct for the NetworkSimulationShader
typedef struct {
    float latencyMs;
    float packetLossPercentage;
} NetworkSimulationParams;

// Struct for the PNBTRReconstructionShader
typedef struct {
    float mixLevel;
    float sineFrequency;
    bool applySineTest;
} PNBTRReconstructionParams;

#endif /* ShaderTypes_h */ 