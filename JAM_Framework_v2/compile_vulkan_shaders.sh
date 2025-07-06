#!/bin/bash

# Vulkan Shader Compilation Script
# Compiles GLSL compute shaders to SPIR-V for JAMNet VulkanRenderEngine

set -e

SHADER_DIR="/Users/timothydowler/Projects/MIDIp2p/JAM_Framework_v2/shaders/vulkan"
SPIR_V_DIR="${SHADER_DIR}/spirv"

# Create SPIR-V output directory
mkdir -p "${SPIR_V_DIR}"

echo "Compiling Vulkan shaders to SPIR-V..."

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Please install Vulkan SDK."
    echo "Download from: https://vulkan.lunarg.com/sdk/home"
    exit 1
fi

# Compile audio processing shader
echo "Compiling audio_processing.comp..."
glslc "${SHADER_DIR}/audio_processing.comp" -o "${SPIR_V_DIR}/audio_processing.spv"

# Compile PNBTR prediction shader
echo "Compiling pnbtr_predict.comp..."
glslc "${SHADER_DIR}/pnbtr_predict.comp" -o "${SPIR_V_DIR}/pnbtr_predict.spv"

echo "Shader compilation complete!"
echo "SPIR-V files generated in: ${SPIR_V_DIR}"

# Validate compiled shaders
echo "Validating compiled shaders..."
spirv-val "${SPIR_V_DIR}/audio_processing.spv" && echo "✓ audio_processing.spv valid"
spirv-val "${SPIR_V_DIR}/pnbtr_predict.spv" && echo "✓ pnbtr_predict.spv valid"

echo "All shaders validated successfully!"
