#include "JackAudioBackend.h"

#ifdef JAM_ENABLE_JACK
#include <iostream>
#include <cstring>
#include <chrono>
#include <sstream>

namespace JAMNet {

// Global GPU clock callback for JACK time injection
static std::function<uint64_t()> g_jackGPUClockCallback = nullptr;

JackAudioBackend::JackAudioBackend() 
    : client_(nullptr), gpuSharedMemory_(nullptr), gpuBufferSize_(0),
      useGPUMemory_(false), lastGPUTimestamp_(0), calibrationOffset_(0.0f),
      initialized_(false) {
}

JackAudioBackend::~JackAudioBackend() {
    shutdown();
}

bool JackAudioBackend::initialize(const AudioConfig& config) {
    if (initialized_) {
        return true;
    }
    
    config_ = config;
    
    // Check if JACK server is running
    if (!isJackRunning()) {
        std::cerr << "JackAudioBackend: JACK server is not running" << std::endl;
        return false;
    }
    
    // Open JACK client
    jack_status_t status;
    const char* clientName = "JAMNet";
    client_ = jack_client_open(clientName, JackNullOption, &status);
    
    if (!client_) {
        std::cerr << "JackAudioBackend: Failed to open JACK client, status: " << status << std::endl;
        return false;
    }
    
    // Set up callbacks
    jack_set_process_callback(client_, jackProcessCallback, this);
    jack_on_shutdown(client_, jackShutdownCallback, this);
    jack_set_sample_rate_callback(client_, jackSampleRateCallback, this);
    jack_set_buffer_size_callback(client_, jackBufferSizeCallback, this);
    
    // Verify sample rate matches configuration
    jack_nframes_t jackSampleRate = jack_get_sample_rate(client_);
    if (jackSampleRate != config_.sampleRate) {
        std::cout << "JackAudioBackend: JACK sample rate (" << jackSampleRate 
                  << ") differs from requested (" << config_.sampleRate << ")" << std::endl;
        config_.sampleRate = jackSampleRate; // Use JACK's sample rate
    }
    
    // Verify buffer size
    jack_nframes_t jackBufferSize = jack_get_buffer_size(client_);
    if (jackBufferSize != config_.bufferSize) {
        std::cout << "JackAudioBackend: JACK buffer size (" << jackBufferSize
                  << ") differs from requested (" << config_.bufferSize << ")" << std::endl;
        config_.bufferSize = jackBufferSize; // Use JACK's buffer size
    }
    
    // Create output ports
    if (!createPorts()) {
        jack_client_close(client_);
        client_ = nullptr;
        return false;
    }
    
    // Activate the client
    if (jack_activate(client_) != 0) {
        std::cerr << "JackAudioBackend: Failed to activate JACK client" << std::endl;
        destroyPorts();
        jack_client_close(client_);
        client_ = nullptr;
        return false;
    }
    
    // Connect to system ports if requested
    if (config_.deviceName == "default" || config_.deviceName == "system") {
        connectToSystemPorts();
    }
    
    initialized_ = true;
    
    std::cout << "JackAudioBackend: Initialized successfully" << std::endl;
    std::cout << "  Client: " << jack_get_client_name(client_) << std::endl;
    std::cout << "  Sample Rate: " << config_.sampleRate << "Hz" << std::endl;
    std::cout << "  Buffer Size: " << config_.bufferSize << " frames" << std::endl;
    std::cout << "  Channels: " << config_.channels << std::endl;
    std::cout << "  GPU Memory: " << (useGPUMemory_ ? "Enabled" : "Disabled") << std::endl;
    
    return true;
}

void JackAudioBackend::shutdown() {
    if (!initialized_) {
        return;
    }
    
    if (client_) {
        jack_deactivate(client_);
        destroyPorts();
        jack_client_close(client_);
        client_ = nullptr;
    }
    
    // Clear GPU memory references
    gpuSharedMemory_ = nullptr;
    gpuBufferSize_ = 0;
    useGPUMemory_ = false;
    
    // Clear external clock callback
    externalClockCallback_ = nullptr;
    g_jackGPUClockCallback = nullptr;
    
    initialized_ = false;
    
    std::cout << "JackAudioBackend: Shutdown complete" << std::endl;
}

std::string JackAudioBackend::getName() const {
    if (client_) {
        return std::string("JACK: ") + jack_get_client_name(client_);
    }
    return "JACK: Not Connected";
}

bool JackAudioBackend::createPorts() {
    outputPorts_.clear();
    outputPorts_.reserve(config_.channels);
    
    for (uint32_t i = 0; i < config_.channels; ++i) {
        std::string portName = "output_" + std::to_string(i + 1);
        jack_port_t* port = jack_port_register(client_, portName.c_str(),
                                              JACK_DEFAULT_AUDIO_TYPE,
                                              JackPortIsOutput, 0);
        if (!port) {
            std::cerr << "JackAudioBackend: Failed to register port " << portName << std::endl;
            destroyPorts();
            return false;
        }
        outputPorts_.push_back(port);
    }
    
    return true;
}

void JackAudioBackend::destroyPorts() {
    for (auto port : outputPorts_) {
        if (port) {
            jack_port_unregister(client_, port);
        }
    }
    outputPorts_.clear();
}

void JackAudioBackend::pushAudio(const float* data, uint32_t numFrames, uint64_t timestampNs) {
    // For JACK, audio is pulled by the process callback, not pushed
    // This method could be used to queue data for the next process cycle
    std::lock_guard<std::mutex> lock(callbackMutex_);
    lastGPUTimestamp_ = timestampNs;
}

void JackAudioBackend::setProcessCallback(std::function<void(float*, uint32_t, uint64_t)> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    processCallback_ = callback;
}

uint64_t JackAudioBackend::getCurrentTimeNs() {
    if (externalClockCallback_) {
        return externalClockCallback_();
    }
    
    // Fallback to JACK time converted to nanoseconds
    jack_time_t jackTime = jack_get_time();
    return static_cast<uint64_t>(jackTime) * 1000; // Convert microseconds to nanoseconds
}

float JackAudioBackend::getActualLatencyMs() const {
    if (!client_) {
        return 0.0f;
    }
    
    // Calculate latency based on buffer size and sample rate
    float bufferLatencyMs = (float(config_.bufferSize) / float(config_.sampleRate)) * 1000.0f;
    
    // Add estimated system latency (could be more sophisticated)
    float systemLatencyMs = 1.0f; // Conservative estimate
    
    return bufferLatencyMs + systemLatencyMs + calibrationOffset_;
}

bool JackAudioBackend::enableGPUMemoryMode(void* sharedBuffer, size_t bufferSize) {
    if (!sharedBuffer || bufferSize == 0) {
        useGPUMemory_ = false;
        return false;
    }
    
    gpuSharedMemory_ = sharedBuffer;
    gpuBufferSize_ = bufferSize;
    useGPUMemory_ = true;
    
    std::cout << "JackAudioBackend: GPU memory mode enabled, buffer size: " 
              << bufferSize << " bytes" << std::endl;
    
    return true;
}

void JackAudioBackend::setExternalClock(std::function<uint64_t()> clockCallback) {
    externalClockCallback_ = clockCallback;
    g_jackGPUClockCallback = clockCallback;
    
    std::cout << "JackAudioBackend: External GPU clock " 
              << (clockCallback ? "enabled" : "disabled") << std::endl;
}

jack_nframes_t JackAudioBackend::getCurrentBufferSize() const {
    return client_ ? jack_get_buffer_size(client_) : 0;
}

jack_nframes_t JackAudioBackend::getCurrentSampleRate() const {
    return client_ ? jack_get_sample_rate(client_) : 0;
}

bool JackAudioBackend::connectToSystemPorts() {
    if (!client_ || outputPorts_.empty()) {
        return false;
    }
    
    // Get system playback ports
    const char** systemPorts = jack_get_ports(client_, nullptr, JACK_DEFAULT_AUDIO_TYPE,
                                             JackPortIsPhysical | JackPortIsInput);
    if (!systemPorts) {
        std::cout << "JackAudioBackend: No system playback ports found" << std::endl;
        return false;
    }
    
    // Connect our output ports to system ports
    bool success = true;
    for (size_t i = 0; i < outputPorts_.size() && systemPorts[i]; ++i) {
        const char* ourPort = jack_port_name(outputPorts_[i]);
        int result = jack_connect(client_, ourPort, systemPorts[i]);
        if (result != 0) {
            std::cerr << "JackAudioBackend: Failed to connect " << ourPort 
                      << " to " << systemPorts[i] << std::endl;
            success = false;
        } else {
            std::cout << "JackAudioBackend: Connected " << ourPort 
                      << " to " << systemPorts[i] << std::endl;
        }
    }
    
    jack_free(systemPorts);
    return success;
}

std::vector<std::string> JackAudioBackend::getAvailablePorts() const {
    std::vector<std::string> ports;
    
    if (!client_) {
        return ports;
    }
    
    const char** portNames = jack_get_ports(client_, nullptr, nullptr, 0);
    if (portNames) {
        for (int i = 0; portNames[i]; ++i) {
            ports.emplace_back(portNames[i]);
        }
        jack_free(portNames);
    }
    
    return ports;
}

uint64_t JackAudioBackend::getJackTimeWithGPUSync() {
    if (externalClockCallback_) {
        return externalClockCallback_();
    }
    
    // Fallback to standard JACK time
    return static_cast<uint64_t>(jack_get_time()) * 1000; // Convert to nanoseconds
}

// Static methods
bool JackAudioBackend::isJackRunning() {
    jack_client_t* testClient = jack_client_open("jamnet_test", JackNoStartServer, nullptr);
    if (testClient) {
        jack_client_close(testClient);
        return true;
    }
    return false;
}

std::string JackAudioBackend::getJackServerName() {
    // This would need to be implemented based on JACK configuration
    return "default";
}

// JACK callback implementations
int JackAudioBackend::jackProcessCallback(jack_nframes_t nframes, void* arg) {
    JackAudioBackend* backend = static_cast<JackAudioBackend*>(arg);
    
    // Get current timestamp with GPU sync
    uint64_t timestamp = backend->getJackTimeWithGPUSync();
    
    // Get output buffers
    std::vector<float*> buffers;
    buffers.reserve(backend->outputPorts_.size());
    
    for (auto port : backend->outputPorts_) {
        float* buffer = static_cast<float*>(jack_port_get_buffer(port, nframes));
        buffers.push_back(buffer);
    }
    
    // Call the process callback if set
    {
        std::lock_guard<std::mutex> lock(backend->callbackMutex_);
        if (backend->processCallback_ && !buffers.empty()) {
            // For now, we'll call the callback with the first buffer
            // In a more sophisticated implementation, we'd handle multi-channel properly
            backend->processCallback_(buffers[0], nframes, timestamp);
            
            // Copy to additional channels if needed
            for (size_t i = 1; i < buffers.size(); ++i) {
                std::memcpy(buffers[i], buffers[0], nframes * sizeof(float));
            }
        } else {
            // No callback set, output silence
            for (auto buffer : buffers) {
                std::memset(buffer, 0, nframes * sizeof(float));
            }
        }
    }
    
    return 0;
}

void JackAudioBackend::jackShutdownCallback(void* arg) {
    JackAudioBackend* backend = static_cast<JackAudioBackend*>(arg);
    std::cerr << "JackAudioBackend: JACK server shutdown!" << std::endl;
    backend->initialized_ = false;
}

int JackAudioBackend::jackSampleRateCallback(jack_nframes_t nframes, void* arg) {
    JackAudioBackend* backend = static_cast<JackAudioBackend*>(arg);
    std::cout << "JackAudioBackend: Sample rate changed to " << nframes << "Hz" << std::endl;
    backend->config_.sampleRate = nframes;
    return 0;
}

int JackAudioBackend::jackBufferSizeCallback(jack_nframes_t nframes, void* arg) {
    JackAudioBackend* backend = static_cast<JackAudioBackend*>(arg);
    std::cout << "JackAudioBackend: Buffer size changed to " << nframes << " frames" << std::endl;
    backend->config_.bufferSize = nframes;
    return 0;
}

} // namespace JAMNet

#endif // JAM_ENABLE_JACK
