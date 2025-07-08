#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "PnbtrJelliePlugin.h"

/**
 * PNBTR+JELLIE VST3 Plugin Processor
 * 
 * Real VST3 plugin that wraps our proven PnbtrJellieEngine core
 * Provides DAW integration with sub-50μs performance targets
 */
class PnbtrJellieVST3Plugin : public juce::AudioProcessor
{
public:
    // Constructor/Destructor
    PnbtrJellieVST3Plugin();
    ~PnbtrJellieVST3Plugin() override;

    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // Editor management
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    // Plugin identification
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    // Program management
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    // State management
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameter management
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }
    
    // Performance monitoring
    struct PerformanceMonitor {
        std::atomic<double> current_latency_us{0.0};
        std::atomic<double> max_latency_us{0.0};
        std::atomic<float> current_snr_db{0.0f};
        std::atomic<uint64_t> frames_processed{0};
        std::atomic<uint64_t> packets_sent{0};
        std::atomic<uint64_t> packets_received{0};
    };
    
    const PerformanceMonitor& getPerformanceMonitor() const { return performance_monitor_; }

private:
    // Core engine (our proven 13.2μs performer!)
    std::unique_ptr<pnbtr_jellie::PnbtrJellieEngine> tx_engine_;
    std::unique_ptr<pnbtr_jellie::PnbtrJellieEngine> rx_engine_;
    
    // Current operating mode
    std::atomic<bool> is_tx_mode_{true};
    
    // VST3 Parameter management
    juce::AudioProcessorValueTreeState parameters;
    
    // Parameter pointers for efficiency
    std::atomic<float>* plugin_mode_param_;
    std::atomic<float>* network_port_param_;
    std::atomic<float>* network_quality_param_;
    std::atomic<float>* jellie_quality_param_;
    std::atomic<float>* pnbtr_strength_param_;
    std::atomic<float>* pnbtr_window_param_;
    std::atomic<float>* test_sine_enable_param_;
    std::atomic<float>* test_sine_freq_param_;
    std::atomic<float>* test_sine_amplitude_param_;
    std::atomic<float>* test_packet_loss_enable_param_;
    std::atomic<float>* test_packet_loss_percent_param_;
    
    // Processing buffers
    juce::AudioBuffer<float> temp_buffer_;
    std::vector<float> interleaved_input_;
    std::vector<float> interleaved_output_;
    
    // Performance monitoring
    mutable PerformanceMonitor performance_monitor_;
    juce::Time last_stats_update_;
    
    // Processing state
    bool is_prepared_ = false;
    double current_sample_rate_ = 44100.0;
    int current_block_size_ = 512;
    
    // Helper methods
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void updateEngineConfiguration();
    void updatePerformanceStats();
    void processAudioBlock(juce::AudioBuffer<float>& buffer);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PnbtrJellieVST3Plugin)
};

// Parameter IDs for VST3 automation
namespace ParameterIDs
{
    const juce::String pluginMode      {"plugin_mode"};
    const juce::String networkPort     {"network_port"};
    const juce::String networkQuality  {"network_quality"};
    const juce::String jellieQuality   {"jellie_quality"};
    const juce::String pnbtrStrength   {"pnbtr_strength"};
    const juce::String pnbtrWindow     {"pnbtr_window"};
    const juce::String testSineEnable  {"test_sine_enable"};
    const juce::String testSineFreq    {"test_sine_freq"};
    const juce::String testSineAmp     {"test_sine_amp"};
    const juce::String testPacketLossEnable {"test_packet_loss_enable"};
    const juce::String testPacketLossPercent {"test_packet_loss_percent"};
} 