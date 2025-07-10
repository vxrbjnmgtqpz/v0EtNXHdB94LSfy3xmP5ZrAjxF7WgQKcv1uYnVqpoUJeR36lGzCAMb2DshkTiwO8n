#include "TOASTNetworkOscilloscope.h"

// JAM Framework integration (conditionally compiled)
#ifdef JAM_FRAMEWORK_AVAILABLE
    #include "../../../TOASTer/Source/JAMFrameworkIntegration.h"
#endif

TOASTNetworkOscilloscope::TOASTNetworkOscilloscope()
{
    // Initialize default metrics
    NetworkMetrics defaultMetrics;
    defaultMetrics.session_name = "PNBTRJellieTrainer";
    defaultMetrics.multicast_address = "239.255.77.77";
    defaultMetrics.udp_port = 7777;
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        currentMetrics = defaultMetrics;
    }
    
    // Start 30 FPS updates for network visualization
    startTimer(33);
}

TOASTNetworkOscilloscope::~TOASTNetworkOscilloscope()
{
    stopTimer();
}

void TOASTNetworkOscilloscope::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Background
    g.setColour(juce::Colours::black);
    g.fillRect(bounds);
    
    // Border
    g.setColour(juce::Colours::darkgrey);
    g.drawRect(bounds, 1);
    
    // Title
    g.setColour(juce::Colours::white);
    g.setFont(14.0f);
    g.drawText("TOAST Network", headerArea, juce::Justification::centred, true);
    
    // Network status and connection info
    drawConnectionStatus(g, headerArea);
    
    // Network activity visualization
    drawPacketActivity(g, waveformArea);
    
    // Transport synchronization events
    drawTransportSync(g, waveformArea);
    
    // Performance metrics
    drawMetricsDisplay(g, metricsArea);
}

void TOASTNetworkOscilloscope::resized()
{
    auto bounds = getLocalBounds();
    
    headerArea = bounds.removeFromTop(25);
    metricsArea = bounds.removeFromBottom(40);
    waveformArea = bounds;
}

void TOASTNetworkOscilloscope::timerCallback()
{
    updateNetworkMetrics();
    repaint();
}

void TOASTNetworkOscilloscope::setJAMFrameworkIntegration(JAMFrameworkIntegration* integration)
{
#ifdef JAM_FRAMEWORK_AVAILABLE
    jamIntegration = integration;
    
    if (jamIntegration) {
        // Set up callbacks for real TOAST integration
        jamIntegration->setStatusCallback([this](const std::string& status, bool connected) {
            onNetworkStatusChanged(status, connected);
        });
        
        jamIntegration->setPerformanceCallback([this](double latency_us, double throughput_mbps, int active_peers) {
            onPerformanceUpdated(latency_us, throughput_mbps, active_peers);
        });
        
        jamIntegration->setTransportCallback([this](const std::string& command, uint64_t timestamp, double position, double bpm) {
            onTransportCommand(command, timestamp, position, bpm);
        });
        
        // Initialize with session settings
        auto metrics = currentMetrics.load();
        jamIntegration->initialize(metrics.multicast_address, metrics.udp_port, metrics.session_name);
    }
#endif
}

void TOASTNetworkOscilloscope::startNetworkVisualization()
{
#ifdef JAM_FRAMEWORK_AVAILABLE
    if (jamIntegration) {
        jamIntegration->startNetwork();
    }
#endif
}

void TOASTNetworkOscilloscope::stopNetworkVisualization()
{
#ifdef JAM_FRAMEWORK_AVAILABLE
    if (jamIntegration) {
        jamIntegration->stopNetwork();
    }
#endif
}

void TOASTNetworkOscilloscope::setSessionName(const std::string& sessionName)
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    currentMetrics.session_name = sessionName;
}

void TOASTNetworkOscilloscope::setMulticastAddress(const std::string& address, int port)
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    currentMetrics.multicast_address = address;
    currentMetrics.udp_port = port;
}

void TOASTNetworkOscilloscope::addPacketEvent(const std::string& eventType, uint32_t timestamp, bool isOutgoing)
{
    PacketEvent event;
    event.type = eventType;
    event.timestamp = timestamp;
    event.isOutgoing = isOutgoing;
    
    // Color coding for different packet types
    if (eventType == "MIDI") {
        event.colour = juce::Colours::cyan;
    } else if (eventType == "AUDIO") {
        event.colour = juce::Colours::green;
    } else if (eventType == "TRANSPORT") {
        event.colour = juce::Colours::yellow;
    } else if (eventType == "HEARTBEAT") {
        event.colour = juce::Colours::blue;
    } else {
        event.colour = juce::Colours::white;
    }
    
    event.intensity = isOutgoing ? 1.0f : 0.7f;
    
    recentPackets.push_back(event);
    
    // Keep history manageable
    if (recentPackets.size() > MAX_PACKET_HISTORY) {
        recentPackets.erase(recentPackets.begin(), recentPackets.begin() + 20);
    }
}

void TOASTNetworkOscilloscope::addTransportEvent(const std::string& command, uint64_t timestamp)
{
    TransportEvent event;
    event.command = command;
    event.timestamp = timestamp;
    
    // Color coding for transport commands
    if (command == "PLAY") {
        event.colour = juce::Colours::green;
    } else if (command == "STOP") {
        event.colour = juce::Colours::red;
    } else if (command == "PAUSE") {
        event.colour = juce::Colours::orange;
    } else {
        event.colour = juce::Colours::white;
    }
    
    recentTransportEvents.push_back(event);
    
    // Keep history manageable
    if (recentTransportEvents.size() > MAX_TRANSPORT_HISTORY) {
        recentTransportEvents.erase(recentTransportEvents.begin(), recentTransportEvents.begin() + 5);
    }
}

void TOASTNetworkOscilloscope::updateNetworkMetrics()
{
#ifdef JAM_FRAMEWORK_AVAILABLE
    if (jamIntegration) {
        auto perfMetrics = jamIntegration->getPerformanceMetrics();
        
        std::lock_guard<std::mutex> lock(metricsMutex);
        currentMetrics.latency_us = perfMetrics.latency_us;
        currentMetrics.throughput_mbps = perfMetrics.throughput_mbps;
        currentMetrics.packet_loss_rate = perfMetrics.packet_loss_rate;
        currentMetrics.active_peers = perfMetrics.active_peers;
        currentMetrics.is_connected = jamIntegration->isConnected();
    }
#endif
}

void TOASTNetworkOscilloscope::drawConnectionStatus(juce::Graphics& g, juce::Rectangle<int> area)
{
    NetworkMetrics metrics;
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        metrics = currentMetrics;
    }
    
    g.setFont(10.0f);
    
    // Connection status indicator
    juce::Colour statusColour = metrics.is_connected ? juce::Colours::green : juce::Colours::red;
    std::string statusText = metrics.is_connected ? "CONNECTED" : "DISCONNECTED";
    
    g.setColour(statusColour);
    g.fillEllipse(area.getX() + 5, area.getY() + 8, 8, 8);
    
    g.setColour(juce::Colours::white);
    g.drawText(statusText, area.getX() + 18, area.getY(), 80, area.getHeight(), juce::Justification::centredLeft);
    
    // Peer count
    std::string peerText = "Peers: " + std::to_string(metrics.active_peers);
    g.drawText(peerText, area.getX() + 100, area.getY(), 60, area.getHeight(), juce::Justification::centredLeft);
    
    // Session info
    std::string sessionText = metrics.session_name + " @ " + metrics.multicast_address + ":" + std::to_string(metrics.udp_port);
    g.setFont(8.0f);
    g.setColour(juce::Colours::lightblue);
    g.drawText(sessionText, area.getX() + 170, area.getY(), area.getWidth() - 170, area.getHeight(), juce::Justification::centredLeft);
}

void TOASTNetworkOscilloscope::drawPacketActivity(juce::Graphics& g, juce::Rectangle<int> area)
{
    if (recentPackets.empty()) return;
    
    uint32_t currentTime = static_cast<uint32_t>(juce::Time::getMillisecondCounter());
    float timeWindow = 5000.0f; // 5 second window
    
    for (const auto& packet : recentPackets) {
        float age = (currentTime - packet.timestamp) / timeWindow;
        if (age > 1.0f) continue; // Skip old packets
        
        float x = area.getX() + (1.0f - age) * area.getWidth();
        float y = area.getY() + (packet.isOutgoing ? area.getHeight() * 0.3f : area.getHeight() * 0.7f);
        
        float alpha = (1.0f - age) * packet.intensity;
        juce::Colour colour = packet.colour.withAlpha(alpha);
        
        g.setColour(colour);
        g.fillEllipse(x - 2, y - 2, 4, 4);
        
        // Draw packet type indicator
        if (age < 0.1f) { // Only for very recent packets
            g.setFont(8.0f);
            g.drawText(packet.type, x + 5, y - 8, 40, 16, juce::Justification::centredLeft);
        }
    }
}

void TOASTNetworkOscilloscope::drawTransportSync(juce::Graphics& g, juce::Rectangle<int> area)
{
    if (recentTransportEvents.empty()) return;
    
    uint64_t currentTime = static_cast<uint64_t>(juce::Time::getMillisecondCounter() * 1000); // microseconds
    uint64_t timeWindow = 10000000; // 10 second window in microseconds
    
    for (const auto& event : recentTransportEvents) {
        uint64_t age = currentTime - event.timestamp;
        if (age > timeWindow) continue;
        
        float normalizedAge = static_cast<float>(age) / static_cast<float>(timeWindow);
        float x = area.getX() + (1.0f - normalizedAge) * area.getWidth();
        float y = area.getY() + area.getHeight() * 0.1f;
        
        g.setColour(event.colour.withAlpha(1.0f - normalizedAge));
        g.drawLine(x, y, x, y + area.getHeight() * 0.8f, 2.0f);
        
        // Draw command label
        if (normalizedAge < 0.2f) {
            g.setFont(10.0f);
            g.drawText(event.command, x + 3, y, 50, 15, juce::Justification::centredLeft);
        }
    }
}

void TOASTNetworkOscilloscope::drawMetricsDisplay(juce::Graphics& g, juce::Rectangle<int> area)
{
    NetworkMetrics metrics;
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        metrics = currentMetrics;
    }
    
    g.setFont(9.0f);
    g.setColour(juce::Colours::lightgrey);
    
    std::string latencyText = "Latency: " + std::to_string(static_cast<int>(metrics.latency_us)) + "μs";
    std::string lossText = "Loss: " + std::to_string(static_cast<int>(metrics.packet_loss_rate * 100)) + "%";
    std::string jitterText = "Jitter: " + std::to_string(static_cast<int>(metrics.jitter_ms)) + "ms";
    std::string throughputText = "Throughput: " + std::to_string(static_cast<int>(metrics.throughput_mbps)) + "Mbps";
    
    int columnWidth = area.getWidth() / 4;
    
    g.drawText(latencyText, area.getX(), area.getY(), columnWidth, area.getHeight(), juce::Justification::centred);
    g.drawText(lossText, area.getX() + columnWidth, area.getY(), columnWidth, area.getHeight(), juce::Justification::centred);
    g.drawText(jitterText, area.getX() + columnWidth * 2, area.getY(), columnWidth, area.getHeight(), juce::Justification::centred);
    g.drawText(throughputText, area.getX() + columnWidth * 3, area.getY(), columnWidth, area.getHeight(), juce::Justification::centred);
}

// Callback handlers
void TOASTNetworkOscilloscope::onNetworkStatusChanged(const std::string& status, bool connected)
{
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        currentMetrics.is_connected = connected;
    }
    
    // Add a packet event to visualize status change
    addPacketEvent("STATUS", static_cast<uint32_t>(juce::Time::getMillisecondCounter()), false);
}

void TOASTNetworkOscilloscope::onPerformanceUpdated(double latency_us, double throughput_mbps, int active_peers)
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    currentMetrics.latency_us = latency_us;
    currentMetrics.throughput_mbps = throughput_mbps;
    currentMetrics.active_peers = active_peers;
}

void TOASTNetworkOscilloscope::onTransportCommand(const std::string& command, uint64_t timestamp, double position, double bpm)
{
    addTransportEvent(command, timestamp);
    
    // Add packet event for transport sync
    addPacketEvent("TRANSPORT", static_cast<uint32_t>(timestamp / 1000), false);
} 