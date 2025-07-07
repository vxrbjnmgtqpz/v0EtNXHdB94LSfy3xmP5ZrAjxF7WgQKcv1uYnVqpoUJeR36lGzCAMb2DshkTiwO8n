#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_core/juce_core.h>

class NetworkManagerStub : public juce::Thread
{
public:
    NetworkManagerStub() : juce::Thread("NetworkManager") {}
    
    ~NetworkManagerStub() override
    {
        stopThread(1000);
    }
    
    void startNetworking(int port = 7777)
    {
        serverPort = port;
        isActive = true;
        startThread();
        
        // Log networking status
        listeners.call([this](Listener& l) { 
            l.networkStatusChanged("Networking started on port " + juce::String(serverPort)); 
        });
    }
    
    void stopNetworking()
    {
        isActive = false;
        stopThread(1000);
        
        listeners.call([this](Listener& l) { 
            l.networkStatusChanged("Networking stopped"); 
        });
    }
    
    void connectToPeer(const juce::String& address, int port)
    {
        // Stub implementation - would actually create socket connection
        juce::String statusMsg = "Attempting connection to " + address + ":" + juce::String(port);
        
        listeners.call([statusMsg](Listener& l) { 
            l.networkStatusChanged(statusMsg); 
        });
        
        // Simulate connection after delay
        juce::Timer::callAfterDelay(1000, [this, address]() {
            listeners.call([address](Listener& l) { 
                l.peerConnected(address); 
            });
        });
    }
    
    void sendSyncMessage(const juce::String& message)
    {
        // Stub implementation - would send via socket
        syncMessagesSent++;
        
        listeners.call([message](Listener& l) { 
            l.syncMessageSent(message); 
        });
    }
    
    void sendTransportState(bool isPlaying, double position, double bpm)
    {
        // Stub implementation for transport sync
        juce::String stateMsg = "Transport: " + juce::String(isPlaying ? "Playing" : "Stopped") + 
                               " | Pos: " + juce::String(position, 6) + 
                               " | BPM: " + juce::String(bpm, 1);
        
        listeners.call([stateMsg](Listener& l) { 
            l.transportStateSent(stateMsg); 
        });
    }
    
    struct Listener
    {
        virtual ~Listener() = default;
        virtual void networkStatusChanged(const juce::String& status) {}
        virtual void peerConnected(const juce::String& peerAddress) {}
        virtual void peerDisconnected(const juce::String& peerAddress) {}
        virtual void syncMessageSent(const juce::String& message) {}
        virtual void syncMessageReceived(const juce::String& message) {}
        virtual void transportStateSent(const juce::String& state) {}
        virtual void transportStateReceived(bool isPlaying, double position, double bpm) {}
    };
    
    void addListener(Listener* listener) { listeners.add(listener); }
    void removeListener(Listener* listener) { listeners.remove(listener); }
    
    bool isNetworkActive() const { return isActive; }
    int getConnectedPeers() const { return connectedPeers; }
    int getSyncMessagesSent() const { return syncMessagesSent; }

private:
    void run() override
    {
        // Stub network thread - would handle socket I/O
        while (!threadShouldExit() && isActive)
        {
            wait(100); // Check every 100ms
            
            // Simulate periodic network activity
            if (++networkActivityCounter % 50 == 0) // Every 5 seconds
            {
                listeners.call([this](Listener& l) { 
                    l.networkStatusChanged("Network active - " + juce::String(connectedPeers) + " peers"); 
                });
            }
        }
    }

    std::atomic<bool> isActive{false};
    int serverPort = 7777;
    std::atomic<int> connectedPeers{0};
    std::atomic<int> syncMessagesSent{0};
    int networkActivityCounter = 0;
    
    juce::ListenerList<Listener> listeners;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NetworkManagerStub)
};
