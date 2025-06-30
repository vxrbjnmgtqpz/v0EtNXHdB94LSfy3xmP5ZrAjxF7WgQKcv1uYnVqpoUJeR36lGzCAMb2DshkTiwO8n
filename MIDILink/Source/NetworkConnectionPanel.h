#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
class NetworkConnectionPanel : public juce::Component
{
public:
    NetworkConnectionPanel();
    ~NetworkConnectionPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void connectButtonClicked();
    void disconnectButtonClicked();
    
    juce::Label titleLabel;
    juce::TextEditor ipAddressEditor;
    juce::TextEditor portEditor;
    juce::TextButton connectButton;
    juce::TextButton disconnectButton;
    juce::Label statusLabel;
    
    bool isConnected = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NetworkConnectionPanel)
};