#include "MainComponent.h"
#include "ProfessionalTransportController.h"
#include "MIDITestPanel.h"
#include "JAMNetworkPanel.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Create the professional transport controller with full professional interface
    transportPanel = new ProfessionalTransportController();
    addAndMakeVisible(transportPanel);
    
    // Create MIDI test panel for professional MIDI monitoring and testing
    midiTestPanel = new MIDITestPanel();
    addAndMakeVisible(midiTestPanel);
    
    // Create JAM Network Panel for JAM_Framework_v2 integration
    jamNetworkPanel = new JAMNetworkPanel();
    addAndMakeVisible(jamNetworkPanel);
    
    setSize(1200, 800); // Larger size for full professional interface
    
    // Start UI update timer for smooth professional interface updates
    startTimer(50); // 20 FPS for UI updates
}

MainComponent::~MainComponent()
{
    stopTimer();
    delete transportPanel;
    delete midiTestPanel;
    delete jamNetworkPanel;
}

void MainComponent::paint(juce::Graphics& g)
{
    // Professional dark background
    g.fillAll(juce::Colour(0xff1a1a1a));
    
    g.setColour(juce::Colour(0xff3a3a3a));
    g.drawRect(getLocalBounds(), 2);
    
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(24.0f, juce::Font::bold));
    g.drawText("TOASTer - Professional MIDI/Audio Tool", 
               getLocalBounds().removeFromTop(50), 
               juce::Justification::centred, true);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromTop(50); // Title area
    
    // VERTICAL ROW LAYOUT: All panels stacked in rows
    auto mainArea = bounds.reduced(10);
    auto totalHeight = mainArea.getHeight();
    
    // ROW 1: Transport panel (full width, 33% of height)
    auto transportHeight = static_cast<int>(totalHeight * 0.33f);
    auto transportArea = mainArea.removeFromTop(transportHeight);
    transportPanel->setBounds(transportArea.reduced(5));
    
    mainArea.removeFromTop(10); // Gap
    
    // ROW 2: Media test panel (full width, 33% of height)
    auto mediaHeight = static_cast<int>(totalHeight * 0.33f);
    auto mediaArea = mainArea.removeFromTop(mediaHeight);
    midiTestPanel->setBounds(mediaArea.reduced(5));
    
    mainArea.removeFromTop(10); // Gap
    
    // ROW 3: JAM Network panel (remaining height)
    jamNetworkPanel->setBounds(mainArea.reduced(5));
}

void MainComponent::timerCallback()
{
    // Trigger repaints for smooth professional interface
    repaint();
}
