#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>

// Forward declaration
class SchematicMainWindow;

/**
 * Simple launcher component that can be added to existing GUI
 * Creates and manages the SchematicMainWindow
 * Does NOT modify existing MainComponent
 */
class SchematicLauncher : public juce::Component
{
public:
    SchematicLauncher();
    ~SchematicLauncher() override;
    
    //==============================================================================
    // Component interface
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    //==============================================================================
    // Launcher controls
    void showSchematicWindow();
    void hideSchematicWindow();
    bool isSchematicWindowVisible() const;
    
private:
    //==============================================================================
    // UI components
    juce::TextButton launchButton;
    juce::Label titleLabel;
    
    //==============================================================================
    // Schematic window
    std::unique_ptr<SchematicMainWindow> schematicWindow;
    
    //==============================================================================
    // Button callbacks
    void launchButtonClicked();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SchematicLauncher)
}; 