#include "SchematicLauncher.h"
#include "SchematicMainWindow.h"

//==============================================================================
SchematicLauncher::SchematicLauncher()
{
    // Set up title
    titleLabel.setText("Schematic View", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(juce::FontOptions(14.0f, juce::Font::bold)));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Set up launch button
    launchButton.setButtonText("Open Schematic Window");
    launchButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkblue);
    launchButton.onClick = [this]() { launchButtonClicked(); };
    addAndMakeVisible(launchButton);
    
    // Set initial size
    setSize(200, 80);
}

SchematicLauncher::~SchematicLauncher()
{
    // Unique pointer will clean up automatically
}

//==============================================================================
void SchematicLauncher::paint(juce::Graphics& g)
{
    // Draw background
    g.fillAll(juce::Colours::darkgrey.darker());
    g.setColour(juce::Colours::grey);
    g.drawRect(getLocalBounds(), 1);
}

void SchematicLauncher::resized()
{
    auto bounds = getLocalBounds().reduced(5);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5); // Small gap
    launchButton.setBounds(bounds);
}

//==============================================================================
void SchematicLauncher::showSchematicWindow()
{
    if (!schematicWindow) {
        schematicWindow = std::make_unique<SchematicMainWindow>();
    }
    
    if (schematicWindow) {
        schematicWindow->showWindow();
        launchButton.setButtonText("Hide Schematic Window");
    }
}

void SchematicLauncher::hideSchematicWindow()
{
    if (schematicWindow) {
        schematicWindow->hideWindow();
        launchButton.setButtonText("Open Schematic Window");
    }
}

bool SchematicLauncher::isSchematicWindowVisible() const
{
    return schematicWindow && schematicWindow->isVisible();
}

//==============================================================================
void SchematicLauncher::launchButtonClicked()
{
    if (isSchematicWindowVisible()) {
        hideSchematicWindow();
    } else {
        showSchematicWindow();
    }
} 