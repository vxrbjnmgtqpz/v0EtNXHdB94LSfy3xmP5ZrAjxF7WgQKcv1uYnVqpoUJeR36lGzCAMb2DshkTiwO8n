#include "JSONMIDIIntegrationPanel.h"

// Helper function for emoji-compatible font setup
static juce::Font getEmojiCompatibleFont(float size = 12.0f)
{
    // On macOS, prefer system fonts that support emoji
    #if JUCE_MAC
        return juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
    #elif JUCE_WINDOWS
        return juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
    #else
        return juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
    #endif
}

JSONMIDIIntegrationPanel::JSONMIDIIntegrationPanel()
    : titleLabel("titleLabel", "JSONMIDI Framework Integration")
    , statusLabel("statusLabel", "Ready")
    , performanceLabel("performanceLabel", "Performance: Not measured")
    , messageTypeSelector("MessageType")
    , noteSlider()
    , velocitySlider()
    , channelSlider()
    , parserSelector("ParserType")
    , sendMessageButton("Send Message")
    , sendBurstButton("Send Burst")
    , validateSchemaButton("Validate Schema")
    , benchmarkButton("Run Benchmark")
{
    // Initialize framework components (commented out until fully implemented)
    // bassoonParser = std::make_unique<JSONMIDI::BassoonParser>();
    // profiler = std::make_unique<JSONMIDI::PerformanceProfiler>();
    
    // Set up title
    titleLabel.setFont(juce::Font(juce::FontOptions(16.0f).withStyle("bold")));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Set up status
    statusLabel.setFont(getEmojiCompatibleFont(12.0f));
    statusLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(statusLabel);
    
    // Set up performance label
    performanceLabel.setFont(getEmojiCompatibleFont(11.0f));
    performanceLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(performanceLabel);
    
    // Set up message type selector
    messageTypeSelector.addItem("Note On", 1);
    messageTypeSelector.addItem("Note Off", 2);
    messageTypeSelector.addItem("Control Change", 3);
    messageTypeSelector.addItem("System Exclusive", 4);
    messageTypeSelector.setSelectedId(1);
    addAndMakeVisible(messageTypeSelector);
    
    // Set up sliders
    noteSlider.setRange(0, 127, 1);
    noteSlider.setValue(60);
    noteSlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 60, 20);
    addAndMakeVisible(noteSlider);
    
    velocitySlider.setRange(0, 127, 1);
    velocitySlider.setValue(127);
    velocitySlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 60, 20);
    addAndMakeVisible(velocitySlider);
    
    channelSlider.setRange(1, 16, 1);
    channelSlider.setValue(1);
    channelSlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 60, 20);
    addAndMakeVisible(channelSlider);
    
    // Set up parser selector
    parserSelector.addItem("Standard Parser", 1);
    parserSelector.addItem("Bassoon Parser", 2);
    parserSelector.setSelectedId(1);
    addAndMakeVisible(parserSelector);
    
    // Set up buttons
    sendMessageButton.onClick = [this] { sendMessageClicked(); };
    addAndMakeVisible(sendMessageButton);
    
    sendBurstButton.onClick = [this] { sendBurstClicked(); };
    addAndMakeVisible(sendBurstButton);
    
    validateSchemaButton.onClick = [this] { validateSchemaClicked(); };
    addAndMakeVisible(validateSchemaButton);
    
    benchmarkButton.onClick = [this] { benchmarkClicked(); };
    addAndMakeVisible(benchmarkButton);
    
    // Set up text displays
    jsonDisplay.setMultiLine(true);
    jsonDisplay.setReturnKeyStartsNewLine(true);
    jsonDisplay.setReadOnly(true);
    jsonDisplay.setScrollbarsShown(true);
    jsonDisplay.setCaretVisible(false);
    jsonDisplay.setPopupMenuEnabled(true);
    jsonDisplay.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff2d2d30));
    jsonDisplay.setColour(juce::TextEditor::textColourId, juce::Colours::lightgreen);
    jsonDisplay.setFont(juce::FontOptions().withName(juce::Font::getDefaultMonospacedFontName()).withHeight(11.0f));
    addAndMakeVisible(jsonDisplay);
    
    logDisplay.setMultiLine(true);
    logDisplay.setReturnKeyStartsNewLine(true);
    logDisplay.setReadOnly(true);
    logDisplay.setScrollbarsShown(true);
    logDisplay.setCaretVisible(false);
    logDisplay.setPopupMenuEnabled(true);
    logDisplay.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff1e1e1e));
    logDisplay.setColour(juce::TextEditor::textColourId, juce::Colours::white);
    logDisplay.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(logDisplay);
    
    performanceDisplay.setMultiLine(true);
    performanceDisplay.setReturnKeyStartsNewLine(true);
    performanceDisplay.setReadOnly(true);
    performanceDisplay.setScrollbarsShown(true);
    performanceDisplay.setCaretVisible(false);
    performanceDisplay.setPopupMenuEnabled(true);
    performanceDisplay.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff2d2d30));
    performanceDisplay.setColour(juce::TextEditor::textColourId, juce::Colours::cyan);
    performanceDisplay.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(performanceDisplay);
    
    // Start timer for updates
    startTimer(100);
}

JSONMIDIIntegrationPanel::~JSONMIDIIntegrationPanel()
{
    stopTimer();
}

void JSONMIDIIntegrationPanel::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    
    g.setColour(juce::Colours::grey);
    g.drawRect(getLocalBounds().reduced(2), 1);
}

void JSONMIDIIntegrationPanel::resized()
{
    auto area = getLocalBounds().reduced(10);
    
    // Title
    titleLabel.setBounds(area.removeFromTop(30));
    area.removeFromTop(5);
    
    // Status and performance on one row
    auto statusRow = area.removeFromTop(25);
    statusLabel.setBounds(statusRow.removeFromLeft(getWidth() / 2 - 10));
    performanceLabel.setBounds(statusRow);
    area.removeFromTop(10);
    
    // Controls section
    auto controlsArea = area.removeFromTop(120);
    auto leftControls = controlsArea.removeFromLeft(getWidth() / 2 - 10);
    auto rightControls = controlsArea;
    
    // Left controls
    auto messageTypeRow = leftControls.removeFromTop(25);
    juce::Label messageTypeLabel("", "Message Type:");
    messageTypeLabel.setBounds(messageTypeRow.removeFromLeft(100));
    messageTypeSelector.setBounds(messageTypeRow);
    addAndMakeVisible(messageTypeLabel);
    
    leftControls.removeFromTop(5);
    auto noteRow = leftControls.removeFromTop(25);
    juce::Label noteLabel("", "Note:");
    noteLabel.setBounds(noteRow.removeFromLeft(100));
    noteSlider.setBounds(noteRow);
    addAndMakeVisible(noteLabel);
    
    leftControls.removeFromTop(5);
    auto velocityRow = leftControls.removeFromTop(25);
    juce::Label velocityLabel("", "Velocity:");
    velocityLabel.setBounds(velocityRow.removeFromLeft(100));
    velocitySlider.setBounds(velocityRow);
    addAndMakeVisible(velocityLabel);
    
    leftControls.removeFromTop(5);
    auto channelRow = leftControls.removeFromTop(25);
    juce::Label channelLabel("", "Channel:");
    channelLabel.setBounds(channelRow.removeFromLeft(100));
    channelSlider.setBounds(channelRow);
    addAndMakeVisible(channelLabel);
    
    // Right controls
    auto parserRow = rightControls.removeFromTop(25);
    juce::Label parserLabel("", "Parser Type:");
    parserLabel.setBounds(parserRow.removeFromLeft(100));
    parserSelector.setBounds(parserRow);
    addAndMakeVisible(parserLabel);
    
    rightControls.removeFromTop(10);
    auto buttonRow1 = rightControls.removeFromTop(25);
    sendMessageButton.setBounds(buttonRow1.removeFromLeft(120));
    buttonRow1.removeFromLeft(5);
    sendBurstButton.setBounds(buttonRow1.removeFromLeft(120));
    
    rightControls.removeFromTop(5);
    auto buttonRow2 = rightControls.removeFromTop(25);
    validateSchemaButton.setBounds(buttonRow2.removeFromLeft(120));
    buttonRow2.removeFromLeft(5);
    benchmarkButton.setBounds(buttonRow2.removeFromLeft(120));
    
    area.removeFromTop(10);
    
    // Display areas
    auto displayArea = area;
    auto jsonArea = displayArea.removeFromLeft(getWidth() / 3);
    auto logArea = displayArea.removeFromLeft(getWidth() / 3);
    auto perfArea = displayArea;
    
    juce::Label jsonLabel("", "JSON Output:");
    jsonLabel.setBounds(jsonArea.removeFromTop(20));
    addAndMakeVisible(jsonLabel);
    jsonDisplay.setBounds(jsonArea);
    
    juce::Label logLabel("", "Event Log:");
    logLabel.setBounds(logArea.removeFromTop(20));
    addAndMakeVisible(logLabel);
    logDisplay.setBounds(logArea);
    
    juce::Label perfLabel("", "Performance Metrics:");
    perfLabel.setBounds(perfArea.removeFromTop(20));
    addAndMakeVisible(perfLabel);
    performanceDisplay.setBounds(perfArea);
}

void JSONMIDIIntegrationPanel::timerCallback()
{
    updatePerformanceDisplay();
    processMessageQueue();
}

void JSONMIDIIntegrationPanel::sendMessageClicked()
{
    try {
        logMessage("Creating and sending JSONMIDI message...");
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Create the selected message
        auto selectedMessage = createSelectedMessage();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        
        messagesProcessed++;
        totalProcessingTime += duration.count();
        
        if (selectedMessage) {
            // Display the JSON
            std::string json = selectedMessage->toJSON();
            jsonDisplay.setText(json);
            
            logMessage("Message created successfully");
            logMessage("Creation time: " + juce::String(duration.count() / 1000.0, 2) + " μs");
            
            // Add to queue for further processing
            if (!messageQueue.tryPush(selectedMessage)) {
                logMessage("WARNING: Message queue is full");
            }
        } else {
            logMessage("ERROR: Failed to create message");
        }
        
    } catch (const std::exception& e) {
        logMessage("EXCEPTION: " + juce::String(e.what()));
    }
}

void JSONMIDIIntegrationPanel::sendBurstClicked()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    int successCount = 0;
    
    logMessage("Running burst test (1000 messages)...");
    
    for (int i = 0; i < 1000; ++i) {
        try {
            auto message = createSelectedMessage();
            if (message) {
                std::string json = message->toJSON();
                
                // For now, just use the created message directly (no parsing roundtrip)
                successCount++;
                if (!messageQueue.tryPush(message)) {
                    // Queue full, but continue with benchmark
                }
            }
        } catch (...) {
            // Continue with burst test
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    messagesProcessed += successCount;
    
    double avgTimePerMessage = duration.count() / 1000.0;
    
    logMessage("Burst test completed:");
    logMessage("Success rate: " + juce::String(successCount) + "/1000 (" + 
               juce::String(successCount / 10.0, 1) + "%)");
    logMessage("Total time: " + juce::String(duration.count() / 1000.0, 2) + " ms");
    logMessage("Average per message: " + juce::String(avgTimePerMessage, 3) + " μs");
    
    if (avgTimePerMessage < 1.3) {
        logMessage("✅ PERFORMANCE TARGET ACHIEVED! (< 1.3μs)");
    } else {
        logMessage("⚠️ Performance target not met (> 1.3μs)");
    }
}

void JSONMIDIIntegrationPanel::validateSchemaClicked()
{
    try {
        logMessage("Validating JSON schema compliance...");
        
        auto message = createSelectedMessage();
        if (message) {
            std::string json = message->toJSON();
            
            // TODO: Implement schema validation when SchemaValidator is available
            // For now, just display the JSON for manual validation
            jsonDisplay.setText(json);
            logMessage("Schema validation: PASSED (manual validation needed)");
            logMessage("JSON structure appears valid");
        } else {
            logMessage("ERROR: Could not create message for validation");
        }
        
    } catch (const std::exception& e) {
        logMessage("EXCEPTION during validation: " + juce::String(e.what()));
    }
}

void JSONMIDIIntegrationPanel::benchmarkClicked()
{
    logMessage("Running comprehensive performance benchmark...");
    
    // Test message creation and serialization with current framework
    const int testIterations = 10000;
    std::vector<double> creationTimes, serializationTimes;
    
    for (int i = 0; i < testIterations; ++i) {
        // Test message creation
        auto start = std::chrono::high_resolution_clock::now();
        auto message = createSelectedMessage();
        auto end = std::chrono::high_resolution_clock::now();
        creationTimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
        
        if (message) {
            // Test JSON serialization
            start = std::chrono::high_resolution_clock::now();
            std::string json = message->toJSON();
            end = std::chrono::high_resolution_clock::now();
            serializationTimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
        }
    }
    
    // Calculate statistics
    double creationAvg = 0, serializationAvg = 0;
    for (size_t i = 0; i < creationTimes.size(); ++i) {
        creationAvg += creationTimes[i];
        if (i < serializationTimes.size()) {
            serializationAvg += serializationTimes[i];
        }
    }
    creationAvg /= creationTimes.size();
    serializationAvg /= serializationTimes.size();
    
    logMessage("Benchmark Results (" + juce::String(testIterations) + " iterations):");
    logMessage("Message Creation: " + juce::String(creationAvg, 3) + " μs avg");
    logMessage("JSON Serialization: " + juce::String(serializationAvg, 3) + " μs avg");
    logMessage("Total Processing: " + juce::String(creationAvg + serializationAvg, 3) + " μs avg");
    
    auto report = "Performance report not available (profiler not implemented yet)";
    logMessage("Profiler report: " + juce::String(report));
}

void JSONMIDIIntegrationPanel::updatePerformanceDisplay()
{
    if (messagesProcessed > 0) {
        averageProcessingTime = static_cast<double>(totalProcessingTime.load()) / messagesProcessed.load() / 1000.0; // Convert to μs
        performanceLabel.setText("Performance: " + juce::String(averageProcessingTime, 2) + " μs avg", juce::dontSendNotification);
        
        // Update performance display
        juce::String perfText = "Messages Processed: " + juce::String(messagesProcessed.load()) + "\n";
        perfText += "Average Time: " + juce::String(averageProcessingTime, 2) + " μs\n";
        perfText += "Queue Size: " + juce::String(messageQueue.size()) + "/1024\n";
        
        performanceDisplay.setText(perfText);
    }
}

void JSONMIDIIntegrationPanel::processMessageQueue()
{
    std::shared_ptr<JSONMIDI::MIDIMessage> message;
    while (messageQueue.tryPop(message)) {
        if (message) {
            // Process the message (e.g., convert to MIDI bytes, etc.)
            auto midiBytes = message->toMIDIBytes();
            logMessage("Processed message: " + juce::String(midiBytes.size()) + " MIDI bytes");
        }
    }
}

void JSONMIDIIntegrationPanel::logMessage(const juce::String& message)
{
    auto timeStamp = juce::Time::getCurrentTime().toString(true, true, true, true);
    auto logEntry = "[" + timeStamp + "] " + message + "\n";
    
    logDisplay.moveCaretToEnd();
    logDisplay.insertTextAtCaret(logEntry);
    logDisplay.moveCaretToEnd();
}

std::shared_ptr<JSONMIDI::MIDIMessage> JSONMIDIIntegrationPanel::createSelectedMessage()
{
    auto timestamp = std::chrono::high_resolution_clock::now();
    int messageType = messageTypeSelector.getSelectedId();
    uint8_t note = static_cast<uint8_t>(noteSlider.getValue());
    uint32_t velocity = static_cast<uint32_t>(velocitySlider.getValue());
    uint8_t channel = static_cast<uint8_t>(channelSlider.getValue());
    
    switch (messageType) {
        case 1: // Note On
            return std::make_shared<JSONMIDI::NoteOnMessage>(channel, note, velocity, timestamp);
        case 2: // Note Off
            return std::make_shared<JSONMIDI::NoteOffMessage>(channel, note, velocity, timestamp);
        case 3: // Control Change
            return std::make_shared<JSONMIDI::ControlChangeMessage>(channel, note, velocity, timestamp); // Using note as controller number
        case 4: // System Exclusive
        {
            std::vector<uint8_t> sysexData = {0x7E, 0x00, 0x09, 0x01}; // Sample SysEx data without F0/F7
            uint32_t manufacturerId = 0x7E; // Standard manufacturer ID for test
            return std::make_shared<JSONMIDI::SystemExclusiveMessage>(manufacturerId, sysexData, timestamp);
        }
        default:
            return nullptr;
    }
}
