#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace FontUtils {
    /**
     * Get a clean, emoji-free font for TOASTer UI
     * Uses system fonts that render consistently without artifacts
     */
    inline juce::Font getCleanFont(float size = 12.0f, bool bold = false) {
        #if JUCE_MAC
            // Use SF Pro without emoji fallback
            auto options = juce::FontOptions()
                .withName("SF Pro Text")
                .withHeight(size);
            if (bold) {
                options = options.withStyle("Bold");
            }
            return juce::Font(options);
        #elif JUCE_WINDOWS
            // Use Segoe UI without emoji
            auto options = juce::FontOptions()
                .withName("Segoe UI")
                .withHeight(size);
            if (bold) {
                options = options.withStyle("Bold");
            }
            return juce::Font(options);
        #else
            // Use default system font
            auto options = juce::FontOptions()
                .withHeight(size);
            if (bold) {
                options = options.withStyle("Bold");
            }
            return juce::Font(options);
        #endif
    }
    
    /**
     * Get a monospace font for code/data display
     */
    inline juce::Font getMonospaceFont(float size = 12.0f) {
        #if JUCE_MAC
            return juce::Font(juce::FontOptions()
                .withName("SF Mono")
                .withHeight(size));
        #elif JUCE_WINDOWS
            return juce::Font(juce::FontOptions()
                .withName("Consolas")
                .withHeight(size));
        #else
            return juce::Font(juce::FontOptions()
                .withName("monospace")
                .withHeight(size));
        #endif
    }
}
