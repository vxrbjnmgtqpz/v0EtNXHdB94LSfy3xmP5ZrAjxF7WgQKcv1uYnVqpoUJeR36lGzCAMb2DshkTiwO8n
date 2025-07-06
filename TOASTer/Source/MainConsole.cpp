#include <iostream>
#include <juce_core/juce_core.h>

int main()
{
    std::cout << "TOASTer Console Test - Build Success!" << std::endl;
    std::cout << "JUCE Version: " << juce::SystemStats::getJUCEVersion() << std::endl;
    return 0;
}
