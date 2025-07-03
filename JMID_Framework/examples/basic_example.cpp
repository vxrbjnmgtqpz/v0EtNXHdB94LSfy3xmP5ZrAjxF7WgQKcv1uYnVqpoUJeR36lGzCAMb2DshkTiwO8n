#include "JMIDMessage.h"
#include <iostream>

int main() {
    std::cout << "Basic JMID Example - Phase 1.1 Foundation" << std::endl;
    
    auto timestamp = std::chrono::high_resolution_clock::now();
    JMID::NoteOnMessage noteOn(1, 60, 127, timestamp);
    
    std::cout << "Generated Note On JSON: " << noteOn.toJSON() << std::endl;
    
    return 0;
}
