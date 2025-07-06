/**
 * JAM Framework v2: API Elimination Demo
 * 
 * Compile and run this to see the revolutionary message-based architecture
 * in action, completely eliminating traditional framework APIs.
 */

#include "api_elimination_example.h"
#include <iostream>

int main() {
    std::cout << "🚀 JAMNet API Elimination Revolution Demo 🚀" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        jam::APIEliminationDemo demo;
        demo.runDemo();
        
        std::cout << "\n🎉 Demo completed successfully!" << std::endl;
        std::cout << "The stream IS the interface - APIs are eliminated!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
