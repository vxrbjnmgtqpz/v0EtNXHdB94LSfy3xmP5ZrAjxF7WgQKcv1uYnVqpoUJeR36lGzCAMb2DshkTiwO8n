#include <iostream>
#include "GPURenderEngine.h"

using namespace JAMNet;

int main() {
    std::cout << "=== GPU Engine Creation Test ===" << std::endl;
    
    try {
        std::cout << "Calling GPURenderEngine::create()..." << std::endl;
        auto gpuEngine = GPURenderEngine::create();
        
        if (gpuEngine) {
            std::cout << "GPU engine created successfully!" << std::endl;
        } else {
            std::cout << "GPU engine creation returned nullptr" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
    }
    
    std::cout << "Test completed" << std::endl;
    return 0;
}
