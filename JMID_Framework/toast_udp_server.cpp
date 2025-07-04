#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace TOAST;

class UDPServer {
private:
    int sockfd;
    int port_;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    bool running = false;
    std::thread serverThread;
    ClockDriftArbiter arbiter;

public:
    UDPServer(int port = 8080) : port_(port), client_len(sizeof(client_addr)) {
        // Create UDP socket
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create UDP socket");
        }

        // Configure server address
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);

        // Bind socket
        if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to bind UDP socket");
        }

        // Initialize clock arbiter
        arbiter.initialize("udp-server", true);
    }

    ~UDPServer() {
        stop();
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    void start() {
        running = true;
        serverThread = std::thread(&UDPServer::serverLoop, this);
        std::cout << "🔥 TOAST UDP Server - Listening on port " << port_ << std::endl;
        std::cout << "==============================================" << std::endl;
    }

    void stop() {
        running = false;
        if (serverThread.joinable()) {
            serverThread.join();
        }
    }

private:
    void serverLoop() {
        char buffer[4096];
        int messagesReceived = 0;

        while (running) {
            // Receive UDP packet
            ssize_t bytes_received = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                                            (struct sockaddr*)&client_addr, &client_len);
            
            if (bytes_received > 0) {
                messagesReceived++;
                buffer[bytes_received] = '\0';
                
                std::cout << "📦 [" << messagesReceived << "] UDP packet from " 
                         << inet_ntoa(client_addr.sin_addr) << ":" 
                         << ntohs(client_addr.sin_port) << std::endl;
                std::cout << "    Data: " << std::string(buffer, bytes_received) << std::endl;

                // Echo back the message
                std::string response = "ACK: " + std::string(buffer, bytes_received);
                sendto(sockfd, response.c_str(), response.length(), 0,
                      (struct sockaddr*)&client_addr, client_len);
                
                std::cout << "📤 Sent ACK back to client" << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    int port = 8081; // Default to 8081 for UDP to avoid conflict with TCP
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    std::cout << "🌐 This UDP server can be reached at: 192.168.1.188:" << port << std::endl;
    
    try {
        UDPServer server(port);
        server.start();

        std::cout << "🎯 UDP server running on port " << port << ". Press Enter to stop..." << std::endl;
        std::cin.get();
        
        server.stop();
        std::cout << "✅ UDP server stopped." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
