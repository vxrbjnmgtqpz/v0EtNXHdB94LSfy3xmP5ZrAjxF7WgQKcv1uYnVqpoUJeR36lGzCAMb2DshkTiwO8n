#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace TOAST;

class UDPClient {
private:
    int sockfd;
    struct sockaddr_in server_addr;
    ClockDriftArbiter arbiter;

public:
    UDPClient(const std::string& serverIP = "127.0.0.1", int port = 8080) {
        // Create UDP socket
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create UDP socket");
        }

        // Configure server address
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, serverIP.c_str(), &server_addr.sin_addr) <= 0) {
            close(sockfd);
            throw std::runtime_error("Invalid server IP address");
        }

        // Initialize clock arbiter
        arbiter.initialize("udp-client", false);
    }

    ~UDPClient() {
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    void sendMessage(const std::string& message) {
        // Send UDP packet
        ssize_t bytes_sent = sendto(sockfd, message.c_str(), message.length(), 0,
                                   (struct sockaddr*)&server_addr, sizeof(server_addr));
        
        if (bytes_sent < 0) {
            std::cerr << "❌ Failed to send UDP packet" << std::endl;
            return;
        }

        std::cout << "📤 Sent: " << message << " (" << bytes_sent << " bytes)" << std::endl;

        // Wait for response
        char buffer[4096];
        socklen_t server_len = sizeof(server_addr);
        ssize_t bytes_received = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                                        (struct sockaddr*)&server_addr, &server_len);
        
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            std::cout << "📥 Received: " << std::string(buffer, bytes_received) << std::endl;
        }
    }

    void runInteractive() {
        std::cout << "💻 TOAST UDP Client - Interactive Mode" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Commands: send_midi, send_heartbeat, send_custom <message>, quit" << std::endl;
        std::cout << std::endl;

        std::string command;
        int messageCount = 0;

        while (true) {
            std::cout << "UDP> ";
            std::getline(std::cin, command);

            if (command == "quit" || command == "exit") {
                break;
            }
            else if (command == "send_midi") {
                messageCount++;
                std::string midiMsg = "MIDI_NOTE_ON:C4:100:" + std::to_string(messageCount);
                sendMessage(midiMsg);
            }
            else if (command == "send_heartbeat") {
                auto now = std::chrono::high_resolution_clock::now();
                auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()).count();
                std::string heartbeat = "HEARTBEAT:" + std::to_string(timestamp);
                sendMessage(heartbeat);
            }
            else if (command.substr(0, 11) == "send_custom") {
                if (command.length() > 12) {
                    std::string customMsg = command.substr(12);
                    sendMessage(customMsg);
                } else {
                    std::cout << "Usage: send_custom <your message>" << std::endl;
                }
            }
            else if (command == "test_burst") {
                std::cout << "🔥 Sending 10 rapid UDP packets..." << std::endl;
                for (int i = 0; i < 10; i++) {
                    std::string msg = "BURST_MSG_" + std::to_string(i);
                    sendMessage(msg);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            else if (command == "help") {
                std::cout << "Available commands:" << std::endl;
                std::cout << "  send_midi      - Send a MIDI note message" << std::endl;
                std::cout << "  send_heartbeat - Send a heartbeat with timestamp" << std::endl;
                std::cout << "  send_custom <msg> - Send custom message" << std::endl;
                std::cout << "  test_burst     - Send 10 rapid messages" << std::endl;
                std::cout << "  quit/exit      - Exit client" << std::endl;
            }
            else {
                std::cout << "Unknown command. Type 'help' for available commands." << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    std::string serverIP = "127.0.0.1";
    if (argc > 1) {
        serverIP = argv[1];
    }

    std::cout << "💻 TOAST UDP Client - Connecting to " << serverIP << ":8080" << std::endl;

    try {
        UDPClient client(serverIP, 8080);
        client.runInteractive();
        
        std::cout << "✅ UDP client session ended." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
