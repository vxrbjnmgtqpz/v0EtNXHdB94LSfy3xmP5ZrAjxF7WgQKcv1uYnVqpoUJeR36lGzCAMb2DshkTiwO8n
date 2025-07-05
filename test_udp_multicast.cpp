#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int main() {
    std::cout << "🔍 Testing UDP Multicast 239.255.77.77:7777..." << std::endl;
    
    // Create socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cout << "❌ Socket creation failed" << std::endl;
        return -1;
    }
    
    // Enable reuse
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Setup multicast address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(7777);
    inet_pton(AF_INET, "239.255.77.77", &addr.sin_addr);
    
    // Join multicast group
    struct ip_mreq mreq;
    mreq.imr_multiaddr = addr.sin_addr;
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        std::cout << "❌ Failed to join multicast group" << std::endl;
        close(sock);
        return -1;
    }
    
    // Bind socket
    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(7777);
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        std::cout << "❌ Bind failed" << std::endl;
        close(sock);
        return -1;
    }
    
    std::cout << "✅ UDP Multicast socket setup successful!" << std::endl;
    std::cout << "📡 Sending test message..." << std::endl;
    
    // Send test message
    const char* message = "TOAST_TEST_MESSAGE";
    sendto(sock, message, strlen(message), 0, (struct sockaddr*)&addr, sizeof(addr));
    
    std::cout << "📨 Test message sent to multicast group" << std::endl;
    std::cout << "🎧 Listening for 3 seconds..." << std::endl;
    
    // Try to receive (with timeout)
    fd_set readfds;
    struct timeval timeout;
    timeout.tv_sec = 3;
    timeout.tv_usec = 0;
    
    FD_ZERO(&readfds);
    FD_SET(sock, &readfds);
    
    int result = select(sock + 1, &readfds, NULL, NULL, &timeout);
    if (result > 0) {
        char buffer[1024];
        struct sockaddr_in sender;
        socklen_t sender_len = sizeof(sender);
        
        ssize_t received = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                                  (struct sockaddr*)&sender, &sender_len);
        if (received > 0) {
            buffer[received] = '\0';
            std::cout << "📩 Received: " << buffer << std::endl;
            std::cout << "✅ UDP Multicast is working!" << std::endl;
        }
    } else {
        std::cout << "⏰ No messages received (timeout)" << std::endl;
        std::cout << "ℹ️  This is normal if only one instance is running" << std::endl;
    }
    
    close(sock);
    std::cout << "🏁 Test complete" << std::endl;
    return 0;
}
