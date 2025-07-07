# 🤝 JAMNet Collaboration Quick Start

## For Setting Up on a New Computer/VS Code Account

### 1. **Quick Setup**
```bash
# Clone the repository
git clone https://github.com/vxrbjnmgtqpz/v0EtNXHdB94LSfy3xmP5ZrAjxF7WgQKcv1uYnVqpoUJeR36lGzCAMb2DshkTiwO8n.git
cd v0EtNXHdB94LSfy3xmP5ZrAjxF7WgQKcv1uYnVqpoUJeR36lGzCAMb2DshkTiwO8n

# Run the automated setup script
./setup-collaboration.sh
```

### 2. **Daily Sync Workflow**
```bash
# Fetch latest changes
git fetch origin
git checkout main
git pull origin main

# Create your working branch
git checkout -b feature/your-work-description

# When ready to share
git push -u origin feature/your-work-description
```

### 3. **Key Project Info**
- **Architecture**: GPU-dominant (not GPU-accelerated)
- **Networking**: Pure UDP, no TCP/HTTP
- **Data Format**: JSONL streaming
- **Build System**: CMake + Make
- **Primary Frameworks**: JAM_Framework_v2, TOASTer, JDAT_Framework

### 4. **Important Directories**
```
JAMNet/
├── JAM_Framework_v2/       # Core UDP/GPU-native framework
├── TOASTer/               # Transport layer protocols  
├── JDAT_Framework/        # Data analysis tools
├── JVID_Framework/        # Video processing
├── Documentation/         # Full project documentation
└── setup-collaboration.sh # Automated setup script
```

### 5. **For Detailed Setup Instructions**
See the complete collaboration guide in your other VS Code instance or check the Documentation/ folder.

---
**Ready to contribute to the GPU-dominant networking revolution! 🚀**
